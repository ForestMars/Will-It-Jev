// src/router/service.rs

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tracing::warn;
use uuid::Uuid;

use crate::legacy_tiers::LegacyTierNumbers;
use crate::pipeline::identity::{Slot, TierId, FALLBACK_TIER_NAME};
use crate::pipeline::registry::TierDirectory;
use crate::pipeline::runner::{CascadeRunner, Resolution, RunOutcome, TierTrace};
use crate::pipeline::tagger::JointTagStore;
use crate::pipeline::task::{
    Decision, Reading, String, Task, TierPolicyMap, TierReading, LABEL_JEV, LABEL_LLM,
};
use crate::tagging::{is_sampled, TagInput, TagQueue};
use crate::telemetry::{
    FinalRouting, Hashes, RouteRecord, Sampling, Tier1Outcome as Tier1Record,
    TaskDecisionRecord, TaskLabelScore, TierOutcomeRecord, Tier2Outcome as Tier2Record, Tier3Outcome as Tier3Record, Tier4Outcome as Tier4Record,
};

use super::{
    cascade::Backend,
    dispatch::{downstream_tier_name, DownstreamDispatcher},
};

pub struct AppDependencies {
    pub cascade: Arc<CascadeRunner>,
    /// All configured task contracts, used by the generic task execution API.
    pub task_profiles: Vec<Task>,
    pub route_profile: Task,
    pub tier_policies: Arc<TierPolicyMap>,
    pub dispatcher: DownstreamDispatcher,
    pub hashes: Hashes,
    pub capture_tx: mpsc::Sender<RouteRecord>,
    pub tag_queue: TagQueue,
    pub tag_profile: Option<Task>,
    pub joint_tags: JointTagStore,
    pub tag_in_request: bool,
    pub tag_sample_prob: f32,
    pub joint_tag_ttl: Duration,
    pub budget: Duration,
    /// Numbers for the numbered telemetry fields. Consulted only when a record
    /// is built; nothing upstream of telemetry sees them.
    pub legacy: Arc<LegacyTierNumbers>,
}

#[derive(Clone)]
pub struct AppState {
    cascade: Arc<CascadeRunner>,
    route_profile: Task,
    tier_policies: Arc<TierPolicyMap>,
    dispatcher: Arc<DownstreamDispatcher>,
    hashes: Arc<Hashes>,
    capture_tx: mpsc::Sender<RouteRecord>,
    tag_queue: TagQueue,
    tag_profile: Option<Task>,
    joint_tags: JointTagStore,
    tag_in_request: bool,
    tag_sample_prob: f32,
    joint_tag_ttl: Duration,
    budget: Duration,
    tiers: Arc<TierDirectory>,
    legacy: Arc<LegacyTierNumbers>,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    prompt: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    model: Option<String>,
    stream: Option<bool>,
    state: Option<Value>,
    questions: Option<Map<String, Value>>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

pub fn app(dependencies: AppDependencies) -> Router {
    let task_api = super::task_api::router(
        Arc::clone(&dependencies.cascade),
        dependencies.task_profiles,
        Arc::clone(&dependencies.tier_policies),
        dependencies.budget,
    );
    let state = AppState {
        tiers: dependencies.cascade.directory(),
        legacy: dependencies.legacy,
        cascade: dependencies.cascade,
        route_profile: dependencies.route_profile,
        tier_policies: dependencies.tier_policies,
        dispatcher: Arc::new(dependencies.dispatcher),
        hashes: Arc::new(dependencies.hashes),
        capture_tx: dependencies.capture_tx,
        tag_queue: dependencies.tag_queue,
        tag_profile: dependencies.tag_profile,
        joint_tags: dependencies.joint_tags,
        tag_in_request: dependencies.tag_in_request,
        tag_sample_prob: dependencies.tag_sample_prob,
        joint_tag_ttl: dependencies.joint_tag_ttl,
        budget: dependencies.budget,
    };

    Router::new()
        .route("/healthz", get(|| async { StatusCode::OK }))
        .route("/v1/completions", post(completion))
        .route("/v1/chat/completions", post(completion))
        .route("/v1/decisions", post(completion))
        .with_state(state)
        .nest("/v1/tasks", task_api)
}

async fn completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    if request.stream == Some(true) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "streaming_not_supported", "detail": "send stream=false or omit stream"})),
        )
            .into_response();
    }
    if request.messages.as_ref().is_some_and(Vec::is_empty) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid_request", "detail": "messages must not be empty"})),
        )
            .into_response();
    }
    if request.messages.is_some()
        && (request.prompt.is_some() || request.state.is_some() || request.questions.is_some())
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid_request", "detail": "provide messages or prompt/state/questions, not both"})),
        )
            .into_response();
    }
    if request.state.is_some() != request.questions.is_some() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "invalid_request",
                "detail": "state and questions must be provided together"
            })),
        )
            .into_response();
    }

    let route_input = match (&request.prompt, &request.state, &request.questions) {
        (Some(prompt), _, _) => Some(prompt.clone()),
        (None, Some(state), Some(questions)) => {
            Some(format!("{state}\n{}", Value::Object(questions.clone())))
        }
        (None, None, None) => request.messages.as_ref().map(|messages| {
            messages
                .iter()
                .rev()
                .find(|message| message.role == "user")
                .map(|message| message.content.clone())
                .unwrap_or_else(|| messages_to_prompt(messages))
        }),
        _ => None,
    };
    let Some(route_input) = route_input else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "invalid_request",
                "detail": "provide prompt, or both state and questions"
            })),
        )
            .into_response();
    };

    let fallback_prompt = match (&request.state, &request.questions) {
        (Some(state), Some(questions)) => format!(
            "Answer the user's typed decision request. Use the supplied state and questions and \
             return one JSON object keyed by question id. For choice questions, return the chosen \
             label; for score questions, return the score; for noul questions, return the noul \
             probability.\nState:\n{}\nQuestions:\n{}",
            state,
            Value::Object(questions.clone())
        ),
        _ => request
            .messages
            .as_ref()
            .map(|messages| messages_to_prompt(messages))
            .unwrap_or_else(|| route_input.clone()),
    };

    tracing::info!(
        chat_completion = request.messages.is_some(),
        prompt_len = route_input.len(),
        "completion request accepted"
    );

    // Mint the correlation ID before any inference. The identical ID controls
    // deterministic sampling and joins route/task records in ClickHouse.
    let request_id = Uuid::new_v4().to_string();
    let route_started = Instant::now();
    let include_joint_tag = state.tag_in_request
        && state.tag_profile.is_some()
        && is_sampled(&request_id, state.tag_sample_prob);
    let request_profiles = if include_joint_tag {
        vec![
            state.route_profile.clone(),
            state.tag_profile.clone().expect("checked above"),
        ]
    } else {
        vec![state.route_profile.clone()]
    };

    let outcome = if include_joint_tag {
        let split = Arc::clone(&state.cascade).run_split(
            route_input.clone(),
            request_profiles.clone(),
            BTreeSet::from([String::from("route")]),
            Arc::clone(&state.tier_policies),
            state.budget,
        );
        state.joint_tags.register(request_id.clone(), split.done, state.joint_tag_ttl);
        split.required.await.unwrap_or_else(|error| {
            warn!(request_id = %request_id, error = %error, "required cascade result channel closed");
            RunOutcome { tasks: Default::default(), tier_calls: Vec::new() }
        })
    } else {
        state.cascade.run(
            &route_input,
            &request_profiles,
            &state.tier_policies,
            state.budget,
        ).await
    };
    for call in &outcome.tier_calls {
        tracing::debug!(
            tier = %call.tier,
            task_count = call.tasks.len(),
            elapsed_ms = call.took.as_secs_f64() * 1000.0,
            "cascade tier completed"
        );
    }
    let (backend, mut record) =
        route_record(request_id, &route_input, &state.hashes, &state.tiers, &state.legacy, &request_profiles, &outcome, route_started);

    tracing::info!(
        request_id = %record.request_id,
        backend = %backend,
        decided_by_tier = record.cascade_decision_tier,
        "dispatching completion request to downstream backend"
    );

    match state
        .dispatcher
        .execute(
            &record.request_id,
            backend,
            &fallback_prompt,
            request.state.as_ref(),
            request.questions.as_ref(),
        )
        .await
    {
        Ok(result) => {
            let payload = if request.messages.is_some() {
                openai_chat_response(&result.payload, request.model.as_deref())
            } else {
                result.payload.clone()
            };
            let response_text = payload.to_string();
            let request_id = record.request_id.clone();
            record.final_routing.backend = downstream_tier_name(result.tier).into();
            record.final_routing.tier = result.tier;
            record.final_routing.tier_name = downstream_tier_name(result.tier).into();
            record.final_routing.latency_us += result.request_latency.as_micros();
            record.downstream_invocation_latency_ms =
                result.invocation_latency.as_secs_f64() * 1000.0;
            record.downstream_outcome = Some(result.outcome.into());
            record.downstream_tier = Some(format!("tier{}", result.tier));
            record.is_failover = result.is_failover;
            record.failure_reason = result.failure_reason;
            if result.is_failover {
                record.request_latency_failover_ms =
                    Some(result.request_latency.as_secs_f64() * 1000.0);
            } else {
                record.request_latency_primary_ms =
                    Some(result.request_latency.as_secs_f64() * 1000.0);
            }
            capture(&state.capture_tx, record);
            state.tag_queue.enqueue(TagInput {
                request_id,
                prompt: route_input.clone(),
                response: Some(response_text),
            });
            (StatusCode::OK, Json(payload)).into_response()
        }
        Err(error) => {
            let detail = error.error.to_string();
            let request_id = record.request_id.clone();
            record.final_routing.backend = "llm".into();
            record.final_routing.tier = error.tier;
            record.final_routing.tier_name = downstream_tier_name(error.tier).into();
            record.final_routing.latency_us += error.request_latency.as_micros();
            record.downstream_invocation_latency_ms =
                error.invocation_latency.as_secs_f64() * 1000.0;
            record.downstream_outcome = Some(error.outcome.into());
            record.downstream_tier = Some(format!("tier{}", error.tier));
            record.is_failover = error.is_failover;
            record.failure_reason = Some(error.failure_reason);
            if error.is_failover {
                record.request_latency_failover_ms =
                    Some(error.request_latency.as_secs_f64() * 1000.0);
            }
            capture(&state.capture_tx, record);
            state.tag_queue.enqueue(TagInput {
                request_id,
                prompt: route_input.clone(),
                response: Some(format!(
                    "{{\"error\":\"generative_provider_unavailable\",\"detail\":{detail:?}}}"
                )),
            });
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({
                    "error": "generative_provider_unavailable",
                    "detail": detail
                })),
            )
                .into_response()
        }
    }
}

fn route_record(
    request_id: String,
    prompt: &str,
    hashes: &Hashes,
    tiers: &TierDirectory,
    legacy: &LegacyTierNumbers,
    profiles: &[Task],
    outcome: &RunOutcome,
    started: Instant,
) -> (Backend, RouteRecord) {
    let route_outcome = outcome.tasks.get(&String::from("route"));
    // The numeric value is the legacy ordinal (0 for a tier without one); the
    // name is the identity. Fallback keeps its historical value of 6.
    let (backend, decision_tier, decision_tier_name, decision_tier_id) = match route_outcome.map(|task| &task.resolution) {
        Some(Resolution::Tier { tier_id, tier, reading, .. }) => (
            backend_for_label(&reading.label),
            legacy.ordinal_of(tier).unwrap_or(0),
            tier.to_string(),
            tier_id.to_string(),
        ),
        Some(Resolution::Fallback { label }) => {
            (backend_for_label(label), 6, FALLBACK_TIER_NAME.to_string(), String::new())
        }
        Some(Resolution::Deferred) | None => (Backend::Llm, 6, FALLBACK_TIER_NAME.to_string(), String::new()),
    };

    let trace = route_outcome.map(|task| task.trace.as_slice()).unwrap_or(&[]);
    for entry in trace {
        tracing::debug!(
            tier = %entry.tier,
            model_id = %entry.model_id,
            decision = ?entry.decision,
            "route task tier trace"
        );
    }
    let tier1_outcome = trace_for_ordinal(trace, legacy, 1)
        .map(tier1_record)
        .unwrap_or(Tier1Record {
            confidence: 0.0,
            decided: false,
            reason: "disabled".into(),
        });
    let tier2_outcome = trace_for_ordinal(trace, legacy, 2).map(tier2_record);
    let tier3_outcome = trace_for_ordinal(trace, legacy, 3).map(tier3_record);
    let tier4_outcome = trace_for_ordinal(trace, legacy, 4).map(tier3_record);
    let latency_us = started.elapsed().as_micros();
    let task_decisions = outcome.tasks.iter().map(|(task_id, task_outcome)| {
        let output_kind = profiles.iter()
            .find(|profile| profile.name.as_str() == task_id.as_str())
            .map(|profile| profile.output_kind.as_str())
            .unwrap_or("unknown");
        let latency_ms = outcome.tier_calls.iter()
            .filter(|call| call.tasks.iter().any(|task| task.as_str() == task_id.as_str()))
            .map(|call| call.took.as_secs_f64() * 1000.0)
            .sum();
        match &task_outcome.resolution {
            Resolution::Tier { tier_id, tier, model_id, reading } => TaskDecisionRecord {
                task_id: task_id.as_str().to_string(),
                output_kind: output_kind.to_string(),
                status: "resolved".into(),
                latency_ms,
                resolving_tier: legacy.ordinal_of(tier),
                resolving_tier_name: Some(tier.to_string()),
                resolving_tier_id: Some(tier_id.to_string()),
                model_id: Some(model_id.clone()),
                label: Some(reading.label.clone()),
                confidence: Some(reading.confidence),
                scores: reading.scores.iter().map(|(label, score)| TaskLabelScore {
                    label: label.clone(),
                    score: *score,
                }).collect(),
            },
            Resolution::Fallback { label } => TaskDecisionRecord {
                task_id: task_id.as_str().to_string(),
                output_kind: output_kind.to_string(),
                status: "fallback".into(),
                latency_ms,
                resolving_tier: None,
                resolving_tier_name: None,
                resolving_tier_id: None,
                model_id: None,
                label: Some(label.clone()),
                confidence: None,
                scores: Vec::new(),
            },
            Resolution::Deferred => TaskDecisionRecord {
                task_id: task_id.as_str().to_string(),
                output_kind: output_kind.to_string(),
                status: "deferred".into(),
                latency_ms,
                resolving_tier: None,
                resolving_tier_name: None,
                model_id: None,
                label: None,
                confidence: None,
                scores: Vec::new(),
            },
        }
    }).collect();

    let record = RouteRecord {
        request_id,
        timestamp: Utc::now(),
        prompt_raw: prompt.to_string(),
        hashes: hashes.clone(),
        tasks: task_decisions,
        tier_outcomes: tier_outcome_records(trace, tiers),
        tier1_outcome,
        tier2_outcome,
        tier3_outcome,
        tier4_outcome: tier4_outcome.map(|record| Tier4Record {
            score: record.score,
            reason: record.reason,
        }),
        cascade_decision_tier: decision_tier,
        decided_by_tier_name: decision_tier_name.clone(),
        decided_by_tier_id: decision_tier_id,
        final_routing: FinalRouting {
            backend: backend.to_string(),
            tier: decision_tier,
            tier_name: decision_tier_name,
            latency_us,
        },
        router_classification_latency_us: latency_us,
        downstream_invocation_latency_ms: 0.0,
        request_latency_primary_ms: None,
        request_latency_failover_ms: None,
        downstream_outcome: None,
        downstream_tier: None,
        is_failover: false,
        failure_reason: None,
        sampling: Sampling::default(),
    };

    (backend, record)
}

fn backend_for_label(label: &str) -> Backend {
    if label == LABEL_JEV {
        Backend::Jev
    } else {
        Backend::Llm
    }
}

/// Every tier that ran for the route task, by name, in execution order. This is
/// the complete per-tier record; it covers tiers that have no legacy number and
/// so never appear in the numbered fields below.
fn tier_outcome_records(trace: &[TierTrace], tiers: &TierDirectory) -> Vec<TierOutcomeRecord> {
    trace
        .iter()
        .map(|entry| {
            let (status, label, confidence, scores) = match &entry.reading {
                TierReading::Scored(reading) => (
                    "scored",
                    Some(reading.label.clone()),
                    Some(reading.confidence),
                    reading
                        .scores
                        .iter()
                        .map(|(label, score)| TaskLabelScore { label: label.clone(), score: *score })
                        .collect(),
                ),
                TierReading::Abstain(_) => ("abstain", None, None, Vec::new()),
                TierReading::Unsupported => ("unsupported", None, None, Vec::new()),
            };
            TierOutcomeRecord {
                tier: entry.tier.to_string(),
                tier_id: entry.tier_id.to_string(),
                position: tiers.slot(&entry.tier_id).map(Slot::index).unwrap_or_default(),
                model_id: entry.model_id.clone(),
                status: status.into(),
                decided: matches!(&entry.decision, Decision::Resolve(_)),
                reason: decision_reason(entry),
                label,
                confidence,
                scores,
            }
        })
        .collect()
}

/// The numbered telemetry fields (`tier1_outcome` .. `tier4_outcome`) describe
/// the original four tiers. Find the trace entry of whichever registered tier
/// carries that legacy number.
fn trace_for_ordinal<'a>(
    trace: &'a [TierTrace],
    legacy: &LegacyTierNumbers,
    ordinal: u8,
) -> Option<&'a TierTrace> {
    let id = legacy.name_of(ordinal)?;
    trace.iter().find(|entry| &entry.tier == id)
}

fn decision_reason(trace: &TierTrace) -> String {
    match &trace.decision {
        Decision::Resolve(_) => "resolved".into(),
        Decision::Pass { reason, .. } => (*reason).into(),
    }
}

fn decision_score(trace: &TierTrace) -> f32 {
    match &trace.decision {
        Decision::Resolve(reading) => reading.confidence,
        Decision::Pass { score, .. } => *score,
    }
}

fn tier1_record(trace: &TierTrace) -> Tier1Record {
    let confidence = match &trace.reading {
        TierReading::Scored(reading) => reading
            .scores
            .iter()
            .find(|(label, _)| label == LABEL_JEV)
            .map(|(_, score)| *score)
            .unwrap_or(reading.confidence),
        TierReading::Abstain(_) | TierReading::Unsupported => 0.0,
    };
    Tier1Record {
        confidence,
        decided: matches!(&trace.decision, Decision::Resolve(_)),
        reason: decision_reason(trace),
    }
}

fn tier2_record(trace: &TierTrace) -> Tier2Record {
    let (p_jev, p_llm, raw_probabilities) = match &trace.reading {
        TierReading::Scored(Reading { scores, .. }) => {
            let p_jev = scores
                .iter()
                .find(|(label, _)| label == LABEL_JEV)
                .map(|(_, score)| *score)
                .unwrap_or(0.0);
            let p_llm = scores
                .iter()
                .find(|(label, _)| label == LABEL_LLM)
                .map(|(_, score)| *score)
                .unwrap_or(0.0);
            (p_jev, p_llm, scores.iter().map(|(_, score)| *score).collect())
        }
        TierReading::Abstain(_) | TierReading::Unsupported => (0.0, 0.0, Vec::new()),
    };
    Tier2Record {
        p_jev,
        p_llm,
        raw_probabilities,
        reason: decision_reason(trace),
    }
}

fn tier3_record(trace: &TierTrace) -> Tier3Record {
    Tier3Record {
        score: decision_score(trace),
        reason: decision_reason(trace),
    }
}

fn capture(sender: &mpsc::Sender<RouteRecord>, record: RouteRecord) {
    match sender.try_send(record) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(record)) => {
            warn!(request_id = %record.request_id, "routing capture queue is full; dropping record");
        }
        Err(mpsc::error::TrySendError::Closed(record)) => {
            warn!(request_id = %record.request_id, "routing capture writer is unavailable; dropping record");
        }
    }
}

fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|message| format!("{}: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn openai_chat_response(payload: &Value, requested_model: Option<&str>) -> Value {
    if payload.get("choices").is_some() {
        return payload.clone();
    }
    let content = payload
        .get("response")
        .and_then(Value::as_str)
        .or_else(|| payload.pointer("/message/content").and_then(Value::as_str))
        .unwrap_or_default();
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .or(requested_model)
        .unwrap_or("unknown");
    json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4()),
        "object": "chat.completion",
        "created": Utc::now().timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_response_wraps_an_ollama_style_payload() {
        let payload = json!({"response": "hello", "model": "test-model"});
        let response = openai_chat_response(&payload, None);
        assert_eq!(response["choices"][0]["message"]["content"], "hello");
        assert_eq!(response["model"], "test-model");
    }

    #[test]
    fn existing_chat_completion_payload_is_preserved() {
        let payload = json!({"choices": [{"message": {"content": "hello"}}]});
        assert_eq!(openai_chat_response(&payload, None), payload);
    }

    // ---- tier identity in the route record ---------------------------------

    use crate::pipeline::identity::tname;
    use crate::pipeline::registry::TierRegistry;
    use crate::pipeline::runner::TaskOutcome;
    use crate::pipeline::task::{FallbackPolicy, OutputKind};
    use crate::pipeline::tier::{FakeEvaluate, TierContext, TierInstance, TierOutput};
    use async_trait::async_trait;

    struct NamedTier;

    #[async_trait]
    impl FakeEvaluate for NamedTier {
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            TierOutput::abstain_all("named", ctx.tasks, "test")
        }
    }

    /// rules, prefilter, modernbert in that execution order.
    fn directory() -> TierDirectory {
        let mut registry = TierRegistry::new();
        for name in ["rules", "prefilter", "modernbert"] {
            registry.register(Arc::new(TierInstance::fake(name, Arc::new(NamedTier)))).unwrap();
        }
        TierDirectory::from_order(&registry, &[tname("rules"), tname("prefilter"), tname("modernbert")])
    }

    /// rules is 1 and modernbert is 3; prefilter has no legacy number.
    fn legacy() -> LegacyTierNumbers {
        LegacyTierNumbers::new([(tname("rules"), 1), (tname("modernbert"), 3)]).unwrap()
    }

    fn route_profile() -> Task {
        Task {
            cascade: crate::pipeline::task::Cascade::default(),
            name: String::from("route"),
            output_kind: OutputKind::Binary,
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Label(LABEL_LLM.into()),
        }
    }

    fn test_tier_id(name: &str) -> TierId {
        directory().info_by_name(&tname(name)).expect("test tier exists").id
    }

    fn trace(tier: &str, reading: TierReading, decision: Decision) -> TierTrace {
        TierTrace { tier_id: test_tier_id(tier), tier: tname(tier), model_id: format!("m-{tier}"), reading, decision }
    }

    fn pass() -> Decision {
        Decision::Pass { reason: "below_threshold", top_guess: None, score: 0.4 }
    }

    fn record_for(resolution: Resolution, trace: Vec<TierTrace>) -> RouteRecord {
        let outcome = RunOutcome {
            tasks: [(String::from("route"), TaskOutcome { resolution, trace })].into(),
            tier_calls: Vec::new(),
        };
        let hashes = Hashes { tier1_toml: "t1".into(), tier2_bin: "t2".into() };
        route_record(
            "11111111-1111-4111-8111-111111111111".into(),
            "prompt",
            &hashes,
            &directory(),
            &legacy(),
            &[route_profile()],
            &outcome,
            Instant::now(),
        ).1
    }

    #[test]
    fn route_record_carries_legacy_number_and_tier_name() {
        let reading = Reading::new(LABEL_JEV, 0.97);
        let record = record_for(
            Resolution::Tier { tier_id: test_tier_id("modernbert"), tier: tname("modernbert"), model_id: "m".into(), reading: reading.clone() },
            vec![
                trace("rules", TierReading::Abstain("conflict"), pass()),
                trace("modernbert", TierReading::Scored(reading.clone()), Decision::Resolve(reading)),
            ],
        );
        assert_eq!(record.decided_by_tier_name, "modernbert");
        assert_eq!(record.cascade_decision_tier, 3);
        assert_eq!(record.final_routing.tier, 3);
        assert_eq!(record.tasks[0].resolving_tier, Some(3));
        assert_eq!(record.tasks[0].resolving_tier_name.as_deref(), Some("modernbert"));
        // Numbered outcomes are filled by legacy ordinal, not by position.
        assert!(!record.tier1_outcome.decided);
        assert_eq!(record.tier1_outcome.reason, "below_threshold");
        assert!(record.tier3_outcome.is_some());
        assert!(record.tier2_outcome.is_none());
    }

    #[test]
    fn a_tier_without_a_legacy_number_is_identified_by_name_only() {
        let reading = Reading::new(LABEL_JEV, 0.9);
        let record = record_for(
            Resolution::Tier { tier_id: test_tier_id("prefilter"), tier: tname("prefilter"), model_id: "m".into(), reading: reading.clone() },
            vec![trace("prefilter", TierReading::Scored(reading.clone()), Decision::Resolve(reading))],
        );
        assert_eq!(record.decided_by_tier_name, "prefilter");
        assert_eq!(record.cascade_decision_tier, 0);
        assert_eq!(record.tasks[0].resolving_tier, None);
        assert_eq!(record.tasks[0].resolving_tier_name.as_deref(), Some("prefilter"));
        // No numbered field claims the prefilter trace.
        assert!(record.tier3_outcome.is_none());
        assert_eq!(record.tier1_outcome.reason, "disabled");
    }

    #[test]
    fn fallback_is_named_and_keeps_its_historical_number() {
        let record = record_for(Resolution::Fallback { label: LABEL_LLM.into() }, Vec::new());
        assert_eq!(record.decided_by_tier_name, "fallback");
        assert_eq!(record.cascade_decision_tier, 6);
        assert_eq!(record.tasks[0].resolving_tier_name, None);
    }

    #[test]
    fn route_record_json_exposes_tier_names() {
        let record = record_for(Resolution::Fallback { label: LABEL_LLM.into() }, Vec::new());
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["decided_by_tier_name"], "fallback");
        assert_eq!(json["cascade_decision_tier"], 6);
        // Lines written before this field existed still deserialize.
        let mut legacy = json.clone();
        legacy.as_object_mut().unwrap().remove("decided_by_tier_name");
        let parsed: RouteRecord = serde_json::from_value(legacy).unwrap();
        assert_eq!(parsed.decided_by_tier_name, "");
    }

    // ---- name-keyed per-tier outcomes --------------------------------------

    #[test]
    fn tier_outcomes_list_every_tier_by_name_in_execution_order() {
        let reading = Reading { label: LABEL_JEV.into(), confidence: 0.97, scores: vec![(LABEL_JEV.into(), 0.97), (LABEL_LLM.into(), 0.03)] };
        let record = record_for(
            Resolution::Tier { tier_id: test_tier_id("modernbert"), tier: tname("modernbert"), model_id: "m".into(), reading: reading.clone() },
            vec![
                trace("rules", TierReading::Abstain("conflict"), pass()),
                trace("prefilter", TierReading::Unsupported, pass()),
                trace("modernbert", TierReading::Scored(reading.clone()), Decision::Resolve(reading)),
            ],
        );
        let names: Vec<_> = record.tier_outcomes.iter().map(|o| o.tier.as_str()).collect();
        assert_eq!(names, ["rules", "prefilter", "modernbert"]);

        let rules = &record.tier_outcomes[0];
        assert_eq!((rules.position, rules.status.as_str(), rules.decided), (0, "abstain", false));
        assert_eq!(rules.label, None);

        // A tier with no legacy ordinal is fully described here even though no
        // numbered field can hold it.
        let prefilter = &record.tier_outcomes[1];
        assert_eq!((prefilter.position, prefilter.status.as_str()), (1, "unsupported"));

        let winner = &record.tier_outcomes[2];
        assert_eq!((winner.position, winner.decided), (2, true));
        assert_eq!(winner.reason, "resolved");
        assert_eq!(winner.label.as_deref(), Some(LABEL_JEV));
        assert_eq!(winner.confidence, Some(0.97));
        assert_eq!(winner.scores.len(), 2);
        assert_eq!(winner.model_id, "m-modernbert");
    }

    #[test]
    fn a_tier_without_a_legacy_number_decides_visibly_in_tier_outcomes() {
        let reading = Reading::new(LABEL_JEV, 0.9);
        let record = record_for(
            Resolution::Tier { tier_id: test_tier_id("prefilter"), tier: tname("prefilter"), model_id: "m".into(), reading: reading.clone() },
            vec![trace("prefilter", TierReading::Scored(reading.clone()), Decision::Resolve(reading))],
        );
        assert!(record.tier3_outcome.is_none());
        assert_eq!(record.tier_outcomes.len(), 1);
        assert!(record.tier_outcomes[0].decided);
        assert_eq!(record.tier_outcomes[0].tier, "prefilter");
    }

    #[test]
    fn final_routing_carries_a_tier_name_matching_its_number() {
        let reading = Reading::new(LABEL_JEV, 0.97);
        let decided = record_for(
            Resolution::Tier { tier_id: test_tier_id("modernbert"), tier: tname("modernbert"), model_id: "m".into(), reading: reading.clone() },
            vec![trace("modernbert", TierReading::Scored(reading.clone()), Decision::Resolve(reading))],
        );
        assert_eq!((decided.final_routing.tier, decided.final_routing.tier_name.as_str()), (3, "modernbert"));

        let fallback = record_for(Resolution::Fallback { label: LABEL_LLM.into() }, Vec::new());
        assert_eq!((fallback.final_routing.tier, fallback.final_routing.tier_name.as_str()), (6, "fallback"));
        assert!(fallback.tier_outcomes.is_empty());
    }

    #[test]
    fn downstream_backends_are_named_by_what_they_run() {
        assert_eq!(downstream_tier_name(5), "jev");
        assert_eq!(downstream_tier_name(6), "llm");
    }

    #[test]
    fn records_written_before_tier_outcomes_still_deserialize() {
        let record = record_for(Resolution::Fallback { label: LABEL_LLM.into() }, Vec::new());
        let mut json = serde_json::to_value(&record).unwrap();
        json.as_object_mut().unwrap().remove("tier_outcomes");
        json["final_routing"].as_object_mut().unwrap().remove("tier_name");
        let parsed: RouteRecord = serde_json::from_value(json).unwrap();
        assert!(parsed.tier_outcomes.is_empty());
        assert_eq!(parsed.final_routing.tier_name, "");
    }
}
