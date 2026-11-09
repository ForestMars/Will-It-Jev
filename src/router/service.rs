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

use crate::pipeline::registry::{TierDirectory, FALLBACK_TIER_NAME};
use crate::pipeline::runner::{CascadeRunner, Resolution, RunOutcome, TierTrace};
use crate::pipeline::tagger::JointTagStore;
use crate::pipeline::task::{
    Decision, Reading, TaskId, TaskProfile, TierPolicyMap, TierReading, LABEL_JEV, LABEL_LLM,
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
    pub task_profiles: Vec<TaskProfile>,
    pub route_profile: TaskProfile,
    pub tier_policies: Arc<TierPolicyMap>,
    pub dispatcher: DownstreamDispatcher,
    pub hashes: Hashes,
    pub capture_tx: mpsc::Sender<RouteRecord>,
    pub tag_queue: TagQueue,
    pub tag_profile: Option<TaskProfile>,
    pub joint_tags: JointTagStore,
    pub tag_in_request: bool,
    pub tag_sample_prob: f32,
    pub joint_tag_ttl: Duration,
    pub budget: Duration,
}

#[derive(Clone)]
pub struct AppState {
    cascade: Arc<CascadeRunner>,
    route_profile: TaskProfile,
    tier_policies: Arc<TierPolicyMap>,
    dispatcher: Arc<DownstreamDispatcher>,
    hashes: Arc<Hashes>,
    capture_tx: mpsc::Sender<RouteRecord>,
    tag_queue: TagQueue,
    tag_profile: Option<TaskProfile>,
    joint_tags: JointTagStore,
    tag_in_request: bool,
    tag_sample_prob: f32,
    joint_tag_ttl: Duration,
    budget: Duration,
    tiers: Arc<TierDirectory>,
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
            BTreeSet::from([TaskId::route()]),
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
            tier_instance_id = %call.tier_instance_id,
            tier_id = %call.tier_id,
            model_id = %call.model_id,
            task_count = call.tasks.len(),
            elapsed_ms = call.took.as_secs_f64() * 1000.0,
            "cascade tier completed"
        );
    }
    let (backend, mut record) =
        route_record(request_id, &route_input, &state.hashes, &state.tiers, &request_profiles, &outcome, route_started);

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
    profiles: &[TaskProfile],
    outcome: &RunOutcome,
    started: Instant,
) -> (Backend, RouteRecord) {
    let route_outcome = outcome.tasks.get(&TaskId::route());
    // The compatibility number is the one-based index of the configured
    // instance in the ordered vector. Instance identity remains the name.
    let (backend, decision_tier, decision_tier_name) = match route_outcome.map(|task| &task.resolution) {
        Some(Resolution::Tier { tier_instance_id, reading, .. }) => (
            backend_for_label(&reading.label),
            tiers.position(tier_instance_id).map(|position| (position + 1).min(u8::MAX as usize) as u8).unwrap_or(0),
            tier_instance_id.to_string(),
        ),
        Some(Resolution::Fallback { label }) => {
            (backend_for_label(label), 6, FALLBACK_TIER_NAME.to_string())
        }
        Some(Resolution::Deferred) | None => (Backend::Llm, 6, FALLBACK_TIER_NAME.to_string()),
    };

    let trace = route_outcome.map(|task| task.trace.as_slice()).unwrap_or(&[]);
    for entry in trace {
        tracing::debug!(
            tier_instance_id = %entry.tier_instance_id,
            tier_id = %entry.tier_id,
            model_id = %entry.model_id,
            decision = ?entry.decision,
            "route task tier trace"
        );
    }
    let tier1_outcome = trace_for_ordinal(trace, tiers, 1)
        .map(tier1_record)
        .unwrap_or(Tier1Record {
            confidence: 0.0,
            decided: false,
            reason: "disabled".into(),
        });
    let tier2_outcome = trace_for_ordinal(trace, tiers, 2).map(tier2_record);
    let tier3_outcome = trace_for_ordinal(trace, tiers, 3).map(tier3_record);
    let tier4_outcome = trace_for_ordinal(trace, tiers, 4).map(tier3_record);
    let latency_us = started.elapsed().as_micros();
    let task_decisions = outcome.tasks.iter().map(|(task_id, task_outcome)| {
        let output_kind = profiles.iter()
            .find(|profile| profile.task.as_str() == task_id.as_str())
            .map(|profile| profile.output_kind.as_str())
            .unwrap_or("unknown");
        let latency_ms = outcome.tier_calls.iter()
            .filter(|call| call.tasks.iter().any(|task| task.as_str() == task_id.as_str()))
            .map(|call| call.took.as_secs_f64() * 1000.0)
            .sum();
        match &task_outcome.resolution {
            Resolution::Tier { tier_instance_id, tier_id, model_id, reading } => TaskDecisionRecord {
                task_id: task_id.as_str().to_string(),
                output_kind: output_kind.to_string(),
                status: "resolved".into(),
                latency_ms,
                resolving_tier: tiers.position(tier_instance_id).map(|position| (position + 1).min(u8::MAX as usize) as u8),
                resolving_tier_name: Some(tier_instance_id.to_string()),
                resolving_tier_instance_id: Some(tier_instance_id.to_string()),
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
                resolving_tier_instance_id: None,
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
                resolving_tier_instance_id: None,
                resolving_tier_id: None,
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
/// the complete per-tier record; it covers tiers that have no legacy ordinal and
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
                tier_instance_id: entry.tier_instance_id.to_string(),
                tier_id: entry.tier_id.to_string(),
                tier: entry.tier_instance_id.to_string(),
                position: tiers.position(&entry.tier_instance_id).unwrap_or_default(),
                legacy_ordinal: None,
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
/// carries that legacy ordinal.
fn trace_for_ordinal<'a>(
    trace: &'a [TierTrace],
    tiers: &TierDirectory,
    ordinal: u8,
) -> Option<&'a TierTrace> {
    let position = usize::from(ordinal.checked_sub(1)?);
    trace.iter().find(|entry| tiers.position(&entry.tier_instance_id) == Some(position))
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
    use crate::pipeline::registry::{iid, tid, TierInstance, TierRegistry};
    use crate::pipeline::runner::{TaskOutcome, TierTrace};
    use crate::pipeline::task::{FallbackPolicy, OutputKind};
    use crate::pipeline::tier::{Tier, TierContext, TierOutput};
    use async_trait::async_trait;

    struct TestExecutor {
        tier_id: crate::pipeline::registry::TierId,
    }

    #[async_trait]
    impl Tier for TestExecutor {
        fn id(&self) -> &crate::pipeline::registry::TierId { &self.tier_id }
        fn name(&self) -> &str { self.tier_id.as_str() }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> crate::pipeline::tier::TierOutput {
            TierOutput::abstain_all(self.tier_id.as_str(), ctx.tasks, "test")
        }
    }

    fn directory() -> TierDirectory {
        let mut registry = TierRegistry::new();
        for (instance, definition) in [
            ("rules", "rules"),
            ("encoder_blue", "grpc"),
            ("encoder_green", "grpc"),
        ] {
            let tier_id = tid(definition);
            let executor: Arc<dyn Tier> = Arc::new(TestExecutor { tier_id: tier_id.clone() });
            registry.register(TierInstance::new(iid(instance), tier_id, None, executor).unwrap()).unwrap();
        }
        registry.directory()
    }

    fn profile() -> TaskProfile {
        TaskProfile {
            task: TaskId::route(),
            output_kind: OutputKind::Binary,
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Label(LABEL_LLM.into()),
        }
    }

    fn trace(instance: &str, definition: &str, reading: TierReading, decision: Decision) -> TierTrace {
        TierTrace {
            tier_instance_id: iid(instance),
            tier_id: tid(definition),
            model_id: format!("m-{instance}"),
            reading,
            decision,
        }
    }

    fn record_for(resolution: Resolution, trace: Vec<TierTrace>) -> RouteRecord {
        let profile = profile();
        let outcome = RunOutcome {
            tasks: [(
                TaskId::route(),
                TaskOutcome { resolution, trace },
            )].into(),
            tier_calls: Vec::new(),
        };
        let (backend, record) = route_record(
            "req-1".into(),
            "prompt",
            &Hashes { tier1_toml: "a".into(), tier2_bin: "b".into() },
            &directory(),
            &[profile],
            &outcome,
            Instant::now(),
        );
        assert_eq!(backend, Backend::Jev);
        record
    }

    fn pass() -> Decision {
        Decision::Pass { reason: "below_threshold", top_guess: None, score: 0.4 }
    }

    #[test]
    fn route_record_keeps_instance_definition_and_model_distinct() {
        let reading = Reading::new(LABEL_JEV, 0.97);
        let record = record_for(
            Resolution::Tier {
                tier_instance_id: iid("encoder_green"),
                tier_id: tid("grpc"),
                model_id: "harrier-v1".into(),
                reading: reading.clone(),
            },
            vec![
                trace("rules", "rules", TierReading::Abstain("conflict"), pass()),
                trace("encoder_blue", "grpc", TierReading::Unsupported, pass()),
                trace("encoder_green", "grpc", TierReading::Scored(reading.clone()), Decision::Resolve(reading)),
            ],
        );
        assert_eq!(record.decided_by_tier_name, "encoder_green");
        assert_eq!(record.cascade_decision_tier, 3);
        assert_eq!(record.tasks[0].resolving_tier_instance_id.as_deref(), Some("encoder_green"));
        assert_eq!(record.tasks[0].resolving_tier_id.as_deref(), Some("grpc"));
        assert_eq!(record.tasks[0].model_id.as_deref(), Some("harrier-v1"));
        assert_eq!(record.tier_outcomes[1].tier_instance_id, "encoder_blue");
        assert_eq!(record.tier_outcomes[1].tier_id, "grpc");
        assert_eq!(record.tier_outcomes[2].model_id, "harrier-v1");
        assert_eq!(record.tier_outcomes[2].position, 2);
        assert_eq!(record.tier3_outcome.as_ref().unwrap().reason, "resolved");
    }

    #[test]
    fn fallback_has_no_tier_provenance() {
        let record = record_for(Resolution::Fallback { label: LABEL_LLM.into() }, Vec::new());
        assert_eq!(record.decided_by_tier_name, "fallback");
        assert_eq!(record.cascade_decision_tier, 6);
        assert_eq!(record.tasks[0].resolving_tier_instance_id, None);
        assert_eq!(record.tier_outcomes.len(), 0);
    }

    #[test]
    fn route_record_json_exposes_tier_triplet() {
        let reading = Reading::new(LABEL_JEV, 0.97);
        let record = record_for(
            Resolution::Tier {
                tier_instance_id: iid("encoder_green"),
                tier_id: tid("grpc"),
                model_id: "harrier-v1".into(),
                reading: reading.clone(),
            },
            vec![trace("encoder_green", "grpc", TierReading::Scored(reading.clone()), Decision::Resolve(reading))],
        );
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["tasks"][0]["resolving_tier_instance_id"], "encoder_green");
        assert_eq!(json["tasks"][0]["resolving_tier_id"], "grpc");
        assert_eq!(json["tasks"][0]["model_id"], "harrier-v1");
        assert_eq!(json["tier_outcomes"][0]["tier_instance_id"], "encoder_green");
        assert_eq!(json["tier_outcomes"][0]["tier_id"], "grpc");
    }
}
