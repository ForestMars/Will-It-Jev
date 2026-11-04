// src/router/service.rs

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

use crate::pipeline::runner::{CascadeRunner, Resolution, RunOutcome, TierTrace};
use crate::pipeline::task::{
    Decision, Reading, TaskId, TaskProfile, TierPolicyMap, TierReading, LABEL_JEV, LABEL_LLM,
};
use crate::tagging::{TagInput, TagQueue};
use crate::telemetry::{
    FinalRouting, Hashes, RouteRecord, Sampling, Tier1Outcome as Tier1Record,
    Tier2Outcome as Tier2Record, Tier3Outcome as Tier3Record, Tier4Outcome as Tier4Record,
};

use super::{cascade::Backend, dispatch::DownstreamDispatcher};

#[derive(Clone)]
pub struct AppState {
    cascade: Arc<CascadeRunner>,
    route_profile: TaskProfile,
    tier_policies: Arc<TierPolicyMap>,
    dispatcher: Arc<DownstreamDispatcher>,
    hashes: Arc<Hashes>,
    capture_tx: mpsc::Sender<RouteRecord>,
    tag_queue: TagQueue,
    budget: Duration,
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

pub fn app(
    cascade: Arc<CascadeRunner>,
    route_profile: TaskProfile,
    tier_policies: Arc<TierPolicyMap>,
    dispatcher: DownstreamDispatcher,
    hashes: Hashes,
    capture_tx: mpsc::Sender<RouteRecord>,
    tag_queue: TagQueue,
    budget: Duration,
) -> Router {
    let state = AppState {
        cascade,
        route_profile,
        tier_policies,
        dispatcher: Arc::new(dispatcher),
        hashes: Arc::new(hashes),
        capture_tx,
        tag_queue,
        budget,
    };

    Router::new()
        .route("/healthz", get(|| async { StatusCode::OK }))
        .route("/v1/completions", post(completion))
        .route("/v1/chat/completions", post(completion))
        .route("/v1/decisions", post(completion))
        .with_state(state)
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

    let route_started = Instant::now();
    let outcome = state
        .cascade
        .run(
            &route_input,
            std::slice::from_ref(&state.route_profile),
            &state.tier_policies,
            state.budget,
        )
        .await;
    for call in &outcome.tier_calls {
        tracing::debug!(
            tier = call.tier,
            task_count = call.tasks.len(),
            elapsed_ms = call.took.as_secs_f64() * 1000.0,
            "cascade tier completed"
        );
    }
    let (backend, mut record) =
        route_record(&route_input, &state.hashes, &outcome, route_started);

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
            record.final_routing.backend = if result.tier == 5 {
                "jev".into()
            } else {
                "llm".into()
            };
            record.final_routing.tier = result.tier;
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
    prompt: &str,
    hashes: &Hashes,
    outcome: &RunOutcome,
    started: Instant,
) -> (Backend, RouteRecord) {
    let request_id = Uuid::new_v4().to_string();
    let route_outcome = outcome.tasks.get(&TaskId::route());
    let (backend, decision_tier) = match route_outcome.map(|task| &task.resolution) {
        Some(Resolution::Tier { tier, reading, .. }) => {
            (backend_for_label(&reading.label), *tier)
        }
        Some(Resolution::Fallback { label }) => (backend_for_label(label), 6),
        Some(Resolution::Deferred) | None => (Backend::Llm, 6),
    };

    let trace = route_outcome.map(|task| task.trace.as_slice()).unwrap_or(&[]);
    let tier1_outcome = trace_for_tier(trace, 1)
        .map(tier1_record)
        .unwrap_or(Tier1Record {
            confidence: 0.0,
            decided: false,
            reason: "disabled".into(),
        });
    let tier2_outcome = trace_for_tier(trace, 2).map(tier2_record);
    let tier3_outcome = trace_for_tier(trace, 3).map(tier3_record);
    let tier4_outcome = trace_for_tier(trace, 4).map(tier3_record);
    let latency_us = started.elapsed().as_micros();

    let record = RouteRecord {
        request_id,
        timestamp: Utc::now(),
        prompt_raw: prompt.to_string(),
        hashes: hashes.clone(),
        tier1_outcome,
        tier2_outcome,
        tier3_outcome,
        tier4_outcome: tier4_outcome.map(|record| Tier4Record {
            score: record.score,
            reason: record.reason,
        }),
        cascade_decision_tier: decision_tier,
        final_routing: FinalRouting {
            backend: backend.to_string(),
            tier: decision_tier,
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

fn trace_for_tier(trace: &[TierTrace], tier: u8) -> Option<&TierTrace> {
    trace.iter().find(|entry| entry.tier == tier)
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
        TierReading::Abstain(_) => 0.0,
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
        TierReading::Abstain(_) => (0.0, 0.0, Vec::new()),
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
}
