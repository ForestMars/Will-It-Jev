use std::{sync::Arc, time::Duration};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};

use crate::tagging::{TagInput, TagQueue};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tracing::warn;

use crate::telemetry::{self, Hashes, RouteRecord};

use super::{
    cascade::{self, Backend},
    dispatch::DownstreamDispatcher,
    tier1_runner::Tier1Engine,
    tier2_runner::Tier2Runner,
    tier3_runner::Tier3Runner,
};

#[derive(Clone)]
pub struct AppState {
    tier1: Option<Arc<Tier1Engine>>,
    tier2: Option<Arc<Tier2Runner>>,
    tier3: Option<Arc<Tier3Runner>>,
    tier4: Option<Arc<Tier3Runner>>,
    dispatcher: Arc<DownstreamDispatcher>,
    hashes: Arc<Hashes>,
    capture_tx: mpsc::Sender<RouteRecord>,
    tag_queue: TagQueue,
    budget: Duration,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    prompt: Option<String>,
    state: Option<Value>,
    questions: Option<Map<String, Value>>,
}

pub fn app(
    tier1: Option<Tier1Engine>,
    tier2: Option<Tier2Runner>,
    tier3: Option<Tier3Runner>,
    tier4: Option<Tier3Runner>,
    dispatcher: DownstreamDispatcher,
    hashes: Hashes,
    capture_tx: mpsc::Sender<RouteRecord>,
    tag_queue: TagQueue,
    budget: Duration,
) -> Router {
    let state = AppState {
        tier1: tier1.map(Arc::new),
        tier2: tier2.map(Arc::new),
        tier3: tier3.map(Arc::new),
        tier4: tier4.map(Arc::new),
        dispatcher: Arc::new(dispatcher),
        hashes: Arc::new(hashes),
        capture_tx,
        tag_queue,
        budget,
    };
    Router::new()
        .route("/healthz", get(|| async { StatusCode::OK }))
        .route("/v1/completions", post(completion))
        .route("/v1/decisions", post(completion))
        .with_state(state)
}

async fn completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
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
        _ => route_input.clone(),
    };

    let (_, mut record) = cascade::route(
        telemetry::next_req_id(),
        &route_input,
        state.tier1.as_deref(),
        state.tier2.as_deref(),
        state.tier3.as_deref(),
        state.tier4.as_deref(),
        state.budget,
        &state.hashes,
    )
    .await;

    match state
        .dispatcher
        .execute(
            &record.request_id,
            if record.final_routing.backend == Backend::Jev.to_string() {
                Backend::Jev
            } else {
                Backend::Llm
            },
            &fallback_prompt,
            request.state.as_ref(),
            request.questions.as_ref(),
        )
        .await
    {
        Ok(result) => {
            let payload = result.payload.clone();
            let response_text = payload.to_string();
            let request_id = record.request_id.clone();
            record.final_routing.backend = if result.tier == 5 { "jev" } else { "llm" }.into();
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
                response: Some(format!("{{\"error\":\"generative_provider_unavailable\",\"detail\":{detail:?}}}")),
            });
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": "generative_provider_unavailable", "detail": detail })),
            )
                .into_response()
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use serde_json::Value;
    use tokio::io::AsyncWriteExt;
    use tower::ServiceExt;

    async fn stub_backend(body: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = body.to_string();
        tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                let mut bytes = [0u8; 4096];
                let _ = stream.readable().await;
                let _ = stream.try_read(&mut bytes);
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}/")
    }

    #[tokio::test]
    async fn completion_reaches_selected_backend() {
        let tier1 = Tier1Engine::new(
            crate::router::tier1_runner::Tier1Automaton::from_toml_file("config/tier1.toml")
                .unwrap(),
        )
        .unwrap();
        let tier2 = Tier2Runner::new("models/tier2.bin", 0.95).unwrap();
        let tier3 = Tier3Runner::new(
            "http://127.0.0.1:1".into(),
            0.95,
            0.7,
            Duration::from_millis(20),
        )
        .unwrap();
        let jev = stub_backend(
            r#"{"model":"laya:en","answers":{"__router_jev_eligible":{"choice":"yes","confidence":1.0,"probabilities":{"yes":1.0,"no":0.0}},"answer":{"choice":"yes","confidence":1.0,"probabilities":{"yes":1.0,"no":0.0}}}}"#,
        )
        .await;
        let frontier = stub_backend(r#"{"response":"frontier","model":"gemma3:latest"}"#).await;
        let dispatcher = DownstreamDispatcher::new(
            jev,
            frontier,
            "gemma3:latest".into(),
            Duration::from_millis(100),
            2,
            Duration::from_secs(5),
            Duration::from_secs(30),
        )
        .unwrap();
        let (_tag_tx, _tag_rx) = tokio::sync::mpsc::channel::<crate::telemetry::RouteRecord>(2);
        let (tag_input_tx, mut tag_input_rx) = mpsc::channel(1);
        let release_tagger = Arc::new(tokio::sync::Notify::new());
        let tag_dir = std::env::temp_dir().join(format!(
            "router-service-tags-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (tag_queue, tag_handle) = crate::tagging::spawn(
            std::sync::Arc::new(NoopTagger {
                input_tx: tag_input_tx,
                release: Arc::clone(&release_tagger),
            }),
            crate::tagging::TagConfig {
                base_path: tag_dir.join("task_tags.jsonl"),
                ..crate::tagging::TagConfig::default()
            },
        );
        let (tx, _rx) = mpsc::channel(2);
        let application = app(
            Some(tier1),
            Some(tier2),
            None,
            Some(tier3),
            dispatcher,
            Hashes {
                tier1_toml: String::new(),
                tier2_bin: String::new(),
            },
            tx,
            tag_queue,
            Duration::from_millis(30),
        );

        let response = tokio::time::timeout(
            Duration::from_secs(1),
            application.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"prompt":"What is 2+2?","state":"What is 2+2?","questions":{"answer":{"type":"choice","instructions":"Is 2+2 equal to 4?","criteria":{"yes":"The expression equals four.","no":"The expression does not equal four."}}}}"#,
                    ))
                    .unwrap(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["answers"]["answer"]["choice"], "yes");
        assert!(payload["answers"].get("__router_jev_eligible").is_none());

        let tag_input = tokio::time::timeout(Duration::from_secs(1), tag_input_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(tag_input.prompt.contains("What is 2+2?"));
        assert!(!tag_input.request_id.is_empty());
        assert_eq!(
            serde_json::from_str::<Value>(tag_input.response.as_deref().unwrap()).unwrap(),
            payload
        );
        release_tagger.notify_one();
        tag_handle.await.unwrap().unwrap();
        std::fs::remove_dir_all(tag_dir).unwrap();
    }

    struct NoopTagger {
        input_tx: mpsc::Sender<crate::tagging::TagInput>,
        release: Arc<tokio::sync::Notify>,
    }

    impl crate::tagging::Tagger for NoopTagger {
        fn model(&self) -> &str { "noop-tagger" }
        fn version(&self) -> &str { "test" }
        fn taxonomy_version(&self) -> &str { "test-v1" }
        fn tag(
            &self,
            input: &crate::tagging::TagInput,
        ) -> impl std::future::Future<Output = Result<crate::tagging::TaggerOutput, crate::tagging::TagError>> + Send {
            let input_tx = self.input_tx.clone();
            let input = input.clone();
            let release = Arc::clone(&self.release);
            async move {
                input_tx.send(input).await.unwrap();
                release.notified().await;
                Ok(crate::tagging::TaggerOutput {
                    verdict: crate::tagging::TaggerVerdict::Tagged {
                        primary: crate::tagging::CategoryScore {
                            category_id: "engineering".to_string(),
                            score: 1.0,
                        },
                        alternatives: vec![],
                    },
                    usage: Some(crate::tagging::TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        estimated: true,
                    }),
                })
            }
        }
    }
}
