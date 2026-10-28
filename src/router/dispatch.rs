use std::time::{Duration, Instant};

use serde_json::Value;
use tracing::{info, warn};

use super::cascade::Backend;
use super::tier4_runner::{DownstreamError, Tier4Runner as JevRunner};
use super::tier6_runner::Tier6Runner as LlmRunner;

pub struct DownstreamDispatcher {
    jev: Option<JevRunner>,
    llm: Option<LlmRunner>,
}

pub struct DispatchOutcome {
    pub payload: Value,
    pub tier: u8,
    pub outcome: &'static str,
    pub is_failover: bool,
    pub invocation_latency: Duration,
    pub request_latency: Duration,
    pub failure_reason: Option<String>,
}

#[derive(Debug)]
pub struct DispatchFailure {
    pub error: DownstreamError,
    pub tier: u8,
    pub outcome: &'static str,
    pub is_failover: bool,
    pub invocation_latency: Duration,
    pub request_latency: Duration,
    pub failure_reason: String,
}

impl DownstreamDispatcher {
    #[cfg(test)]
    pub fn new(
        jev_endpoint: String,
        llm_endpoint: String,
        llm_model: String,
        downstream_timeout: Duration,
        breaker_threshold: usize,
        breaker_window: Duration,
        breaker_open_duration: Duration,
    ) -> Result<Self, reqwest::Error> {
        Self::new_with_enabled(
            jev_endpoint,
            llm_endpoint,
            llm_model,
            downstream_timeout,
            breaker_threshold,
            breaker_window,
            breaker_open_duration,
            true,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_enabled(
        jev_endpoint: String,
        llm_endpoint: String,
        llm_model: String,
        downstream_timeout: Duration,
        breaker_threshold: usize,
        breaker_window: Duration,
        breaker_open_duration: Duration,
        jev_enabled: bool,
        llm_enabled: bool,
    ) -> Result<Self, reqwest::Error> {
        Ok(Self {
            jev: if jev_enabled {
                Some(JevRunner::new(
                    jev_endpoint,
                    downstream_timeout,
                    breaker_threshold,
                    breaker_window,
                    breaker_open_duration,
                )?)
            } else {
                None
            },
            llm: if llm_enabled {
                Some(LlmRunner::new(
                    llm_endpoint,
                    llm_model,
                    downstream_timeout,
                )?)
            } else {
                None
            },
        })
    }

    pub async fn execute(
        &self,
        request_id: &str,
        backend: Backend,
        prompt: &str,
        state: Option<&Value>,
        questions: Option<&serde_json::Map<String, Value>>,
    ) -> Result<DispatchOutcome, DispatchFailure> {
        if backend == Backend::Jev {
            let primary_started = Instant::now();
            let Some(jev) = self.jev.as_ref() else {
                return match self.llm.as_ref() {
                    Some(_) => {
                        self.invoke_llm(
                            request_id,
                            prompt,
                            Some("tier5_jev_disabled".into()),
                            Duration::ZERO,
                        )
                        .await
                    }
                    None => Err(disabled_failure(5, "jev_and_llm_disabled")),
                };
            };
            let jev_result = match (state, questions) {
                (Some(state), Some(questions)) => jev.invoke(state, questions).await,
                _ => {
                    return self
                        .invoke_llm(
                            request_id,
                            prompt,
                            Some("typed_decision_request_required".into()),
                            Duration::ZERO,
                        )
                        .await
                }
            };
            match jev_result {
                Ok((payload, elapsed)) => {
                    downstream_metric(request_id, "tier5", "success", false, elapsed);
                    return Ok(DispatchOutcome {
                        payload,
                        tier: 5,
                        outcome: "success",
                        is_failover: false,
                        invocation_latency: elapsed,
                        request_latency: primary_started.elapsed(),
                        failure_reason: None,
                    });
                }
                Err(error) => {
                    let (outcome, reason) = classify_error(&error);
                    downstream_metric(
                        request_id,
                        "tier5",
                        outcome,
                        false,
                        primary_started.elapsed(),
                    );
                    warn!(
                        request_id,
                        reason_code = reason,
                        error = %error,
                        "Jev Tier 5 failed; falling through to LLM Tier 6"
                    );
                    return match self.llm.as_ref() {
                        Some(_) => {
                            self.invoke_llm(
                                request_id,
                                prompt,
                                Some(reason.to_string()),
                                primary_started.elapsed(),
                            )
                            .await
                        }
                        None => Err(disabled_failure(6, "llm_disabled_after_jev_failure")),
                    };
                }
            }
        }

        if self.llm.is_some() {
            self.invoke_llm(request_id, prompt, None, Duration::ZERO)
                .await
        } else {
            Err(disabled_failure(6, "llm_disabled"))
        }
    }

    async fn invoke_llm(
        &self,
        request_id: &str,
        prompt: &str,
        failure_reason: Option<String>,
        elapsed_before: Duration,
    ) -> Result<DispatchOutcome, DispatchFailure> {
        let is_failover = failure_reason.is_some();
        let started = Instant::now();
        let Some(llm) = self.llm.as_ref() else {
            return Err(disabled_failure(6, "llm_disabled"));
        };
        match llm.invoke(prompt).await {
            Ok((payload, elapsed)) => {
                downstream_metric(request_id, "tier6", "success", is_failover, elapsed);
                Ok(DispatchOutcome {
                    payload,
                    tier: 6,
                    outcome: "success",
                    is_failover,
                    invocation_latency: elapsed,
                    request_latency: elapsed_before + started.elapsed(),
                    failure_reason,
                })
            }
            Err(error) => {
                let (outcome, reason) = classify_error(&error);
                downstream_metric(request_id, "tier6", outcome, is_failover, started.elapsed());
                warn!(
                    request_id,
                    reason_code = reason,
                    error = %error,
                    "Tier 6 generative downstream failed"
                );
                Err(DispatchFailure {
                    error,
                    tier: 6,
                    outcome,
                    is_failover,
                    invocation_latency: started.elapsed(),
                    request_latency: elapsed_before + started.elapsed(),
                    failure_reason: failure_reason.unwrap_or_else(|| reason.to_string()),
                })
            }
        }
    }
}

fn disabled_failure(tier: u8, reason: &'static str) -> DispatchFailure {
    DispatchFailure {
        error: DownstreamError::Disabled,
        tier,
        outcome: "disabled",
        is_failover: false,
        invocation_latency: Duration::ZERO,
        request_latency: Duration::ZERO,
        failure_reason: reason.to_string(),
    }
}

fn classify_error(error: &DownstreamError) -> (&'static str, &'static str) {
    match error {
        DownstreamError::Timeout => ("timeout", "timeout"),
        DownstreamError::NotEligible => ("not_eligible", "jev_not_eligible"),
        DownstreamError::CircuitOpen => ("connection_error", "circuit_open"),
        DownstreamError::Transport(_) => ("connection_error", "transport_error"),
        DownstreamError::HttpStatus(status, _) => {
            if *status == 503 {
                ("connection_error", "upstream_503")
            } else {
                ("connection_error", "upstream_http_error")
            }
        }
        DownstreamError::InvalidResponse(_) => ("connection_error", "invalid_response"),
        DownstreamError::Disabled => ("disabled", "disabled"),
    }
}

fn downstream_metric(
    request_id: &str,
    tier: &'static str,
    outcome: &'static str,
    is_failover: bool,
    elapsed: Duration,
) {
    info!(
        request_id,
        tier,
        outcome,
        is_failover = is_failover.to_string(),
        downstream_invocation_latency_ms = elapsed.as_secs_f64() * 1000.0,
        "downstream invocation"
    );
}

#[cfg(test)]
mod tests {
    use super::DownstreamDispatcher;
    use crate::router::cascade::Backend;
    use serde_json::Value;
    use std::time::Duration;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    async fn stub(status: &'static str, body: &str, delay: Duration) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let body = body.to_string();
        tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request).await;
                tokio::time::sleep(delay).await;
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{address}/")
    }

    fn dispatcher(jev: String, gen: String, timeout: Duration) -> DownstreamDispatcher {
        DownstreamDispatcher::new(
            jev,
            gen,
            "gemma3:latest".into(),
            timeout,
            2,
            Duration::from_secs(10),
            Duration::from_secs(30),
        )
        .unwrap()
    }

    fn decision_request() -> (Value, serde_json::Map<String, Value>) {
        (
            Value::String("Can Jev answer this?".into()),
            serde_json::from_value(serde_json::json!({
                "answer": {
                    "type": "choice",
                    "criteria": {"yes": "yes", "no": "no"}
                }
            }))
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn disabled_downstream_tiers_are_not_invoked() {
        let router = DownstreamDispatcher::new_with_enabled(
            "not a url".into(),
            "not a url".into(),
            "unused".into(),
            Duration::from_millis(100),
            2,
            Duration::from_secs(10),
            Duration::from_secs(30),
            false,
            false,
        )
        .unwrap();

        let result = router
            .execute("test-disabled", Backend::Llm, "prompt", None, None)
            .await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("disabled tier unexpectedly returned a response"),
        };

        assert_eq!(error.tier, 6);
        assert_eq!(error.outcome, "disabled");
        assert_eq!(error.failure_reason, "llm_disabled");
    }

    #[tokio::test]
    async fn connection_refused_falls_through_to_tier6() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let frontier = stub("200 OK", r#"{"completion":"frontier"}"#, Duration::ZERO).await;
        let router = dispatcher(
            format!("http://{address}/"),
            frontier,
            Duration::from_millis(300),
        );
        let (state, questions) = decision_request();

        let result = router
            .execute(
                "test-connection",
                Backend::Jev,
                "prompt",
                Some(&state),
                Some(&questions),
            )
            .await
            .unwrap();
        assert_eq!(result.tier, 6);
        assert!(result.is_failover);
        assert_eq!(result.failure_reason.as_deref(), Some("transport_error"));
        assert_eq!(
            result.payload["completion"],
            Value::String("frontier".into())
        );
    }

    #[tokio::test]
    async fn downstream_timeout_falls_through_and_excludes_primary_latency() {
        let jev = stub(
            "200 OK",
            r#"{"completion":"late"}"#,
            Duration::from_millis(200),
        )
        .await;
        let frontier = stub("200 OK", r#"{"completion":"frontier"}"#, Duration::ZERO).await;
        let router = dispatcher(jev, frontier, Duration::from_millis(50));
        let (state, questions) = decision_request();

        let result = router
            .execute(
                "test-timeout",
                Backend::Jev,
                "prompt",
                Some(&state),
                Some(&questions),
            )
            .await
            .unwrap();
        assert_eq!(result.tier, 6);
        assert!(result.is_failover);
        assert_eq!(result.failure_reason.as_deref(), Some("timeout"));
        assert_eq!(
            result.payload["completion"],
            Value::String("frontier".into())
        );
    }

    #[tokio::test]
    async fn laya_ineligible_decision_falls_through_to_tier6() {
        let jev = stub(
            "200 OK",
            r#"{"answers":{"__router_jev_eligible":{"choice":"no","confidence":1.0}}}"#,
            Duration::ZERO,
        )
        .await;
        let frontier = stub("200 OK", r#"{"completion":"frontier"}"#, Duration::ZERO).await;
        let router = dispatcher(jev, frontier, Duration::from_millis(300));
        let (state, questions) = decision_request();

        let result = router
            .execute(
                "test-not-eligible",
                Backend::Jev,
                "prompt",
                Some(&state),
                Some(&questions),
            )
            .await
            .unwrap();

        assert_eq!(result.tier, 6);
        assert!(result.is_failover);
        assert_eq!(result.failure_reason.as_deref(), Some("jev_not_eligible"));
        assert_eq!(
            result.payload["completion"],
            Value::String("frontier".into())
        );
    }
}
