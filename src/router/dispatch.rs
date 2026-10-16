use std::time::{Duration, Instant};

use serde_json::Value;
use tracing::{info, warn};

use super::cascade::Backend;
use super::tier4_runner::{DownstreamError, Tier4Runner};
use super::tier5_runner::Tier5Runner;

pub struct DownstreamDispatcher {
    tier4: Tier4Runner,
    tier5: Tier5Runner,
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
    pub fn new(
        tier4_endpoint: String,
        tier5_endpoint: String,
        downstream_timeout: Duration,
        breaker_threshold: usize,
        breaker_window: Duration,
        breaker_open_duration: Duration,
    ) -> Result<Self, reqwest::Error> {
        Ok(Self {
            tier4: Tier4Runner::new(
                tier4_endpoint,
                downstream_timeout,
                breaker_threshold,
                breaker_window,
                breaker_open_duration,
            )?,
            tier5: Tier5Runner::new(tier5_endpoint, downstream_timeout)?,
        })
    }

    pub async fn execute(
        &self,
        request_id: &str,
        backend: Backend,
        prompt: &str,
    ) -> Result<DispatchOutcome, DispatchFailure> {
        if backend == Backend::Jev {
            let primary_started = Instant::now();
            match self.tier4.invoke(prompt).await {
                Ok((payload, elapsed)) => {
                    downstream_metric(request_id, "tier4", "success", false, elapsed);
                    return Ok(DispatchOutcome {
                        payload,
                        tier: 4,
                        outcome: "success",
                        is_failover: false,
                        invocation_latency: elapsed,
                        request_latency: primary_started.elapsed(),
                        failure_reason: None,
                    });
                }
                Err(error) => {
                    let (outcome, reason) = classify_error(&error);
                    downstream_metric(request_id, "tier4", outcome, false, primary_started.elapsed());
                    warn!(
                        request_id,
                        reason_code = reason,
                        error = %error,
                        "Jev downstream failed; falling through to Tier 5"
                    );
                    return self
                        .invoke_tier5(request_id, prompt, Some(reason.to_string()), primary_started.elapsed())
                        .await;
                }
            }
        }

        self.invoke_tier5(request_id, prompt, None, Duration::ZERO).await
    }

    async fn invoke_tier5(
        &self,
        request_id: &str,
        prompt: &str,
        failure_reason: Option<String>,
        elapsed_before: Duration,
    ) -> Result<DispatchOutcome, DispatchFailure> {
        let is_failover = failure_reason.is_some();
        let started = Instant::now();
        match self.tier5.invoke(prompt).await {
            Ok((payload, elapsed)) => {
                downstream_metric(request_id, "tier5", "success", is_failover, elapsed);
                Ok(DispatchOutcome {
                    payload,
                    tier: 5,
                    outcome: "success",
                    is_failover,
                    invocation_latency: elapsed,
                    request_latency: elapsed_before + started.elapsed(),
                    failure_reason,
                })
            }
            Err(error) => {
                let (outcome, reason) = classify_error(&error);
                downstream_metric(request_id, "tier5", outcome, is_failover, started.elapsed());
                warn!(
                    request_id,
                    reason_code = reason,
                    error = %error,
                    "Tier 5 generative downstream failed"
                );
                Err(DispatchFailure {
                    error,
                    tier: 5,
                    outcome,
                    is_failover,
                    invocation_latency: started.elapsed(),
                    request_latency: elapsed_before + started.elapsed(),
                    failure_reason: failure_reason
                        .unwrap_or_else(|| reason.to_string()),
                })
            }
        }
    }
}

fn classify_error(error: &DownstreamError) -> (&'static str, &'static str) {
    match error {
        DownstreamError::Timeout => ("timeout", "timeout"),
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
            timeout,
            2,
            Duration::from_secs(10),
            Duration::from_secs(30),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn connection_refused_falls_through_to_tier5() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let frontier = stub("200 OK", r#"{"completion":"frontier"}"#, Duration::ZERO).await;
        let router = dispatcher(format!("http://{address}/"), frontier, Duration::from_millis(300));

        let result = router.execute("test-connection", Backend::Jev, "prompt").await.unwrap();
        assert_eq!(result.tier, 5);
        assert!(result.is_failover);
        assert_eq!(result.failure_reason.as_deref(), Some("transport_error"));
        assert_eq!(result.payload["completion"], Value::String("frontier".into()));
    }

    #[tokio::test]
    async fn downstream_timeout_falls_through_and_excludes_primary_latency() {
        let jev = stub("200 OK", r#"{"completion":"late"}"#, Duration::from_millis(200)).await;
        let frontier = stub("200 OK", r#"{"completion":"frontier"}"#, Duration::ZERO).await;
        let router = dispatcher(jev, frontier, Duration::from_millis(50));

        let result = router.execute("test-timeout", Backend::Jev, "prompt").await.unwrap();
        assert_eq!(result.tier, 5);
        assert!(result.is_failover);
        assert_eq!(result.failure_reason.as_deref(), Some("timeout"));
        assert_eq!(result.payload["completion"], Value::String("frontier".into()));
    }
}
