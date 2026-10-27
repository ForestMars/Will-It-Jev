use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{json, Value};
use tokio::time::timeout;

#[derive(Debug)]
pub enum DownstreamError {
    Disabled,
    CircuitOpen,
    Timeout,
    NotEligible,
    Transport(String),
    HttpStatus(u16, String),
    InvalidResponse(String),
}

impl std::fmt::Display for DownstreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disabled => write!(f, "downstream tier is disabled by configuration"),
            Self::CircuitOpen => write!(f, "circuit breaker is open"),
            Self::Timeout => write!(f, "downstream request timed out"),
            Self::NotEligible => write!(f, "Laya determined Jev is not eligible"),
            Self::Transport(error) => write!(f, "downstream transport error: {error}"),
            Self::HttpStatus(status, body) => {
                write!(f, "downstream returned HTTP {status}: {body}")
            }
            Self::InvalidResponse(error) => write!(f, "invalid downstream JSON response: {error}"),
        }
    }
}

pub struct CircuitBreaker {
    failure_threshold: usize,
    failure_window: Duration,
    open_duration: Duration,
    state: std::sync::Mutex<CircuitState>,
}

struct CircuitState {
    failures: std::collections::VecDeque<Instant>,
    open_until: Option<Instant>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, failure_window: Duration, open_duration: Duration) -> Self {
        Self {
            failure_threshold: failure_threshold.max(1),
            failure_window,
            open_duration,
            state: std::sync::Mutex::new(CircuitState {
                failures: std::collections::VecDeque::new(),
                open_until: None,
            }),
        }
    }

    pub fn is_open(&self) -> bool {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.open_until.is_some_and(|until| until > now) {
            return true;
        }
        state.open_until = None;
        false
    }

    pub fn record_failure(&self) {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while state
            .failures
            .front()
            .is_some_and(|at| now.duration_since(*at) > self.failure_window)
        {
            state.failures.pop_front();
        }
        state.failures.push_back(now);
        if state.failures.len() >= self.failure_threshold {
            state.open_until = Some(now + self.open_duration);
            state.failures.clear();
        }
    }

    pub fn record_success(&self) {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        state.failures.clear();
        state.open_until = None;
    }
}

pub struct Tier4Runner {
    client: Client,
    endpoint: String,
    timeout: Duration,
    breaker: CircuitBreaker,
}

impl Tier4Runner {
    pub fn new(
        endpoint: String,
        timeout: Duration,
        failure_threshold: usize,
        failure_window: Duration,
        breaker_open_duration: Duration,
    ) -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: Client::builder().build()?,
            endpoint,
            timeout,
            breaker: CircuitBreaker::new(
                failure_threshold,
                failure_window,
                breaker_open_duration,
            ),
        })
    }

    pub async fn invoke(
        &self,
        state: &Value,
        questions: &serde_json::Map<String, Value>,
    ) -> Result<(Value, Duration), DownstreamError> {
        if self.breaker.is_open() {
            return Err(DownstreamError::CircuitOpen);
        }

        let started = Instant::now();
        let result = timeout(self.timeout, async {
            let mut eligibility_id = "__router_jev_eligible".to_string();
            while questions.contains_key(&eligibility_id) {
                eligibility_id.push('_');
            }
            let mut questions = questions.clone();
            questions.insert(
                eligibility_id.clone(),
                json!({
                    "type": "choice",
                    "instructions": "Can Jev correctly answer the user's request using the requested typed output?",
                    "criteria": {
                        "yes": "Jev can answer the request.",
                        "no": "Jev cannot answer the request."
                    }
                }),
            );
            let response = self
                .client
                .post(&self.endpoint)
                .json(&json!({
                    "model": "laya",
                    "state": state,
                    "questions": questions,
                    "stream": false,
                }))
                .send()
                .await
                .map_err(|error| DownstreamError::Transport(error.to_string()))?;
            let status = response.status();
            if !status.is_success() {
                let body = response
                    .text()
                    .await
                    .map_err(|error| DownstreamError::Transport(error.to_string()))?;
                return Err(DownstreamError::HttpStatus(status.as_u16(), body));
            }
            let mut payload = response
                .json::<Value>()
                .await
                .map_err(|error| DownstreamError::InvalidResponse(error.to_string()))?;
            let eligibility = payload
                .get("answers")
                .and_then(|answers| answers.get(&eligibility_id))
                .ok_or_else(|| {
                    DownstreamError::InvalidResponse(
                        format!("Ollaya response is missing answers.{eligibility_id}"),
                    )
                })?;
            let eligible = eligibility
                .get("choice")
                .and_then(Value::as_str)
                .is_some_and(|choice| choice == "yes" || choice == "jev_eligible");
            let confidence = eligibility
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if !eligible || confidence < 0.7 {
                return Err(DownstreamError::NotEligible);
            }
            if let Some(answers) = payload.get_mut("answers").and_then(Value::as_object_mut) {
                answers.remove(&eligibility_id);
            }
            Ok(payload)
        })
        .await
        .map_err(|_| DownstreamError::Timeout)
        .and_then(|result| result);
        let elapsed = started.elapsed();
        match result {
            Ok(value) => {
                self.breaker.record_success();
                Ok((value, elapsed))
            }
            Err(DownstreamError::NotEligible) => {
                self.breaker.record_success();
                Err(DownstreamError::NotEligible)
            }
            Err(error) => {
                self.breaker.record_failure();
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CircuitBreaker;
    use std::time::Duration;

    #[test]
    fn breaker_opens_after_consecutive_failures_and_resets_on_success() {
        let breaker = CircuitBreaker::new(2, Duration::from_secs(2), Duration::from_secs(1));
        breaker.record_failure();
        assert!(!breaker.is_open());
        breaker.record_failure();
        assert!(breaker.is_open());
        std::thread::sleep(Duration::from_millis(1010));
        assert!(!breaker.is_open());
        breaker.record_failure();
        breaker.record_success();
        breaker.record_failure();
        assert!(!breaker.is_open());
    }
}
