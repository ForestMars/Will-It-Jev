// src/router/tier6_runner.rs

use super::tier5_runner::DownstreamError;
use serde_json::Value;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Tier6Runner {
    client: reqwest::Client,
    endpoint: String,
    model: String,
}

impl Tier6Runner {
    pub fn new(endpoint: String, model: String, timeout: Duration) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder().timeout(timeout).build()?;
        Ok(Self {
            client,
            endpoint,
            model,
        })
    }

    pub async fn invoke(&self, prompt: &str) -> Result<(Value, Duration), DownstreamError> {
        let start = Instant::now();
        let payload = serde_json::json!({
            "model": &self.model,
            "prompt": prompt,
            "stream": false
        });

        let response = self
            .client
            .post(&self.endpoint)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    DownstreamError::Timeout
                } else {
                    DownstreamError::Transport(e.to_string())
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(DownstreamError::HttpStatus(status.as_u16(), text));
        }

        let body = response
            .json::<Value>()
            .await
            .map_err(|e| DownstreamError::Transport(e.to_string()))?;

        Ok((body, start.elapsed()))
    }
}
