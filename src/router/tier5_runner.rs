use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{json, Value};
use tokio::time::timeout;

use super::tier4_runner::DownstreamError;

pub struct Tier5Runner {
    client: Client,
    endpoint: String,
    model: String,
    timeout: Duration,
}

impl Tier5Runner {
    pub fn new(
        endpoint: String,
        model: String,
        timeout: Duration,
    ) -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: Client::builder().build()?,
            endpoint,
            model,
            timeout: timeout.min(Duration::from_secs(25)),
        })
    }

    pub async fn invoke(&self, prompt: &str) -> Result<(Value, Duration), DownstreamError> {
        let started = Instant::now();
        let result = timeout(self.timeout, async {
            let response = self
                .client
                .post(&self.endpoint)
                .json(&json!({
                    "model": self.model,
                    "prompt": prompt,
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
            response
                .json::<Value>()
                .await
                .map_err(|error| DownstreamError::InvalidResponse(error.to_string()))
        })
        .await;
        let elapsed = started.elapsed();
        match result {
            Err(_) => Err(DownstreamError::Timeout),
            Ok(Err(error)) => Err(error),
            Ok(Ok(value)) => Ok((value, elapsed)),
        }
    }
}
