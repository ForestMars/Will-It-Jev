// src/router/tier4_runner.rs
//
// Tier 4: Decoder State Evaluation Runner
// Applies a task (passed in as context) to an encoded input via the remote
// sidecar endpoint, returning a confidence score and evaluation reason.
// Tier 4 is entirely agnostic of downstream routing backends (like Jev or LLM).

use std::time::{Duration, Instant};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier4Outcome {
    pub score: f32,
    pub reason: String,
}

#[derive(Clone)]
pub struct Tier4Runner {
    client: Client,
    endpoint: String,
    timeout: Duration,
}

impl Tier4Runner {
    /// Creates a new `Tier4Runner` instance targeting the decoder state endpoint.
    pub fn new(endpoint: String, timeout: Duration) -> Result<Self, reqwest::Error> {
        let client = Client::builder()
            .timeout(timeout)
            .build()?;
        Ok(Self {
            client,
            endpoint,
            timeout,
        })
    }

    /// Evaluates the prompt/context against the decoder state endpoint within the allowed budget.
    pub async fn evaluate(
        &self,
        prompt: &str,
        budget: Duration,
    ) -> Result<Tier4Outcome, Box<dyn std::error::Error + Send + Sync>> {
        let _start = Instant::now();
        
        // Use the remaining cascade budget or the configured runner timeout, whichever is smaller
        let effective_timeout = std::cmp::min(self.timeout, budget);

        let payload = serde_json::json!({
            "prompt": prompt,
        });

        let response = self.client
            .post(&self.endpoint)
            .timeout(effective_timeout)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("decoder state endpoint returned status {status}: {body}").into());
        }

        // Parse response expected from the decoder state service
        #[derive(Deserialize)]
        struct RawResponse {
            score: f32,
            reason: Option<String>,
        }

        let parsed = response.json::<RawResponse>().await?;

        Ok(Tier4Outcome {
            score: parsed.score,
            reason: parsed.reason.unwrap_or_else(|| "decoder_state_evaluated".into()),
        })
    }
}