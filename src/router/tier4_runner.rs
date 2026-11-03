use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Represents the routing decision or classification returned by Tier 4.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Tier4Route {
    Primary,
    Secondary,
    Fallback,
    Custom(String),
}

/// Represents the full evaluation outcome from Tier 4, including confidence metrics and telemetry details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier4Outcome {
    pub route: Tier4Route,
    pub confidence: f32,
    pub raw_score: Option<f32>,
    pub latency_ms: u64,
}

/// Runner responsible for communicating with the Tier 4 backend endpoint (e.g., Harrier / secondary transformer model).
#[derive(Clone)]
pub struct Tier4Runner {
    client: reqwest::Client,
    endpoint: String,
    high_threshold: f32,
    low_threshold: f32,
    timeout: Duration,
}

impl Tier4Runner {
    /// Creates a new `Tier4Runner` instance with custom endpoint configuration and thresholds.
    pub fn new(
        endpoint: impl Into<String>,
        high_threshold: f32,
        low_threshold: f32,
        timeout: Duration,
    ) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()?;
        Ok(Self {
            client,
            endpoint: endpoint.into(),
            high_threshold,
            low_threshold,
            timeout,
        })
    }

    /// Evaluates a prompt and context against the Tier 4 model endpoint.
    pub async fn evaluate(
        &self,
        prompt: &str,
        context: &str,
    ) -> Result<Tier4Outcome, Box<dyn std::error::Error + Send + Sync>> {
        let start = std::time::Instant::now();
        
        let payload = serde_json::json!({
            "prompt": prompt,
            "context": context,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
        });

        let response = self.client
            .post(&self.endpoint)
            .json(&payload)
            .send()
            .await?;

        let mut outcome = response.json::<Tier4Outcome>().await?;
        outcome.latency_ms = start.elapsed().as_millis() as u64;

        Ok(outcome)
    }
}