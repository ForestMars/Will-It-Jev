use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

#[allow(dead_code)]
pub fn init() {
    // The runtime configures and installs the global tracing subscriber in main().
    // Keeping this as a no-op avoids duplicate global subscriber initialization.
}

pub fn next_req_id() -> u64 {
    REQ_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRecord {
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub prompt_raw: String,
    pub hashes: Hashes,
    pub tier1_outcome: Tier1Outcome,
    pub tier2_outcome: Option<Tier2Outcome>,
    pub tier3_outcome: Option<Tier3Outcome>,
    #[serde(default)]
    pub tier4_outcome: Option<Tier3Outcome>,
    pub final_routing: FinalRouting,
    #[serde(default)]
    pub router_classification_latency_us: u128,
    #[serde(default)]
    pub downstream_invocation_latency_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_latency_primary_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_latency_failover_ms: Option<f64>,
    #[serde(default)]
    pub downstream_outcome: Option<String>,
    #[serde(default)]
    pub downstream_tier: Option<String>,
    #[serde(default)]
    pub is_failover: bool,
    #[serde(default)]
    pub failure_reason: Option<String>,
    #[serde(default)]
    pub sampling: Sampling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hashes {
    pub tier1_toml: String,
    pub tier2_bin: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Outcome {
    pub confidence: f32,
    pub decided: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier2Outcome {
    pub p_jev: f32,
    pub p_llm: f32,
    pub raw_probabilities: Vec<f32>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier3Outcome {
    pub score: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalRouting {
    pub backend: String,
    pub tier: u8,
    pub latency_us: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sampling {
    #[serde(default = "default_sample_prob")]
    pub sample_prob: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            sample_prob: default_sample_prob(),
        }
    }
}

fn default_sample_prob() -> f32 {
    1.0
}
