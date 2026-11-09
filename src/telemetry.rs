use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

#[allow(dead_code)]
pub fn init() {
    // The runtime configures and installs the global tracing subscriber in main().
    // Keeping this as a no-op avoids duplicate global subscriber initialization.
}

#[allow(dead_code)]
pub fn next_req_id() -> u64 {
    REQ_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRecord {
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub prompt_raw: String,
    pub hashes: Hashes,
    /// Per-task decisions observed in this run; may include deferred tasks when
    /// the route handler returns before best-effort work has completed.
    #[serde(default)]
    pub tasks: Vec<TaskDecisionRecord>,
    /// What every configured instance said about the route task, in execution
    /// order. Each record preserves instance, definition and model identity.
    #[serde(default)]
    pub tier_outcomes: Vec<TierOutcomeRecord>,
    pub tier1_outcome: Tier1Outcome,
    pub tier2_outcome: Option<Tier2Outcome>,
    pub tier3_outcome: Option<Tier3Outcome>,
    pub tier4_outcome: Option<Tier4Outcome>,
    /// One-based position in the configured instance vector, or 6 for fallback.
    /// The configured instance name is the identity; see `decided_by_tier_name`.
    #[serde(default)]
    pub cascade_decision_tier: u8,
    /// Name of the tier that decided the route, or `fallback`.
    #[serde(default)]
    pub decided_by_tier_name: String,
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

/// One tier's reading and decision for the route task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TierOutcomeRecord {
    /// Configured realization that executed.
    #[serde(default)]
    pub tier_instance_id: String,
    /// Reusable execution contract.
    #[serde(default)]
    pub tier_id: String,
    /// Backward-compatible alias for the instance name.
    pub tier: String,
    /// Zero-based execution position in the tier registry.
    pub position: usize,
    pub model_id: String,
    /// `scored`, `abstain` or `unsupported`.
    pub status: String,
    /// True when this tier resolved the task.
    pub decided: bool,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub scores: Vec<TaskLabelScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDecisionRecord {
    pub task_id: String,
    pub output_kind: String,
    pub status: String,
    /// Sum of tier-call wall time for calls that included this task.
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolving_tier: Option<u8>,
    /// Name of the tier that resolved the task; absent for fallback or deferred.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolving_tier_name: Option<String>,
    /// Stable configured instance identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolving_tier_instance_id: Option<String>,
    /// Reusable execution contract identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolving_tier_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub scores: Vec<TaskLabelScore>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskLabelScore {
    pub label: String,
    pub score: f32,
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

pub type Tier4Outcome = Tier3Outcome;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalRouting {
    pub backend: String,
    /// Legacy number. Before dispatch it is the deciding tier's ordinal; after
    /// dispatch it is the downstream tier that served the request (5 jev, 6 llm).
    pub tier: u8,
    /// Name matching `tier`: the deciding tier's name before dispatch, then the
    /// downstream backend that served the request (`jev` or `llm`).
    #[serde(default)]
    pub tier_name: String,
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
