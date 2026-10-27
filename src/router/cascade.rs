// src/router/cascade.rs
//
// Every tier boundary logs exactly one of: TRY, RESOLVED, ESCALATE.
// Every request ends with exactly one DONE line.
// Faults (timeout, rpc_error, deadline_exhausted) escalate at WARN,
// ordinary low-confidence escalations at INFO.

use std::fmt;
use std::time::{Duration, Instant};

use chrono::Utc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::telemetry::{
    FinalRouting, Hashes, RouteRecord, Sampling, Tier1Outcome as Tier1Record,
    Tier2Outcome as Tier2Record, Tier3Outcome as Tier3Record,
};

use super::tier1_runner::{Tier1Engine, AMBIGUITY_UPPER_BOUND};
use super::tier2_runner::{Tier2Outcome, Tier2Route, Tier2Runner};
use super::tier3_runner::{Tier3Outcome, Tier3Route, Tier3Runner};
use super::modernbert_runner::ModernBertRunner;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Jev,
    Llm,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Jev => write!(f, "jev"),
            Backend::Llm => write!(f, "llm"),
        }
    }
}

#[allow(dead_code)] // read by the axum handler once it dispatches on the decision
#[derive(Debug, Clone, Copy)]
pub struct Decision {
    pub backend: Backend,
    pub decided_by_tier: u8,
}

const FAULT_REASONS: [&str; 3] = ["timeout", "rpc_error", "deadline_exhausted"];

fn escalate(request_id: &str, from: u8, to: u8, reason: &str, detail: String, took: Duration) {
    let line = format!(
        "[request_id={request_id}] ESCALATE  tier={from} -> tier={to}  reason={reason}  {detail}  ({took:?})"
    );
    if FAULT_REASONS.contains(&reason) {
        warn!("{line}");
    } else {
        info!("{line}");
    }
}

fn resolved(
    request_id: &str,
    tier: u8,
    model: &str,
    backend: Backend,
    conf: Option<f32>,
    tier_took: Duration,
    total: Duration,
) -> Decision {
    let conf = conf.map_or_else(|| "n/a".to_string(), |c| format!("{c:.3}"));
    info!("[request_id={request_id}] RESOLVED  tier={tier} ({model}) -> {backend}  conf={conf}  ({tier_took:?})");
    info!("[request_id={request_id}] DONE      decided_by=tier{tier}  backend={backend}  total={total:?}");
    Decision {
        backend,
        decided_by_tier: tier,
    }
}

fn finish_route(
    request_id: &str,
    decision: Decision,
    prompt: &str,
    start: Instant,
    timestamp: &chrono::DateTime<Utc>,
    hashes: &Hashes,
    tier1_outcome: Tier1Record,
    tier2_outcome: Option<Tier2Record>,
    tier3_outcome: Option<Tier3Record>,
    tier4_outcome: Option<Tier3Record>,
) -> (Decision, RouteRecord) {
    let latency_us = start.elapsed().as_micros();
    let record = RouteRecord {
        request_id: request_id.to_string(),
        timestamp: timestamp.clone(),
        prompt_raw: prompt.to_string(),
        hashes: hashes.clone(),
        tier1_outcome,
        tier2_outcome,
        tier3_outcome,
        tier4_outcome,
        final_routing: FinalRouting {
            backend: decision.backend.to_string(),
            tier: decision.decided_by_tier,
            latency_us,
        },
        router_classification_latency_us: start.elapsed().as_micros(),
        downstream_invocation_latency_ms: 0.0,
        request_latency_primary_ms: None,
        request_latency_failover_ms: None,
        downstream_outcome: None,
        downstream_tier: None,
        is_failover: false,
        failure_reason: None,
        sampling: Sampling::default(),
    };
    (decision, record)
}

pub async fn route(
    _req: u64,
    prompt: &str,
    t1: Option<&Tier1Engine>,
    t2: Option<&Tier2Runner>,
    t3: Option<&ModernBertRunner>,
    t4: Option<&Tier3Runner>,
    budget: Duration,
    hashes: &Hashes,
) -> (Decision, RouteRecord) {
    let request_id = Uuid::new_v4().to_string();
    let start = Instant::now();
    let timestamp = Utc::now();

    // ---- Tier 1: heuristic pattern engine, in-process ----
    let tier1_record = if let Some(t1) = t1 {
        info!("[request_id={request_id}] TRY tier=1 (heuristic) prompt={prompt:?}");
        let t = Instant::now();
        let r1 = t1.classify(prompt);
        let record = Tier1Record {
            confidence: r1.confidence,
            decided: r1.decided,
            reason: r1.reason.clone(),
        };
        info!(
            "[request_id={request_id}] SCORE  tier=1 conf={:.3} decided={} reason={}",
            r1.confidence, r1.decided, r1.reason
        );

        if r1.decided && r1.confidence > AMBIGUITY_UPPER_BOUND {
            let decision = resolved(
                &request_id,
                1,
                "heuristic",
                Backend::Jev,
                Some(r1.confidence),
                t.elapsed(),
                start.elapsed(),
            );
            return finish_route(
                &request_id,
                decision,
                prompt,
                start,
                &timestamp,
                hashes,
                record,
                None,
                None,
                None,
            );
        }

        escalate(
            &request_id,
            1,
            2,
            if r1.decided { "not_jev" } else { "ambiguous" },
            format!("conf={:.3} reason={}", r1.confidence, r1.reason),
            t.elapsed(),
        );
        record
    } else {
        info!("[request_id={request_id}] SKIP tier=1 (disabled)");
        Tier1Record {
            confidence: 0.0,
            decided: false,
            reason: "disabled".into(),
        }
    };

    // ---- Tier 2: FastText, in-process ----
    let tier2_record = if let Some(t2) = t2 {
        info!("[request_id={request_id}] TRY tier=2 (fasttext) prompt={prompt:?}");
        let t = Instant::now();
        match t2.evaluate(prompt) {
            Tier2Outcome::Resolved {
                route,
                confidence,
                calibrated_score,
                p_jev,
                p_llm,
                raw_probabilities,
            } => {
                info!("[request_id={request_id}] SCORE  tier=2 raw={confidence:.3} calibrated={calibrated_score:.3}");
                let backend = match route {
                    Tier2Route::Jev => Backend::Jev,
                    Tier2Route::LLM => Backend::Llm,
                };
                let decision = resolved(
                    &request_id,
                    2,
                    "fasttext",
                    backend,
                    Some(calibrated_score),
                    t.elapsed(),
                    start.elapsed(),
                );
                return finish_route(
                    &request_id,
                    decision,
                    prompt,
                    start,
                    &timestamp,
                    hashes,
                    tier1_record,
                    Some(Tier2Record {
                        p_jev,
                        p_llm,
                        raw_probabilities,
                        reason: "resolved".to_string(),
                    }),
                    None,
                    None,
                );
            }

            Tier2Outcome::PassThrough {
                top_guess,
                score,
                reason,
                p_jev,
                p_llm,
                raw_probabilities,
            } => {
                escalate(
                    &request_id,
                    2,
                    3,
                    reason,
                    format!("top_guess={top_guess:?} score={score:.3}"),
                    t.elapsed(),
                );
                Some(Tier2Record {
                    p_jev,
                    p_llm,
                    raw_probabilities,
                    reason: reason.to_string(),
                })
            }
        }
    } else {
        info!("[request_id={request_id}] SKIP tier=2 (disabled)");
        None
    };

    // ---- Tier 3: ModernBERT ONNX Runtime ----
    let tier3_record = if let Some(t3) = t3 {
        info!("[request_id={request_id}] TRY tier=3 (modernbert/onnx) prompt={prompt:?}");
        let t = Instant::now();
        let tier3_result = t3.evaluate(prompt);
        let tier3_record = match &tier3_result {
            Tier3Outcome::Resolved { confidence, .. } => Some(Tier3Record {
                score: *confidence,
                reason: "resolved".to_string(),
            }),
            Tier3Outcome::PassThrough { score, reason, .. } => Some(Tier3Record {
                score: *score,
                reason: reason.to_string(),
            }),
        };
        match tier3_result {
            Tier3Outcome::Resolved {
                route,
                confidence,
                model_id,
            } => {
                let backend = match route {
                    Tier3Route::Jev => Backend::Jev,
                    Tier3Route::Llm => Backend::Llm,
                };
                let decision = resolved(
                    &request_id,
                    3,
                    &model_id,
                    backend,
                    Some(confidence),
                    t.elapsed(),
                    start.elapsed(),
                );
                return finish_route(
                    &request_id,
                    decision,
                    prompt,
                    start,
                    &timestamp,
                    hashes,
                    tier1_record,
                    tier2_record,
                    tier3_record,
                    None,
                );
            }
            Tier3Outcome::PassThrough {
                top_guess,
                score,
                reason,
            } => {
                escalate(
                    &request_id,
                    3,
                    4,
                    reason,
                    format!("top_guess={top_guess:?} score={score:.3}"),
                    t.elapsed(),
                );
                tier3_record
            }
        }
    } else {
        info!("[request_id={request_id}] SKIP tier=3 (disabled)");
        None
    };

    // ---- Tier 4: Harrier sidecar over gRPC ----
    let tier4_record = if let Some(t4) = t4 {
        let left = budget.saturating_sub(start.elapsed());
        info!("[request_id={request_id}] TRY budget_left={left:?} tier=4 (harrier)");
        let t = Instant::now();
        let tier4_result = t4.evaluate(prompt, left).await;
        let tier4_record = match &tier4_result {
            Tier3Outcome::Resolved { confidence, .. } => Some(Tier3Record {
                score: *confidence,
                reason: "resolved".to_string(),
            }),
            Tier3Outcome::PassThrough { score, reason, .. } => Some(Tier3Record {
                score: *score,
                reason: reason.to_string(),
            }),
        };
        match tier4_result {
            Tier3Outcome::Resolved {
                route,
                confidence,
                model_id,
            } => {
                let backend = match route {
                    Tier3Route::Jev => Backend::Jev,
                    Tier3Route::Llm => Backend::Llm,
                };
                let decision = resolved(
                    &request_id,
                    4,
                    &model_id,
                    backend,
                    Some(confidence),
                    t.elapsed(),
                    start.elapsed(),
                );
                return finish_route(
                    &request_id,
                    decision,
                    prompt,
                    start,
                    &timestamp,
                    hashes,
                    tier1_record,
                    tier2_record,
                    tier3_record,
                    tier4_record,
                );
            }
            Tier3Outcome::PassThrough {
                top_guess,
                score,
                reason,
            } => {
                escalate(
                    &request_id,
                    4,
                    5,
                    reason,
                    format!("top_guess={top_guess:?} score={score:.3}"),
                    t.elapsed(),
                );
                tier4_record
            }
        }
    } else {
        info!("[request_id={request_id}] SKIP tier=4 (disabled)");
        None
    };

    // The local cascade falls through to the configured generative backend.
    info!("[request_id={request_id}] ESCALATE to downstream backend");
    let decision = resolved(
        &request_id,
        6,
        "llm-fallback",
        Backend::Llm,
        None,
        Duration::ZERO,
        start.elapsed(),
    );
    finish_route(
        &request_id,
        decision,
        prompt,
        start,
        &timestamp,
        hashes,
        tier1_record,
        tier2_record,
        tier3_record,
        tier4_record,
    )
}
