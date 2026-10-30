// src/pipeline/task.rs
//
// Use-case side of the split: task identity, raw readings, and the per-task profile
// that turns a tier's raw reading into "resolve here" or "pass through".

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::Deserialize;

/// Labels the existing binary route task uses on the wire (see tier3_runner::decide).
pub const LABEL_JEV: &str = "jev_capable";
pub const LABEL_LLM: &str = "needs_llm";

/// Identifies a use case. The string is also what travels in `ScoreTasksRequest.tasks`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
#[serde(transparent)]
pub struct TaskId(String);

impl TaskId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    /// The binary decision: jev_capable vs needs_llm.
    pub fn route() -> Self {
        Self::new("route")
    }
    /// Domain / category tagging.
    pub fn tag() -> Self {
        Self::new("tag")
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A tier's raw answer for one task: top label, its confidence, and optionally the
/// whole distribution (used to fill `alternatives` for tagging).
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub label: String,
    pub confidence: f32,
    pub scores: Vec<(String, f32)>,
}

impl Reading {
    pub fn new(label: impl Into<String>, confidence: f32) -> Self {
        let label = label.into();
        Self {
            scores: vec![(label.clone(), confidence)],
            label,
            confidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TierReading {
    Scored(Reading),
    /// The tier has nothing to say for this task. Reasons reuse the cascade's
    /// pass-through vocabulary (timeout, rpc_error, deadline_exhausted, ...) plus
    /// `unsupported_task` for a tier with no head for the task.
    Abstain(&'static str),
}

/// Confidence bar for one task at one tier.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierPolicy {
    /// Bar for any label without its own entry in `label_min_confidence`.
    pub min_confidence: f32,
    /// Per-label bars, e.g. route's asymmetric jev 0.95 / llm 0.70.
    #[serde(default)]
    pub label_min_confidence: BTreeMap<String, f32>,
    /// Labels this task may never exit on at this tier (tier 2 only resolves to jev).
    #[serde(default)]
    pub never_resolve: BTreeSet<String>,
    /// Require confidence strictly above the bar (tier 1's `> AMBIGUITY_UPPER_BOUND`).
    #[serde(default)]
    pub strict: bool,
}

impl TierPolicy {
    pub fn new(min_confidence: f32) -> Self {
        Self {
            min_confidence,
            label_min_confidence: BTreeMap::new(),
            never_resolve: BTreeSet::new(),
            strict: false,
        }
    }
    pub fn with_label(mut self, label: &str, min: f32) -> Self {
        self.label_min_confidence.insert(label.to_string(), min);
        self
    }
    pub fn never(mut self, label: &str) -> Self {
        self.never_resolve.insert(label.to_string());
        self
    }
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }
}

/// What a task does when every tier it may use has passed.
#[derive(Debug, Clone, PartialEq)]
pub enum FallbackPolicy {
    /// Resolve to this label (route: needs_llm, i.e. the generative backend).
    Label(String),
    /// Leave the task unresolved for the caller to hand to something else (tagging:
    /// the existing LLM tagger).
    Defer,
}

#[derive(Debug, Clone)]
pub struct TaskProfile {
    pub task: TaskId,
    /// Allowed label set. A reading outside it never resolves. `None` accepts any.
    pub labels: Option<Vec<String>>,
    /// Tiers this task may use, by tier number. A tier with no entry is skipped.
    pub tiers: BTreeMap<u8, TierPolicy>,
    pub fallback: FallbackPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Resolve(Reading),
    Pass {
        reason: &'static str,
        top_guess: Option<String>,
        score: f32,
    },
}

impl TaskProfile {
    pub fn policy(&self, tier: u8) -> Option<&TierPolicy> {
        self.tiers.get(&tier)
    }

    /// Pure decision logic, the profile-side counterpart of `tier3_runner::decide`.
    pub fn decide(&self, tier: u8, reading: &TierReading) -> Decision {
        let Some(policy) = self.policy(tier) else {
            return pass("tier_not_configured", None, 0.0);
        };
        let reading = match reading {
            TierReading::Abstain(reason) => return pass(reason, None, 0.0),
            TierReading::Scored(reading) => reading,
        };

        let confidence = if reading.confidence.is_nan() {
            0.0
        } else {
            reading.confidence.clamp(0.0, 1.0)
        };
        let guess = Some(reading.label.clone());

        if let Some(allowed) = &self.labels {
            if !allowed.iter().any(|label| *label == reading.label) {
                return pass("unknown_label", None, confidence);
            }
        }
        if policy.never_resolve.contains(&reading.label) {
            return pass("below_threshold", guess, confidence);
        }

        let bar = policy
            .label_min_confidence
            .get(&reading.label)
            .copied()
            .unwrap_or(policy.min_confidence);
        let clears = if policy.strict { confidence > bar } else { confidence >= bar };
        if clears {
            Decision::Resolve(Reading {
                label: reading.label.clone(),
                confidence,
                scores: reading.scores.clone(),
            })
        } else {
            pass("below_threshold", guess, confidence)
        }
    }

    /// The route task exactly as the hard-coded cascade runs it today: tier 1 jev
    /// above 0.78 (strict), tier 2 jev at 0.95 and never llm, tiers 3 and 4 jev 0.95 /
    /// llm 0.70 (the values main.rs passes to Tier3Runner), generative fallback.
    pub fn route_default() -> Self {
        use crate::router::tier1_runner::AMBIGUITY_UPPER_BOUND;
        use crate::router::tier2_runner::DEFAULT_CONFIDENCE_THRESHOLD;
        let sidecar = || TierPolicy::new(0.95).with_label(LABEL_JEV, 0.95).with_label(LABEL_LLM, 0.70);
        let mut tiers = BTreeMap::new();
        tiers.insert(1, TierPolicy::new(AMBIGUITY_UPPER_BOUND).strict().never(LABEL_LLM));
        tiers.insert(2, TierPolicy::new(DEFAULT_CONFIDENCE_THRESHOLD).never(LABEL_LLM));
        tiers.insert(3, sidecar());
        tiers.insert(4, sidecar());
        Self {
            task: TaskId::route(),
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            tiers,
            fallback: FallbackPolicy::Label(LABEL_LLM.into()),
        }
    }
}

fn pass(reason: &'static str, top_guess: Option<String>, score: f32) -> Decision {
    Decision::Pass { reason, top_guess, score }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::tier2_runner::{calibrate, Tier2Route, DEFAULT_CONFIDENCE_THRESHOLD};
    use crate::router::tier3_runner::{decide, Tier3Outcome, Tier3Route};

    fn scored(label: &str, confidence: f32) -> TierReading {
        TierReading::Scored(Reading::new(label, confidence))
    }

    fn resolved_label(d: &Decision) -> Option<&str> {
        match d {
            Decision::Resolve(r) => Some(r.label.as_str()),
            Decision::Pass { .. } => None,
        }
    }

    /// The route profile must decide exactly as tier3_runner::decide does with the
    /// thresholds main.rs passes (0.95 / 0.70), for tiers 3 and 4.
    #[test]
    fn route_profile_matches_legacy_sidecar_decision() {
        let profile = TaskProfile::route_default();
        let confidences = [f32::NAN, -1.0, 0.0, 0.5, 0.69, 0.70, 0.71, 0.94, 0.95, 0.96, 1.0, 2.0];
        for tier in [3u8, 4] {
            for label in [LABEL_JEV, LABEL_LLM, "???"] {
                for conf in confidences {
                    let legacy = decide(label, conf, "m", 0.95, 0.70);
                    let new = profile.decide(tier, &scored(label, conf));
                    match legacy {
                        Tier3Outcome::Resolved { route, .. } => {
                            let want = match route {
                                Tier3Route::Jev => LABEL_JEV,
                                Tier3Route::Llm => LABEL_LLM,
                            };
                            assert_eq!(resolved_label(&new), Some(want), "tier {tier} {label} {conf}");
                        }
                        Tier3Outcome::PassThrough { reason, .. } => match new {
                            Decision::Pass { reason: r, .. } => assert_eq!(r, reason, "tier {tier} {label} {conf}"),
                            other => panic!("tier {tier} {label} {conf}: legacy passed, profile gave {other:?}"),
                        },
                    }
                }
            }
        }
    }

    /// Tier 2 resolves only to jev, at calibrated margin >= the threshold.
    #[test]
    fn route_profile_matches_legacy_tier2_decision() {
        let profile = TaskProfile::route_default();
        let probs = [0.0f32, 0.02, 0.5, 0.97, 0.975, 0.99, 1.0];
        for p_jev in probs {
            for p_llm in probs {
                let (route, _, calibrated) = calibrate(p_jev, p_llm);
                let legacy_resolves =
                    route == Tier2Route::Jev && calibrated >= DEFAULT_CONFIDENCE_THRESHOLD;
                let label = if route == Tier2Route::Jev { LABEL_JEV } else { LABEL_LLM };
                let new = profile.decide(2, &scored(label, calibrated));
                assert_eq!(resolved_label(&new).is_some(), legacy_resolves, "p_jev={p_jev} p_llm={p_llm}");
            }
        }
    }

    #[test]
    fn route_profile_tier2_bar_is_inclusive() {
        let profile = TaskProfile::route_default();
        assert!(resolved_label(&profile.decide(2, &scored(LABEL_JEV, DEFAULT_CONFIDENCE_THRESHOLD))).is_some());
        assert!(resolved_label(&profile.decide(2, &scored(LABEL_LLM, 1.0))).is_none());
    }

    /// Tier 1 resolves when conf > AMBIGUITY_UPPER_BOUND: strictly, so exactly 0.78 passes.
    #[test]
    fn route_profile_tier1_bar_is_strict() {
        use crate::router::tier1_runner::AMBIGUITY_UPPER_BOUND as BAR;
        let profile = TaskProfile::route_default();
        assert!(resolved_label(&profile.decide(1, &scored(LABEL_JEV, BAR))).is_none());
        assert!(resolved_label(&profile.decide(1, &scored(LABEL_JEV, BAR + 0.01))).is_some());
    }

    #[test]
    fn abstain_and_unconfigured_tier_never_resolve() {
        let profile = TaskProfile::route_default();
        assert!(matches!(
            profile.decide(3, &TierReading::Abstain("timeout")),
            Decision::Pass { reason: "timeout", .. }
        ));
        assert!(matches!(
            profile.decide(5, &scored(LABEL_JEV, 1.0)),
            Decision::Pass { reason: "tier_not_configured", .. }
        ));
    }

    #[test]
    fn labels_outside_the_allowed_set_never_resolve() {
        let mut profile = TaskProfile::route_default();
        profile.task = TaskId::tag();
        profile.labels = Some(vec!["engineering".into()]);
        assert!(matches!(
            profile.decide(3, &scored("help_desk", 1.0)),
            Decision::Pass { reason: "unknown_label", .. }
        ));
    }
}
