// src/pipeline/task.rs
//
// Task-side definitions. A TaskProfile describes a use case only: its identity,
// accepted labels, and fallback. Tier-specific calibration is deliberately kept
// in TierPolicyMap, outside this module's TaskProfile.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::Deserialize;

pub const LABEL_JEV: &str = "jev_capable";
pub const LABEL_LLM: &str = "needs_llm";

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
#[serde(transparent)]
pub struct TaskId(String);

impl TaskId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn route() -> Self {
        Self::new("route")
    }

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
    Abstain(&'static str),
}

/// Resolution calibration for one task when it is evaluated by one tier.
/// This is stored in the pipeline policy registry, never inside TaskProfile.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
pub struct TierPolicy {
    pub min_confidence: f32,
    #[serde(default)]
    pub label_min_confidence: BTreeMap<String, f32>,
    #[serde(default)]
    pub never_resolve: BTreeSet<String>,
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

/// Execution calibration indexed independently from task definitions:
/// tier number -> task id -> resolution policy.
pub type TierPolicyMap = BTreeMap<u8, BTreeMap<TaskId, TierPolicy>>;

#[derive(Debug, Clone, PartialEq)]
pub enum FallbackPolicy {
    Label(String),
    Defer,
}

/// A task's semantic contract. There is intentionally no tier number or
/// per-tier policy in this structure.
#[derive(Debug, Clone)]
pub struct TaskProfile {
    pub task: TaskId,
    pub labels: Option<Vec<String>>,
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

    pub fn decide(&self, policy: &TierPolicy, reading: &TierReading) -> Decision {
        let reading = match reading {
            TierReading::Abstain(reason) => {
                return pass(reason, None, 0.0);
            }
            TierReading::Scored(reading) => reading,
        };

        let confidence = if !reading.confidence.is_finite() {
            0.0
        } else {
            reading.confidence.clamp(0.0, 1.0)
        };
        let guess = Some(reading.label.clone());

        if let Some(allowed) = &self.labels {
            if !allowed.iter().any(|label| label == &reading.label) {
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
        let clears = if policy.strict {
            confidence > bar
        } else {
            confidence >= bar
        };

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
}

fn pass(reason: &'static str, top_guess: Option<String>, score: f32) -> Decision {
    Decision::Pass {
        reason,
        top_guess,
        score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(label: &str, confidence: f32) -> TierReading {
        TierReading::Scored(Reading::new(label, confidence))
    }

    #[test]
    fn task_profile_contains_no_tier_calibration() {
        let profile = TaskProfile {
            task: TaskId::route(),
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            fallback: FallbackPolicy::Label(LABEL_LLM.into()),
        };
        assert_eq!(profile.task, TaskId::route());
        assert_eq!(profile.labels.as_ref().unwrap().len(), 2);
        assert_eq!(profile.fallback, FallbackPolicy::Label(LABEL_LLM.into()));
    }

    #[test]
    fn policy_is_supplied_independently_of_task_definition() {
        let profile = TaskProfile {
            task: TaskId::route(),
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            fallback: FallbackPolicy::Label(LABEL_LLM.into()),
        };
        let tier_one_policy = TierPolicy::new(0.78).strict().never(LABEL_LLM);
        let tier_two_policy = TierPolicy::new(0.95).never(LABEL_LLM);

        assert!(matches!(
            profile.decide(&tier_one_policy, &scored(LABEL_JEV, 0.79)),
            Decision::Resolve(_)
        ));
        assert!(matches!(
            profile.decide(&tier_two_policy, &scored(LABEL_JEV, 0.94)),
            Decision::Pass { reason: "below_threshold", .. }
        ));
    }

    #[test]
    fn rejects_unknown_labels_and_non_finite_confidence() {
        let profile = TaskProfile {
            task: TaskId::route(),
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            fallback: FallbackPolicy::Label(LABEL_LLM.into()),
        };
        let policy = TierPolicy::new(0.5);
        assert!(matches!(
            profile.decide(&policy, &scored("other", 0.99)),
            Decision::Pass { reason: "unknown_label", .. }
        ));
        assert!(matches!(
            profile.decide(&policy, &scored(LABEL_JEV, f32::NAN)),
            Decision::Pass { reason: "below_threshold", .. }
        ));
        assert!(matches!(
            profile.decide(&policy, &scored(LABEL_JEV, f32::INFINITY)),
            Decision::Pass { reason: "below_threshold", .. }
        ));
    }
}
