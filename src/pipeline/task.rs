// src/pipeline/task.rs
//
// Task-side definitions. A Task describes a use case only: its identity,
// accepted labels, and fallback. Tier-specific calibration is deliberately kept
// in TierPolicyMap, outside this module's Task.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use super::tier::{TierId, TierName};

pub const LABEL_JEV: &str = "jev_capable";
pub const LABEL_LLM: &str = "needs_llm";

/// Shape of a task's model output. The task contract owns this; tiers merely
/// supply a compatible reading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    #[default]
    Binary,
    Multiclass,
    Multilabel,
    Ordinal,
}

impl OutputKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::Multiclass => "multiclass",
            Self::Multilabel => "multilabel",
            Self::Ordinal => "ordinal",
        }
    }
}

/// Taxonomy-derived label metadata. IDs are stable classifier labels; the
/// descriptive fields are available to model-head builders and observability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskLabelMetadata {
    pub id: String,
    pub name: String,
    pub description: String,
    pub classifier_criterion: String,
    pub group_id: Option<String>,
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
    /// The tier has no head for this task. This is distinct from a model
    /// abstention and is retained in the trace for capability diagnostics.
    Unsupported,
}

/// Resolution calibration for one task when it is evaluated by one tier.
/// This is stored in the pipeline policy registry, never inside Task.
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
    /// `observe` records the tier's reading for the task and never resolves it:
    /// for heads whose output is logged as metadata (taxonomy, urgency), not routed.
    #[serde(default)]
    pub authority: Authority,
    /// The instance these bars were fitted on. If the instance now running under
    /// this tier's name has a different id, the bars may no longer be valid.
    #[serde(default)]
    pub calibrated_for: Option<TierId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Authority {
    #[default]
    Resolve,
    Observe,
}

#[cfg(test)]
impl TierPolicy {
    pub fn new(min_confidence: f32) -> Self {
        Self {
            min_confidence,
            label_min_confidence: BTreeMap::new(),
            never_resolve: BTreeSet::new(),
            strict: false,
            authority: Authority::Resolve,
            calibrated_for: None,
        }
    }

    pub fn observe(mut self) -> Self {
        self.authority = Authority::Observe;
        self
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
/// tier name -> task id -> resolution policy. Execution order is not encoded
/// here; it belongs to the `TierRegistry`.
pub type TierPolicyMap = BTreeMap<TierId, BTreeMap<String, TierPolicy>>;

#[derive(Debug, Clone, PartialEq)]
pub enum FallbackPolicy {
    Label(String),
    Defer,
}

/// Authored ordered fallback route. Names resolve to generated TierIds when the runtime graph is compiled.
#[derive(Debug, Clone)]
pub struct Cascade<T = TierId> {
    pub tiers: Vec<T>,
}

impl<T> Default for Cascade<T> {
    fn default() -> Self { Self { tiers: Vec::new() } }
}

/// A task's output-head contract and its own ordered fallback cascade.
#[derive(Debug, Clone)]
pub struct Task<TierRef = TierId> {
    pub name: String,
    pub cascade: Cascade<TierRef>,
    pub output_kind: OutputKind,
    pub labels: Option<Vec<String>>,
    pub label_metadata: Vec<TaskLabelMetadata>,
    pub taxonomy_version: Option<String>,
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

impl<TierRef> Task<TierRef> {

    pub fn decide(&self, policy: &TierPolicy, reading: &TierReading) -> Decision {
        let reading = match reading {
            TierReading::Abstain(reason) => {
                return pass(reason, None, 0.0);
            }
            TierReading::Unsupported => {
                return pass("unsupported_task", None, 0.0);
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
            if !allowed.iter().any(|label| label == &reading.label)
                || reading.scores.iter().any(|(label, _)| !allowed.iter().any(|known| known == label))
            {
                return pass("unknown_label", None, confidence);
            }
        }

        if policy.authority == Authority::Observe {
            return pass("observe_only", guess, confidence);
        }

        if self.output_kind != OutputKind::Multilabel
            && policy.never_resolve.contains(&reading.label)
        {
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

        if self.output_kind == OutputKind::Multilabel {
            // A multilabel head emits independent per-label probabilities. A
            // task resolves when at least one label clears its own bar; retain
            // the complete vector so consumers can inspect every decision.
            let selected = reading.scores.iter().filter(|(label, score)| {
                let threshold = policy.label_min_confidence.get(label)
                    .copied().unwrap_or(policy.min_confidence);
                score.is_finite()
                    && !policy.never_resolve.contains(label)
                    && if policy.strict { *score > threshold } else { *score >= threshold }
            }).max_by(|left, right| left.1.total_cmp(&right.1));
            if let Some((label, score)) = selected {
                return Decision::Resolve(Reading {
                    label: label.clone(),
                    confidence: score.clamp(0.0, 1.0),
                    scores: reading.scores.clone(),
                });
            }
            return pass("below_threshold", guess, confidence);
        }

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
    fn multilabel_resolution_applies_thresholds_independently() {
        let profile: Task = Task {
            cascade: Cascade::default(),
            name: String::from("risk_flags"),
            output_kind: OutputKind::Multilabel,
            labels: Some(vec!["security".into(), "privacy".into(), "other".into()]),
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        };
        let policy = TierPolicy::new(0.8).with_label("security", 0.95);
        let reading = TierReading::Scored(Reading {
            label: "security".into(),
            confidence: 0.91,
            scores: vec![
                ("security".into(), 0.91),
                ("privacy".into(), 0.84),
                ("other".into(), 0.12),
            ],
        });
        assert!(matches!(
            profile.decide(&policy, &reading),
            Decision::Resolve(Reading { label, confidence, .. })
                if label == "privacy" && (confidence - 0.84).abs() < 1e-6
        ));
    }

    #[test]
    fn an_observing_policy_records_the_reading_and_never_resolves() {
        let profile: Task = Task {
            cascade: crate::pipeline::task::Cascade::default(),
            name: String::from("urgency"),
            output_kind: OutputKind::Multiclass,
            labels: None,
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        };
        let reading = scored("high", 0.99);
        assert!(matches!(
            profile.decide(&TierPolicy::new(0.5), &reading),
            Decision::Resolve(_)
        ));
        assert!(matches!(
            profile.decide(&TierPolicy::new(0.5).observe(), &reading),
            Decision::Pass { reason: "observe_only", .. }
        ));
    }

    #[test]
    fn a_policy_can_name_the_instance_it_was_calibrated_for() {
        let toml = r#"
            min_confidence = 0.9
            authority = "observe"
            calibrated_for = "00112233445566778899aabbccddeeff"
        "#;
        let policy: TierPolicy = toml::from_str(toml).unwrap();
        assert_eq!(policy.authority, Authority::Observe);
        assert_eq!(policy.calibrated_for.map(|id| id.to_string()).as_deref(), Some("00112233445566778899aabbccddeeff"));
        assert!(toml::from_str::<TierPolicy>("min_confidence = 0.9\ncalibrated_for = \"nope\"").is_err());
    }

    #[test]
    fn unsupported_task_is_not_an_ordinary_model_abstention() {
        let profile: Task = Task {
            cascade: crate::pipeline::task::Cascade::default(),
            name: String::from("tag"),
            output_kind: OutputKind::Multiclass,
            labels: None,
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        };
        let decision = profile.decide(&TierPolicy::new(0.5), &TierReading::Unsupported);
        assert!(matches!(decision, Decision::Pass { reason: "unsupported_task", .. }));
    }

    #[test]
    fn task_profile_contains_no_tier_calibration() {
        let profile: Task = Task {
            cascade: crate::pipeline::task::Cascade::default(),
            name: String::from("route"),
            output_kind: OutputKind::Binary,
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Label(LABEL_LLM.into()),
        };
        assert_eq!(profile.name, String::from("route"));
        assert_eq!(profile.labels.as_ref().unwrap().len(), 2);
        assert_eq!(profile.fallback, FallbackPolicy::Label(LABEL_LLM.into()));
    }

    #[test]
    fn policy_is_supplied_independently_of_task_definition() {
        let profile: Task = Task {
            cascade: crate::pipeline::task::Cascade::default(),
            name: String::from("route"),
            output_kind: OutputKind::Binary,
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            label_metadata: Vec::new(),
            taxonomy_version: None,
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
        let profile: Task = Task {
            cascade: crate::pipeline::task::Cascade::default(),
            name: String::from("route"),
            output_kind: OutputKind::Binary,
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            label_metadata: Vec::new(),
            taxonomy_version: None,
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
