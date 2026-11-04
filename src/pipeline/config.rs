// src/pipeline/config.rs
//
// Loads task definitions and tier/task resolution policies as separate concerns.
// Task profiles are defined under [task.*]. Tier calibration is defined under
// [tier_policy.tierN.task_id].

use std::collections::BTreeMap;
use std::{error::Error, fs, path::Path};

use serde::Deserialize;

use super::task::{FallbackPolicy, TaskId, TaskProfile, TierPolicy, TierPolicyMap};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSettings {
    #[serde(default)]
    pub tag_via_cascade: bool,
    #[serde(default = "default_tag_budget_ms")]
    pub tag_budget_ms: u64,
    /// Execution tiers whose sidecars implement the joint ScoreTasks RPC.
    #[serde(default)]
    pub joint_tiers: Vec<u8>,
}

fn default_tag_budget_ms() -> u64 {
    4000
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            tag_via_cascade: false,
            tag_budget_ms: default_tag_budget_ms(),
            joint_tiers: Vec::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProfile {
    labels: Option<Vec<String>>,
    fallback: Option<String>,
    fallback_label: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct RawFile {
    #[serde(default)]
    pipeline: PipelineSettings,
    #[serde(default)]
    task: BTreeMap<String, RawProfile>,
    #[serde(default)]
    tier_policy: BTreeMap<String, BTreeMap<String, TierPolicy>>,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub settings: PipelineSettings,
    profiles: BTreeMap<TaskId, TaskProfile>,
    /// Tier policies are a separate registry, not fields on TaskProfile.
    pub tier_policies: TierPolicyMap,
}

impl PipelineConfig {
    pub fn profile(&self, task: &TaskId) -> Option<TaskProfile> {
        self.profiles.get(task).cloned()
    }

    pub fn has_policy(&self, task: &TaskId) -> bool {
        self.tier_policies
            .values()
            .any(|by_task| by_task.contains_key(task))
    }
}

fn validate_policy(task: &str, tier: u8, policy: &TierPolicy) -> Result<(), Box<dyn Error>> {
    for bar in std::iter::once(policy.min_confidence)
        .chain(policy.label_min_confidence.values().copied())
    {
        if !bar.is_finite() || !(0.0..=1.0).contains(&bar) {
            return Err(format!(
                "tier_policy.tier{tier}.{task}: confidence bar {bar} is outside 0..=1"
            )
            .into());
        }
    }
    Ok(())
}

/// A missing file yields an empty configuration. The application validates that
/// the required route task and policies are explicitly configured.
pub fn load_or_default(path: impl AsRef<Path>) -> Result<PipelineConfig, Box<dyn Error>> {
    let path = path.as_ref();
    if !path.exists() {
        return parse("");
    }
    parse(&fs::read_to_string(path)?)
}

pub fn parse(contents: &str) -> Result<PipelineConfig, Box<dyn Error>> {
    let raw: RawFile = toml::from_str(contents)?;
    let mut profiles = BTreeMap::new();

    for (name, profile) in raw.task {
        let fallback = match (profile.fallback.as_deref(), profile.fallback_label) {
            (None | Some("defer"), None) => FallbackPolicy::Defer,
            (None, Some(label)) => FallbackPolicy::Label(label),
            (Some(_), Some(_)) => {
                return Err(format!(
                    "task.{name}: set fallback or fallback_label, not both"
                )
                .into())
            }
            (Some(other), None) => {
                return Err(format!(
                    "task.{name}: fallback must be \"defer\", got {other:?}"
                )
                .into())
            }
        };
        if let Some(labels) = &profile.labels {
            let unique: std::collections::BTreeSet<_> = labels.iter().collect();
            if unique.len() != labels.len() || labels.iter().any(|label| label.trim().is_empty()) {
                return Err(format!("task.{name}: labels must be non-empty and unique").into());
            }
        }
        if let (Some(labels), FallbackPolicy::Label(label)) = (&profile.labels, &fallback) {
            if !labels.contains(label) {
                return Err(format!("task.{name}: fallback label {label:?} is not in labels").into());
            }
        }
        let task = TaskId::new(name);
        profiles.insert(
            task.clone(),
            TaskProfile {
                task,
                labels: profile.labels,
                fallback,
            },
        );
    }

    let mut tier_policies = TierPolicyMap::new();
    for (tier_key, task_policies) in raw.tier_policy {
        let tier: u8 = tier_key
            .strip_prefix("tier")
            .and_then(|number| number.parse().ok())
            .filter(|number| *number > 0)
            .ok_or_else(|| {
                format!("tier_policy: invalid key {tier_key:?}; expected tier followed by a positive u8")
            })?;

        let by_task = tier_policies.entry(tier).or_default();
        for (task_name, policy) in task_policies {
            let task = TaskId::new(task_name.clone());
            let profile = profiles.get(&task).ok_or_else(|| {
                format!(
                    "tier_policy.tier{tier}.{task_name}: no matching [task.{task_name}] definition"
                )
            })?;
            validate_policy(&task_name, tier, &policy)?;
            if let Some(labels) = &profile.labels {
                for label in policy
                    .label_min_confidence
                    .keys()
                    .chain(policy.never_resolve.iter())
                {
                    if !labels.contains(label) {
                        return Err(format!(
                            "tier_policy.tier{tier}.{task_name}: policy references unknown label {label:?}"
                        )
                        .into());
                    }
                }
            }
            by_task.insert(task, policy);
        }
    }

    Ok(PipelineConfig {
        settings: raw.pipeline,
        profiles,
        tier_policies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_does_not_invent_tasks_or_tier_policies() {
        let cfg = parse("").unwrap();
        assert!(!cfg.settings.tag_via_cascade);
        assert!(cfg.profile(&TaskId::tag()).is_none());
        assert!(cfg.profile(&TaskId::route()).is_none());
        assert!(!cfg.has_policy(&TaskId::route()));
    }

    #[test]
    fn task_definition_and_tier_policy_parse_separately() {
        let cfg = parse(
            r#"
            [pipeline]
            tag_via_cascade = true
            joint_tiers = [4]

            [task.tag]
            labels = ["engineering", "help_desk"]
            fallback = "defer"

            [tier_policy.tier4.tag]
            min_confidence = 0.8

            [tier_policy.tier4.tag.label_min_confidence]
            engineering = 0.9
            "#,
        )
        .unwrap();

        assert_eq!(cfg.settings.joint_tiers, vec![4]);
        let tag = cfg.profile(&TaskId::tag()).unwrap();
        assert_eq!(tag.fallback, FallbackPolicy::Defer);
        assert_eq!(
            cfg.tier_policies[&4][&TaskId::tag()].label_min_confidence["engineering"],
            0.9
        );
    }

    #[test]
    fn rejects_unknown_tiers_bad_thresholds_and_unknown_task_references() {
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n\n[tier_policy.tier0.route]\nmin_confidence = 0.5").is_err());
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n\n[tier_policy.tier3.route]\nmin_confidence = 1.5").is_err());
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n\n[tier_policy.tier3.tag]\nmin_confidence = 0.5").is_err());
        assert!(parse("[task.tag]\nfallback = \"defer\"\nfallback_label = \"x\"").is_err());
        assert!(parse("[task.tag]\nlabels = [\"engineering\"]\nfallback = \"defer\"\n\n[tier_policy.tier3.tag]\nmin_confidence = 0.5\nnever_resolve = [\"unknown\"]").is_err());
    }
}
