// src/pipeline/config.rs
//
// Loads task profiles and pipeline switches from TOML (config/tasks.toml).
//
//   [pipeline]
//   tag_via_cascade = false      # route tagging through the shared tiers
//   tag_budget_ms   = 4000       # deadline for one tagging run
//   joint_tiers     = []         # tier numbers whose sidecar implements ScoreTasks
//
//   [task.tag]
//   labels   = ["engineering", ...]
//   fallback = "defer"           # or: fallback_label = "needs_llm"
//   [task.tag.tier3]
//   min_confidence = 0.8
//   [task.tag.tier3.label_min_confidence]
//   engineering = 0.9

use std::collections::BTreeMap;
use std::{error::Error, fs, path::Path};

use serde::Deserialize;

use super::task::{FallbackPolicy, TaskId, TaskProfile, TierPolicy};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSettings {
    #[serde(default)]
    pub tag_via_cascade: bool,
    #[serde(default = "default_tag_budget_ms")]
    pub tag_budget_ms: u64,
    #[serde(default)]
    pub joint_tiers: Vec<u8>,
}

fn default_tag_budget_ms() -> u64 {
    4000
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self { tag_via_cascade: false, tag_budget_ms: default_tag_budget_ms(), joint_tiers: Vec::new() }
    }
}

#[derive(Debug, Deserialize)]
struct RawProfile {
    labels: Option<Vec<String>>,
    fallback: Option<String>,
    fallback_label: Option<String>,
    #[serde(flatten)]
    tiers: BTreeMap<String, TierPolicy>,
}

#[derive(Debug, Default, Deserialize)]
struct RawFile {
    #[serde(default)]
    pipeline: PipelineSettings,
    #[serde(default)]
    task: BTreeMap<String, RawProfile>,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub settings: PipelineSettings,
    profiles: BTreeMap<TaskId, TaskProfile>,
}

impl PipelineConfig {
    /// The configured profile, or for `route` the built-in one that mirrors the
    /// hard-coded cascade. Any other task has no default: its thresholds have to be
    /// tuned against that task's own data.
    pub fn profile(&self, task: &TaskId) -> Option<TaskProfile> {
        self.profiles
            .get(task)
            .cloned()
            .or_else(|| (*task == TaskId::route()).then(TaskProfile::route_default))
    }
}

/// A missing file yields the defaults (everything off).
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
            (Some(other), None) => {
                return Err(format!("task.{name}: fallback must be \"defer\", got {other:?}").into())
            }
            (Some(_), Some(_)) => {
                return Err(format!("task.{name}: set fallback or fallback_label, not both").into())
            }
        };
        let mut tiers = BTreeMap::new();
        for (key, policy) in profile.tiers {
            let tier: u8 = key
                .strip_prefix("tier")
                .and_then(|n| n.parse().ok())
                .filter(|n| (1..=4).contains(n))
                .ok_or_else(|| format!("task.{name}: unknown key {key:?}; tiers are tier1..tier4"))?;
            for bar in std::iter::once(policy.min_confidence).chain(policy.label_min_confidence.values().copied()) {
                if !(0.0..=1.0).contains(&bar) {
                    return Err(format!("task.{name}.{key}: confidence bar {bar} is outside 0..=1").into());
                }
            }
            tiers.insert(tier, policy);
        }
        let task = TaskId::new(name);
        profiles.insert(task.clone(), TaskProfile { task, labels: profile.labels, tiers, fallback });
    }
    Ok(PipelineConfig { settings: raw.pipeline, profiles })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_is_everything_off_with_route_default() {
        let cfg = parse("").unwrap();
        assert!(!cfg.settings.tag_via_cascade);
        assert!(cfg.profile(&TaskId::tag()).is_none());
        assert!(cfg.profile(&TaskId::route()).is_some());
    }

    #[test]
    fn parses_a_tag_profile() {
        let cfg = parse(
            r#"
            [pipeline]
            tag_via_cascade = true
            joint_tiers = [4]
            [task.tag]
            labels = ["a", "b"]
            [task.tag.tier4]
            min_confidence = 0.8
            [task.tag.tier4.label_min_confidence]
            a = 0.9
            "#,
        )
        .unwrap();
        assert_eq!(cfg.settings.joint_tiers, vec![4]);
        let tag = cfg.profile(&TaskId::tag()).unwrap();
        assert_eq!(tag.fallback, FallbackPolicy::Defer);
        assert_eq!(tag.tiers[&4].label_min_confidence["a"], 0.9);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse("[task.tag.tier9]\nmin_confidence = 0.5").is_err());
        assert!(parse("[task.tag.tier3]\nmin_confidence = 1.5").is_err());
        assert!(parse("[task.tag]\nfallback = \"defer\"\nfallback_label = \"x\"").is_err());
        assert!(parse("[task.tag]\nbogus = 1").is_err());
        assert!(parse("[pipeline]\nnope = true").is_err());
    }

    #[test]
    fn shipped_config_parses() {
        let cfg = load_or_default("config/tasks.toml").unwrap();
        assert!(!cfg.settings.tag_via_cascade);
    }
}
