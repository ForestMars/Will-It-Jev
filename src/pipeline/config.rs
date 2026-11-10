// src/pipeline/config.rs
//
// Loads task definitions, the tier registry, and tier/task resolution policies
// as separate concerns.
//
//   [task.<id>]               what a task is
//   [[tier]]                  the available named tiers; no execution order
//   [tier_policy.<tier>.<id>] when a tier may resolve a task
//
// `<tier>` is a tier name. For configs written before tiers had names, the key
// `tierN` is accepted as an alias for the tier whose `legacy_ordinal` is N.

use std::collections::BTreeMap;
use std::{error::Error, fs, path::Path};

use serde::Deserialize;

use super::cascade::CascadeGraph;
use super::registry::TierId;
use super::task::{FallbackPolicy, OutputKind, TaskId, TaskLabelMetadata, TaskProfile, TierPolicy, TierPolicyMap};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSettings {
    #[serde(default)]
    pub tag_via_cascade: bool,
    /// Include sampled tagging as a best-effort task in the route request's cascade.
    #[serde(default)]
    pub tag_in_request: bool,
    #[serde(default = "default_tag_budget_ms")]
    pub tag_budget_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierExecutor { Rules, Fasttext, Scorer }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierTransport { InProcess, Grpc }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierRequestShape { Score, ScoreTasks }

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTier {
    name: String,
    executor: TierExecutor,
    transport: TierTransport,
    request_shape: Option<TierRequestShape>,
    legacy_ordinal: Option<u8>,
    enabled: Option<bool>,
    endpoint: Option<String>,
    timeout_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TierSpec {
    pub id: TierId,
    pub executor: TierExecutor,
    pub transport: TierTransport,
    pub request_shape: Option<TierRequestShape>,
    pub legacy_ordinal: Option<u8>,
    pub enabled: Option<bool>,
    pub endpoint: Option<String>,
    pub timeout_ms: Option<u64>,
}

/// The registry of available tiers. Its vector order has no scheduling meaning.
fn default_tier_specs() -> Vec<TierSpec> {
    [
        ("rules", TierExecutor::Rules, TierTransport::InProcess, None, 1),
        ("fasttext", TierExecutor::Fasttext, TierTransport::InProcess, None, 2),
        ("modernbert", TierExecutor::Scorer, TierTransport::Grpc, Some(TierRequestShape::Score), 3),
        ("harrier", TierExecutor::Scorer, TierTransport::Grpc, Some(TierRequestShape::Score), 4),
    ].into_iter().map(|(name, executor, transport, request_shape, ordinal)| TierSpec {
        id: TierId::new(name).expect("built-in tier names are valid"),
        executor, transport, request_shape, legacy_ordinal: Some(ordinal),
        enabled: None, endpoint: None, timeout_ms: None,
    }).collect()
}

fn tier_specs_from(raw: Vec<RawTier>) -> Result<Vec<TierSpec>, Box<dyn Error>> {
    if raw.is_empty() { return Ok(default_tier_specs()); }
    let mut specs: Vec<TierSpec> = Vec::with_capacity(raw.len());
    for tier in raw {
        let id = TierId::new(tier.name.clone()).map_err(|error| format!("[[tier]]: {error}"))?;
        let name = id.as_str().to_string();
        if specs.iter().any(|existing| existing.id == id) {
            return Err(format!("[[tier]] {name:?} is defined twice").into());
        }
        if let Some(ordinal) = tier.legacy_ordinal {
            if ordinal == 0 { return Err(format!("[[tier]] {name:?}: legacy_ordinal must be positive").into()); }
            if specs.iter().any(|existing| existing.legacy_ordinal == Some(ordinal)) {
                return Err(format!("[[tier]] {name:?}: legacy_ordinal {ordinal} is already used").into());
            }
        }
        match (tier.executor, tier.transport, tier.request_shape) {
            (TierExecutor::Rules | TierExecutor::Fasttext, TierTransport::InProcess, None) => {
                if tier.endpoint.is_some() || tier.timeout_ms.is_some() {
                    return Err(format!("[[tier]] {name:?}: endpoint/timeout_ms require grpc transport").into());
                }
            }
            (TierExecutor::Scorer, TierTransport::Grpc, Some(TierRequestShape::Score | TierRequestShape::ScoreTasks)) => {}
            (TierExecutor::Rules | TierExecutor::Fasttext, _, _) => {
                return Err(format!("[[tier]] {name:?}: rules/fasttext require transport = \"in_process\" and no request_shape").into());
            }
            (TierExecutor::Scorer, _, _) => {
                return Err(format!("[[tier]] {name:?}: scorer requires transport = \"grpc\" and request_shape = \"score\" or \"score_tasks\"").into());
            }
        }
        if tier.endpoint.as_deref().is_some_and(|value| value.trim().is_empty()) {
            return Err(format!("[[tier]] {name:?}: endpoint must be non-empty").into());
        }
        if tier.timeout_ms == Some(0) {
            return Err(format!("[[tier]] {name:?}: timeout_ms must be greater than zero").into());
        }
        specs.push(TierSpec {
            id, executor: tier.executor, transport: tier.transport,
            request_shape: tier.request_shape, legacy_ordinal: tier.legacy_ordinal,
            enabled: tier.enabled, endpoint: tier.endpoint, timeout_ms: tier.timeout_ms,
        });
    }
    let rules = specs.iter().filter(|spec| spec.executor == TierExecutor::Rules).count();
    let fasttext = specs.iter().filter(|spec| spec.executor == TierExecutor::Fasttext).count();
    if rules > 1 { return Err("only one rules tier may be registered".into()); }
    if fasttext > 1 { return Err("only one fasttext tier may be registered".into()); }
    Ok(specs)
}

fn find_tier<'a>(specs: &'a [TierSpec], reference: &str) -> Option<&'a TierSpec> {
    if let Some(spec) = specs.iter().find(|spec| spec.id.as_str() == reference) {
        return Some(spec);
    }
    reference
        .strip_prefix("tier")
        .and_then(|number| number.parse::<u8>().ok())
        .and_then(|ordinal| specs.iter().find(|spec| spec.legacy_ordinal == Some(ordinal)))
}

fn known_tier_names(specs: &[TierSpec]) -> String {
    specs.iter().map(|spec| spec.id.as_str()).collect::<Vec<_>>().join(", ")
}

fn default_tag_budget_ms() -> u64 {
    4000
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            tag_via_cascade: false,
            tag_in_request: false,
            tag_budget_ms: default_tag_budget_ms(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProfile {
    labels: Option<Vec<String>>,
    /// Ordered TierId path for this task.
    #[serde(default)]
    cascade: Option<Vec<String>>,
    #[serde(default)]
    output_kind: OutputKind,
    taxonomy: Option<String>,
    fallback: Option<String>,
    fallback_label: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawTaxonomy {
    version: String,
    categories: Vec<RawCategory>,
}

#[derive(Debug, Deserialize)]
struct RawCategory {
    name: String,
    #[serde(default)]
    desc: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    classifier_criterion: String,
    #[serde(default)]
    group_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct RawFile {
    #[serde(default)]
    pipeline: PipelineSettings,
    #[serde(default)]
    task: BTreeMap<String, RawProfile>,
    #[serde(default)]
    tier: Vec<RawTier>,
    #[serde(default)]
    tier_policy: BTreeMap<String, BTreeMap<String, TierPolicy>>,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub settings: PipelineSettings,
    /// Available configured tiers; vector order carries no scheduling semantics.
    pub tiers: Vec<TierSpec>,
    /// Union DAG formed from all task-specific cascade paths.
    pub cascades: CascadeGraph,
    profiles: BTreeMap<TaskId, TaskProfile>,
    /// Tier policies are a separate registry, not fields on TaskProfile.
    pub tier_policies: TierPolicyMap,
}

impl PipelineConfig {
    pub fn profile(&self, task: &TaskId) -> Option<TaskProfile> {
        self.profiles.get(task).cloned()
    }

    /// All configured task contracts, returned in stable task-ID order.
    pub fn profiles(&self) -> Vec<TaskProfile> {
        self.profiles.values().cloned().collect()
    }

    pub fn has_policy(&self, task: &TaskId) -> bool {
        self.tier_policies
            .values()
            .any(|by_task| by_task.contains_key(task))
    }
}

fn validate_policy(task: &str, tier: &TierId, policy: &TierPolicy) -> Result<(), Box<dyn Error>> {
    for bar in std::iter::once(policy.min_confidence)
        .chain(policy.label_min_confidence.values().copied())
    {
        if !bar.is_finite() || !(0.0..=1.0).contains(&bar) {
            return Err(format!(
                "tier_policy.{tier}.{task}: confidence bar {bar} is outside 0..=1"
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
    let contents = fs::read_to_string(path)?;
    parse_with_base(&contents, path.parent().unwrap_or_else(|| Path::new(".")))
}

pub fn parse(contents: &str) -> Result<PipelineConfig, Box<dyn Error>> {
    parse_with_base(contents, Path::new("."))
}

fn parse_with_base(contents: &str, base: &Path) -> Result<PipelineConfig, Box<dyn Error>> {
    let raw: RawFile = toml::from_str(contents)?;
    let tiers = tier_specs_from(raw.tier)?;
    let registered: std::collections::BTreeSet<TierId> = tiers.iter().map(|spec| spec.id.clone()).collect();
    let mut profiles = BTreeMap::new();
    let mut task_paths = BTreeMap::new();

    for (name, profile) in raw.task {
        let task = TaskId::new(name.clone());
        let path = match profile.cascade {
            Some(names) => names.into_iter().map(|tier_name| {
                TierId::new(tier_name.clone()).map_err(|error| format!("task.{name}.cascade: {error}"))
            }).collect::<Result<Vec<_>, _>>()?,
            None => tiers.iter().map(|spec| spec.id.clone()).collect(),
        };
        task_paths.insert(task.clone(), path);

        let (labels, label_metadata, taxonomy_version) = if let Some(taxonomy_path) = &profile.taxonomy {
            let taxonomy_file = base.join(taxonomy_path);
            let taxonomy: RawTaxonomy = serde_json::from_str(&fs::read_to_string(&taxonomy_file)?)
                .map_err(|error| format!("task.{name}: failed to load {}: {error}", taxonomy_file.display()))?;
            let mut labels = Vec::with_capacity(taxonomy.categories.len());
            let mut metadata = Vec::with_capacity(taxonomy.categories.len());
            for category in taxonomy.categories {
                labels.push(category.name.clone());
                metadata.push(TaskLabelMetadata {
                    id: category.name,
                    name: category.desc,
                    description: category.description,
                    classifier_criterion: category.classifier_criterion,
                    group_id: category.group_id,
                });
            }
            (Some(labels), metadata, Some(taxonomy.version))
        } else {
            (profile.labels.clone(), Vec::new(), None)
        };
        let fallback = match (profile.fallback.as_deref(), profile.fallback_label) {
            (None | Some("defer"), None) => FallbackPolicy::Defer,
            (None, Some(label)) => FallbackPolicy::Label(label),
            (Some(_), Some(_)) => return Err(format!("task.{name}: set fallback or fallback_label, not both").into()),
            (Some(other), None) => return Err(format!("task.{name}: fallback must be \"defer\", got {other:?}").into()),
        };
        if let Some(labels) = &labels {
            let unique: std::collections::BTreeSet<_> = labels.iter().collect();
            if unique.len() != labels.len() || labels.iter().any(|label| label.trim().is_empty()) {
                return Err(format!("task.{name}: labels must be non-empty and unique").into());
            }
        }
        if let (Some(labels), FallbackPolicy::Label(label)) = (&labels, &fallback) {
            if !labels.contains(label) {
                return Err(format!("task.{name}: fallback label {label:?} is not in labels").into());
            }
        }
        profiles.insert(task.clone(), TaskProfile {
            task, output_kind: profile.output_kind, labels, label_metadata,
            taxonomy_version, fallback,
        });
    }

    let cascades = CascadeGraph::from_task_paths(task_paths, &registered)?;
    let mut tier_policies = TierPolicyMap::new();
    for (tier_key, task_policies) in raw.tier_policy {
        let spec = find_tier(&tiers, &tier_key).ok_or_else(|| format!(
            "tier_policy: unknown tier {tier_key:?}; registered tiers: {}", known_tier_names(&tiers)
        ))?;
        let tier = spec.id.clone();
        let by_task = tier_policies.entry(tier.clone()).or_default();
        for (task_name, policy) in task_policies {
            let task = TaskId::new(task_name.clone());
            let profile = profiles.get(&task).ok_or_else(|| format!(
                "tier_policy.{tier}.{task_name}: no matching [task.{task_name}] definition"
            ))?;
            if !cascades.task_uses(&task, &tier) {
                return Err(format!("tier_policy.{tier}.{task_name}: tier is not in task.{task_name}.cascade").into());
            }
            validate_policy(&task_name, &tier, &policy)?;
            if let Some(labels) = &profile.labels {
                for label in policy.label_min_confidence.keys().chain(policy.never_resolve.iter()) {
                    if !labels.contains(label) {
                        return Err(format!("tier_policy.{tier}.{task_name}: policy references unknown label {label:?}").into());
                    }
                }
            }
            if by_task.insert(task, policy).is_some() {
                return Err(format!(
                    "tier_policy.{tier}.{task_name}: defined twice (a tier name and its tierN alias both appear)"
                ).into());
            }
        }
    }
    Ok(PipelineConfig { settings: raw.pipeline, tiers, cascades, profiles, tier_policies })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_keep_distinct_paths_and_share_nodes() {
        let cfg = parse(r#"
            [task.route]
            labels = ["jev_capable", "needs_llm"]
            fallback_label = "needs_llm"
            cascade = ["rules", "modernbert", "harrier"]
            [task.tag]
            labels = ["engineering", "help_desk"]
            fallback = "defer"
            cascade = ["modernbert", "harrier"]
            [[tier]]
            name = "rules"
            executor = "rules"
            transport = "in_process"
            [[tier]]
            name = "modernbert"
            executor = "scorer"
            transport = "grpc"
            request_shape = "score_tasks"
            endpoint = "http://127.0.0.1:50052"
            [[tier]]
            name = "harrier"
            executor = "scorer"
            transport = "grpc"
            request_shape = "score_tasks"
            endpoint = "http://127.0.0.1:50051"
        "#).unwrap();
        assert_eq!(cfg.cascades.order().iter().map(TierId::as_str).collect::<Vec<_>>(), ["rules", "modernbert", "harrier"]);
        assert_eq!(cfg.cascades.tasks_using(&TierId::new("modernbert").unwrap()).unwrap().len(), 2);
        assert!(cfg.cascades.task_uses(&TaskId::tag(), &TierId::new("harrier").unwrap()));
    }

    #[test]
    fn conflicting_task_orders_fail_as_a_cycle() {
        let result = parse(r#"
            [task.route]
            fallback_label = "needs_llm"
            cascade = ["a", "b"]
            [task.tag]
            fallback = "defer"
            cascade = ["b", "a"]
            [[tier]]
            name = "a"
            executor = "scorer"
            transport = "grpc"
            request_shape = "score"
            endpoint = "http://127.0.0.1:50051"
            [[tier]]
            name = "b"
            executor = "scorer"
            transport = "grpc"
            request_shape = "score"
            endpoint = "http://127.0.0.1:50052"
        "#);
        assert!(matches!(result, Err(error) if error.to_string().contains("cycle")));
    }

    #[test]
    fn omitted_paths_keep_the_legacy_default_sequence() {
        let cfg = parse(r#"
            [task.route]
            fallback_label = "needs_llm"
            [[tier]]
            name = "zeta"
            executor = "scorer"
            transport = "grpc"
            request_shape = "score"
            endpoint = "http://127.0.0.1:50051"
            [[tier]]
            name = "alpha"
            executor = "scorer"
            transport = "grpc"
            request_shape = "score"
            endpoint = "http://127.0.0.1:50052"
        "#).unwrap();
        assert_eq!(cfg.cascades.paths()[&TaskId::route()].iter().map(TierId::as_str).collect::<Vec<_>>(), ["zeta", "alpha"]);
    }

    #[test]
    fn rejects_unknown_and_duplicate_tier_ids_in_task_paths() {
        let prefix = "[task.route]\nfallback_label = \"needs_llm\"\n";
        let tier = "[[tier]]\nname=\"rules\"\nexecutor=\"rules\"\ntransport=\"in_process\"\n";
        assert!(parse(&format!("{prefix}cascade = [\"rules\", \"unknown\"]\n{tier}")).is_err());
        assert!(parse(&format!("{prefix}cascade = [\"rules\", \"rules\"]\n{tier}")).is_err());
    }

    #[test]
    fn validates_executor_transport_and_request_shape_separately() {
        let prefix = "[task.route]\nfallback_label = \"needs_llm\"\n";
        for tier in [
            "[[tier]]\nname=\"x\"\nexecutor=\"rules\"\ntransport=\"grpc\"",
            "[[tier]]\nname=\"x\"\nexecutor=\"scorer\"\ntransport=\"in_process\"",
            "[[tier]]\nname=\"x\"\nexecutor=\"scorer\"\ntransport=\"grpc\"\nrequest_shape=\"other\"",
            "[[tier]]\nname=\"x\"\nexecutor=\"rules\"\ntransport=\"in_process\"\nrequest_shape=\"score\"",
        ] {
            assert!(parse(&format!("{prefix}{tier}")).is_err(), "accepted invalid spec: {tier}");
        }
    }

    #[test]
    fn preserves_taxonomy_and_policy_validation() {
        let cfg = parse_with_base(
            "[task.tag]\ntaxonomy = \"../taskonomies/help_desk.json\"\noutput_kind = \"multiclass\"\nfallback = \"defer\"",
            Path::new("config"),
        ).unwrap();
        let tag = cfg.profile(&TaskId::tag()).unwrap();
        assert_eq!(tag.output_kind, OutputKind::Multiclass);
        assert_eq!(tag.labels.as_ref().unwrap().len(), 20);
        assert_eq!(tag.taxonomy_version.as_deref(), Some("1.1.0"));
        assert!(tag.label_metadata.iter().any(|row| row.id == "app_access_req"));
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n[tier_policy.tier0.route]\nmin_confidence = 0.5").is_err());
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n[tier_policy.unknown.route]\nmin_confidence = 0.5").is_err());
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n[tier_policy.tier3.route]\nmin_confidence = 1.5").is_err());
    }

    #[test]
    fn shipped_config_references_available_tiers() {
        let cfg = load_or_default("config/tasks.toml").unwrap();
        let available: std::collections::BTreeSet<_> = cfg.tiers.iter().map(|spec| spec.id.clone()).collect();
        for path in cfg.cascades.paths().values() {
            assert!(path.iter().all(|tier| available.contains(tier)));
        }
        assert!(cfg.has_policy(&TaskId::route()));
        assert!(!cfg.cascades.order().is_empty());
    }
}
