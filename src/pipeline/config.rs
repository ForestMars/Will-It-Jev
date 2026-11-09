// src/pipeline/config.rs
//
// Task contracts, reusable tier definitions, configured tier instances and
// instance/task policies are separate configuration concerns.
//
//   [task.<id>]                         task contract
//   [[tier]] name + tier + model_id     configured tier instance
//   [tier_policy.<instance>.<task>]    policy for one configured instance
//
// [[tier]] array order is execution order. It contains instance identities,
// never positions or numeric ordinals.

use std::collections::BTreeMap;
use std::{error::Error, fs, path::Path};

use serde::Deserialize;

use super::registry::{TierDefinitionRegistry, TierId, TierInstanceId};
use super::task::{FallbackPolicy, OutputKind, TaskId, TaskLabelMetadata, TaskProfile, TierPolicy, TierPolicyMap};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSettings {
    #[serde(default)]
    pub tag_via_cascade: bool,
    #[serde(default)]
    pub tag_in_request: bool,
    #[serde(default = "default_tag_budget_ms")]
    pub tag_budget_ms: u64,
    /// Instance names whose sidecars implement the joint ScoreTasks RPC.
    #[serde(default)]
    pub joint_tiers: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierKind {
    Rules,
    Fasttext,
    Grpc,
}

fn kind_for_definition(id: &TierId) -> Result<TierKind, Box<dyn Error>> {
    match id.as_str() {
        "rules" => Ok(TierKind::Rules),
        "fasttext" => Ok(TierKind::Fasttext),
        "grpc" => Ok(TierKind::Grpc),
        other => Err(format!("unknown tier definition {other:?}").into()),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTier {
    /// Stable identity of this configured realization.
    name: String,
    /// Reusable execution contract implemented by the instance.
    tier: String,
    /// Selected model artifact/configuration for this instance.
    model_id: Option<String>,
    #[serde(default)]
    joint: bool,
    #[serde(default)]
    enabled: Option<bool>,
    endpoint: Option<String>,
    timeout_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TierSpec {
    pub id: TierInstanceId,
    pub tier_id: TierId,
    pub kind: TierKind,
    pub model_id: Option<String>,
    pub joint: bool,
    pub enabled: Option<bool>,
    pub endpoint: Option<String>,
    pub timeout_ms: Option<u64>,
}

fn default_tier_specs() -> Vec<TierSpec> {
    [
        ("rules", "rules", None, TierKind::Rules, false, None, None),
        ("fasttext", "fasttext", Some("fasttext-default"), TierKind::Fasttext, false, None, None),
        ("modernbert", "grpc", Some("modernbert"), TierKind::Grpc, false, Some("http://[::1]:50052"), Some(500)),
        ("harrier", "grpc", Some("harrier"), TierKind::Grpc, false, Some("http://[::1]:50051"), Some(500)),
    ].into_iter().map(|(name, tier, model_id, kind, joint, endpoint, timeout_ms)| TierSpec {
        id: TierInstanceId::new(name).expect("built-in instance name is valid"),
        tier_id: TierId::new(tier).expect("built-in definition name is valid"),
        kind,
        model_id: model_id.map(str::to_string),
        joint,
        enabled: None,
        endpoint: endpoint.map(str::to_string),
        timeout_ms,
    }).collect()
}

fn tier_specs_from(
    raw: Vec<RawTier>,
    definitions: &TierDefinitionRegistry,
) -> Result<Vec<TierSpec>, Box<dyn Error>> {
    if raw.is_empty() {
        return Ok(default_tier_specs());
    }
    let mut specs = Vec::with_capacity(raw.len());
    for tier in raw {
        let id = TierInstanceId::new(tier.name.clone())
            .map_err(|error| format!("[[tier]] name: {error}"))?;
        if specs.iter().any(|existing: &TierSpec| existing.id == id) {
            return Err(format!("[[tier]] instance {:?} is defined twice", id.as_str()).into());
        }
        let tier_id = TierId::new(tier.tier.clone())
            .map_err(|error| format!("[[tier]] tier: {error}"))?;
        if !definitions.contains(&tier_id) {
            return Err(format!("[[tier]] {} references unregistered definition {tier_id}", id).into());
        }
        let kind = kind_for_definition(&tier_id)?;
        if kind != TierKind::Grpc && (tier.endpoint.is_some() || tier.timeout_ms.is_some()) {
            return Err(format!("[[tier]] {id}: endpoint and timeout_ms apply to grpc instances only").into());
        }
        if kind != TierKind::Grpc && tier.joint {
            return Err(format!("[[tier]] {id}: joint applies to grpc instances only").into());
        }
        if tier.endpoint.as_deref().is_some_and(|endpoint| endpoint.trim().is_empty()) {
            return Err(format!("[[tier]] {id}: endpoint must be non-empty").into());
        }
        if tier.timeout_ms == Some(0) {
            return Err(format!("[[tier]] {id}: timeout_ms must be greater than zero").into());
        }
        if tier.model_id.as_deref().is_some_and(|model| model.trim().is_empty()) {
            return Err(format!("[[tier]] {id}: model_id must be non-empty when supplied").into());
        }
        specs.push(TierSpec {
            id,
            tier_id,
            kind,
            model_id: tier.model_id,
            joint: tier.joint,
            enabled: tier.enabled,
            endpoint: tier.endpoint,
            timeout_ms: tier.timeout_ms,
        });
    }
    Ok(specs)
}

fn find_instance<'a>(specs: &'a [TierSpec], reference: &str) -> Option<&'a TierSpec> {
    specs.iter().find(|spec| spec.id.as_str() == reference)
}

fn known_instance_names(specs: &[TierSpec]) -> String {
    specs.iter().map(|spec| spec.id.as_str()).collect::<Vec<_>>().join(", ")
}

fn default_tag_budget_ms() -> u64 { 4000 }

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            tag_via_cascade: false,
            tag_in_request: false,
            tag_budget_ms: default_tag_budget_ms(),
            joint_tiers: Vec::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProfile {
    labels: Option<Vec<String>>,
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
    /// Configured instances in execution order.
    pub tiers: Vec<TierSpec>,
    profiles: BTreeMap<TaskId, TaskProfile>,
    /// Policies are keyed by TierInstanceId, then TaskId.
    pub tier_policies: TierPolicyMap,
}

impl PipelineConfig {
    pub fn profile(&self, task: &TaskId) -> Option<TaskProfile> {
        self.profiles.get(task).cloned()
    }

    pub fn profiles(&self) -> Vec<TaskProfile> {
        self.profiles.values().cloned().collect()
    }

    pub fn has_policy(&self, task: &TaskId) -> bool {
        self.tier_policies.values().any(|by_task| by_task.contains_key(task))
    }
}

fn validate_policy(task: &str, instance: &TierInstanceId, policy: &TierPolicy) -> Result<(), Box<dyn Error>> {
    for bar in std::iter::once(policy.min_confidence).chain(policy.label_min_confidence.values().copied()) {
        if !bar.is_finite() || !(0.0..=1.0).contains(&bar) {
            return Err(format!("tier_policy.{instance}.{task}: confidence bar {bar} is outside 0..=1").into());
        }
    }
    Ok(())
}

pub fn load_or_default(path: impl AsRef<Path>) -> Result<PipelineConfig, Box<dyn Error>> {
    let path = path.as_ref();
    if !path.exists() { return parse(""); }
    let contents = fs::read_to_string(path)?;
    parse_with_base(&contents, path.parent().unwrap_or_else(|| Path::new(".")))
}

pub fn parse(contents: &str) -> Result<PipelineConfig, Box<dyn Error>> {
    parse_with_base(contents, Path::new("."))
}

fn parse_with_base(contents: &str, base: &Path) -> Result<PipelineConfig, Box<dyn Error>> {
    let raw: RawFile = toml::from_str(contents)?;
    let definitions = TierDefinitionRegistry::builtins();
    let mut tiers = tier_specs_from(raw.tier, &definitions)?;
    let mut profiles = BTreeMap::new();

    for (name, profile) in raw.task {
        // Taxonomy labels and descriptions are canonical source data. Inline
        // labels remain supported for small non-taxonomy tasks such as route.
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
        let task = TaskId::new(name);
        profiles.insert(
            task.clone(),
            TaskProfile {
                task,
                output_kind: profile.output_kind,
                labels,
                label_metadata,
                taxonomy_version,
                fallback,
            },
        );
    }

    let mut tier_policies = TierPolicyMap::new();
    for (instance_key, task_policies) in raw.tier_policy {
        let spec = find_instance(&tiers, &instance_key).ok_or_else(|| {
            format!(
                "tier_policy: unknown instance {instance_key:?}; configured instances: {}",
                known_instance_names(&tiers)
            )
        })?;
        let instance = spec.id.clone();
        let by_task = tier_policies.entry(instance.clone()).or_default();
        for (task_name, policy) in task_policies {
            let task = TaskId::new(task_name.clone());
            let profile = profiles.get(&task).ok_or_else(|| {
                format!("tier_policy.{instance}.{task_name}: no matching [task.{task_name}] definition")
            })?;
            validate_policy(&task_name, &instance, &policy)?;
            if let Some(labels) = &profile.labels {
                for label in policy.label_min_confidence.keys().chain(policy.never_resolve.iter()) {
                    if !labels.contains(label) {
                        return Err(format!("tier_policy.{instance}.{task_name}: policy references unknown label {label:?}").into());
                    }
                }
            }
            if by_task.insert(task, policy).is_some() {
                return Err(format!("tier_policy.{instance}.{task_name}: defined twice").into());
            }
        }
    }

    for name in &raw.pipeline.joint_tiers {
        let spec = find_instance(&tiers, name).ok_or_else(|| {
            format!("pipeline.joint_tiers: unknown instance {name:?}; configured instances: {}", known_instance_names(&tiers))
        })?;
        if spec.kind != TierKind::Grpc {
            return Err(format!("pipeline.joint_tiers: {name} is not a grpc instance").into());
        }
        let id = spec.id.clone();
        tiers.iter_mut().find(|spec| spec.id == id).expect("instance was just found").joint = true;
    }

    Ok(PipelineConfig { settings: raw.pipeline, tiers, profiles, tier_policies })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
        [task.route]
        fallback_label = "needs_llm"
        labels = ["jev_capable", "needs_llm"]

        [[tier]]
        name = "fasttext"
        tier = "fasttext"
        model_id = "fasttext-v1"

        [[tier]]
        name = "encoder_blue"
        tier = "grpc"
        model_id = "modernbert-base"
        endpoint = "http://127.0.0.1:50061"
        timeout_ms = 150
        joint = true

        [[tier]]
        name = "encoder_green"
        tier = "grpc"
        model_id = "harrier-v1"
        endpoint = "http://127.0.0.1:50062"
        timeout_ms = 250

        [tier_policy.fasttext.route]
        min_confidence = 0.95

        [tier_policy.encoder_blue.route]
        min_confidence = 0.80

        [tier_policy.encoder_green.route]
        min_confidence = 0.90
    "#;

    #[test]
    fn instance_vector_order_is_the_execution_order() {
        let cfg = parse(BASE).unwrap();
        let names: Vec<_> = cfg.tiers.iter().map(|tier| tier.id.as_str()).collect();
        assert_eq!(names, ["fasttext", "encoder_blue", "encoder_green"]);
        assert_eq!(cfg.tiers[1].tier_id, cfg.tiers[2].tier_id);
        assert_eq!(cfg.tiers[1].model_id.as_deref(), Some("modernbert-base"));
        assert_eq!(cfg.tiers[2].model_id.as_deref(), Some("harrier-v1"));
    }

    #[test]
    fn policies_are_instance_scoped() {
        let cfg = parse(BASE).unwrap();
        assert!(cfg.tier_policies.contains_key(&TierInstanceId::new("encoder_blue").unwrap()));
        assert!(cfg.tier_policies.contains_key(&TierInstanceId::new("encoder_green").unwrap()));
        assert_eq!(
            cfg.tier_policies[&TierInstanceId::new("encoder_blue").unwrap()][&TaskId::route()].min_confidence,
            0.80
        );
        assert_eq!(
            cfg.tier_policies[&TierInstanceId::new("encoder_green").unwrap()][&TaskId::route()].min_confidence,
            0.90
        );
    }

    #[test]
    fn model_substitution_does_not_change_instance_or_definition_identity() {
        let first = parse(BASE).unwrap();
        let substituted = BASE.replace("modernbert-base", "oryn-v2");
        let second = parse(&substituted).unwrap();
        assert_eq!(first.tiers[1].id, second.tiers[1].id);
        assert_eq!(first.tiers[1].tier_id, second.tiers[1].tier_id);
        assert_ne!(first.tiers[1].model_id, second.tiers[1].model_id);
    }

    #[test]
    fn joint_tiers_are_resolved_by_instance_name() {
        let config = BASE.replace(
            "[task.route]",
            "[pipeline]\njoint_tiers = [\"encoder_green\"]\n\n[task.route]"
        );
        let cfg = parse(&config).unwrap();
        assert!(!cfg.tiers[1].joint);
        assert!(cfg.tiers[2].joint);
    }

    #[test]
    fn rejects_duplicate_instances_unknown_definitions_and_invalid_policies() {
        let duplicate = BASE.replace(
            "[tier_policy.fasttext.route]",
            "[[tier]]\nname = \"encoder_blue\"\ntier = \"grpc\"\nmodel_id = \"other\"\n\n[tier_policy.fasttext.route]"
        );
        assert!(parse(&duplicate).is_err());

        let unknown = BASE.replace("tier = \"fasttext\"", "tier = \"transformer\"");
        assert!(parse(&unknown).is_err());

        let invalid_policy = BASE.replace("min_confidence = 0.80", "min_confidence = 1.5");
        assert!(parse(&invalid_policy).is_err());

        let old_ordinal = BASE.replace(
            "[tier_policy.encoder_blue.route]",
            "[tier_policy.tier3.route]"
        );
        assert!(parse(&old_ordinal).is_err());
    }

    #[test]
    fn empty_config_does_not_invent_task_profiles_or_policies() {
        let cfg = parse("").unwrap();
        assert!(!cfg.settings.tag_via_cascade);
        assert!(!cfg.settings.tag_in_request);
        assert!(cfg.profile(&TaskId::tag()).is_none());
        assert!(cfg.profile(&TaskId::route()).is_none());
        assert!(!cfg.has_policy(&TaskId::route()));
        assert_eq!(cfg.tiers[2].tier_id.as_str(), "grpc");
    }
}
