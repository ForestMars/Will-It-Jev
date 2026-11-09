use std::collections::BTreeMap;
use std::{error::Error, fs, path::Path};

use serde::Deserialize;

use super::registry::{TierId, TierInstanceId};
pub use super::registry::TierKind;
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
    /// Names of configured instances whose sidecars support joint ScoreTasks.
    #[serde(default)]
    pub joint_tiers: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTierDefinition { name: String, kind: TierKind }

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTierInstance {
    name: String,
    tier: String,
    model_id: String,
    #[serde(default)]
    joint: bool,
    enabled: Option<bool>,
    endpoint: Option<String>,
    timeout_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TierDefinitionSpec { pub id: TierId, pub kind: TierKind }

#[derive(Debug, Clone)]
pub struct TierInstanceSpec {
    pub id: TierInstanceId,
    pub tier_id: TierId,
    pub model_id: String,
    pub joint: bool,
    pub enabled: Option<bool>,
    pub endpoint: Option<String>,
    pub timeout_ms: Option<u64>,
}

fn default_tier_definitions() -> Vec<TierDefinitionSpec> {
    [("rules", TierKind::Rules), ("fasttext", TierKind::Fasttext), ("encoder", TierKind::Grpc)]
        .into_iter()
        .map(|(name, kind)| TierDefinitionSpec { id: TierId::new(name).unwrap(), kind })
        .collect()
}

fn default_tier_instances() -> Vec<TierInstanceSpec> {
    [
        ("rules", "rules", "rules", None, None),
        ("fasttext", "fasttext", "fasttext", None, None),
        ("modernbert", "encoder", "modernbert-base", Some("http://[::1]:50052"), Some(500)),
        ("harrier", "encoder", "harrier-270m", Some("http://[::1]:50051"), Some(500)),
    ].into_iter().map(|(name, tier, model_id, endpoint, timeout_ms)| TierInstanceSpec {
        id: TierInstanceId::new(name).unwrap(),
        tier_id: TierId::new(tier).unwrap(),
        model_id: model_id.to_string(),
        joint: false,
        enabled: None,
        endpoint: endpoint.map(str::to_string),
        timeout_ms,
    }).collect()
}

fn tier_specs_from(
    raw_definitions: Vec<RawTierDefinition>,
    raw_instances: Vec<RawTierInstance>,
) -> Result<(Vec<TierDefinitionSpec>, Vec<TierInstanceSpec>), Box<dyn Error>> {
    let definitions = if raw_definitions.is_empty() {
        default_tier_definitions()
    } else {
        let mut result: Vec<TierDefinitionSpec> = Vec::with_capacity(raw_definitions.len());
        for raw in raw_definitions {
            let id = TierId::new(raw.name.clone()).map_err(|error| format!("[[tier_definition]]: {error}"))?;
            if result.iter().any(|item| item.id == id) {
                return Err(format!("[[tier_definition]] {id} is defined twice").into());
            }
            result.push(TierDefinitionSpec { id, kind: raw.kind });
        }
        result
    };

    let instances = if raw_instances.is_empty() {
        default_tier_instances()
    } else {
        let mut result: Vec<TierInstanceSpec> = Vec::with_capacity(raw_instances.len());
        for raw in raw_instances {
            let id = TierInstanceId::new(raw.name.clone())
                .map_err(|error| format!("[[tier_instance]]: {error}"))?;
            if result.iter().any(|item| item.id == id) {
                return Err(format!("[[tier_instance]] {id} is defined twice").into());
            }
            let tier_id = TierId::new(raw.tier.clone())
                .map_err(|error| format!("[[tier_instance]] {id}: {error}"))?;
            let definition = definitions.iter().find(|item| item.id == tier_id)
                .ok_or_else(|| format!("[[tier_instance]] {id}: unknown tier definition {tier_id}"))?;
            if raw.model_id.trim().is_empty() {
                return Err(format!("[[tier_instance]] {id}: model_id must be non-empty").into());
            }
            if definition.kind != TierKind::Grpc && (raw.endpoint.is_some() || raw.timeout_ms.is_some()) {
                return Err(format!("[[tier_instance]] {id}: endpoint and timeout_ms require a grpc definition").into());
            }
            if definition.kind != TierKind::Grpc && raw.joint {
                return Err(format!("[[tier_instance]] {id}: joint requires a grpc definition").into());
            }
            if raw.endpoint.as_deref().is_some_and(|endpoint| endpoint.trim().is_empty()) {
                return Err(format!("[[tier_instance]] {id}: endpoint must be non-empty").into());
            }
            if raw.timeout_ms == Some(0) {
                return Err(format!("[[tier_instance]] {id}: timeout_ms must be greater than zero").into());
            }
            result.push(TierInstanceSpec {
                id, tier_id, model_id: raw.model_id, joint: raw.joint,
                enabled: raw.enabled, endpoint: raw.endpoint, timeout_ms: raw.timeout_ms,
            });
        }
        result
    };

    for instance in &instances {
        if !definitions.iter().any(|definition| definition.id == instance.tier_id) {
            return Err(format!("[[tier_instance]] {} references missing definition {}", instance.id, instance.tier_id).into());
        }
    }
    Ok((definitions, instances))
}

fn find_tier<'a>(instances: &'a [TierInstanceSpec], reference: &str) -> Option<&'a TierInstanceSpec> {
    let id = TierInstanceId::new(reference.to_string()).ok()?;
    instances.iter().find(|instance| instance.id == id)
}

fn known_tier_names(instances: &[TierInstanceSpec]) -> String {
    instances.iter().map(|instance| instance.id.as_str()).collect::<Vec<_>>().join(", ")
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
    tier_definition: Vec<RawTierDefinition>,
    #[serde(default)]
    tier_instance: Vec<RawTierInstance>,
    #[serde(default)]
    tier_policy: BTreeMap<String, BTreeMap<String, TierPolicy>>,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub settings: PipelineSettings,
    /// Reusable execution contracts.
    pub definitions: Vec<TierDefinitionSpec>,
    /// Configured tier instances in execution order.
    pub tiers: Vec<TierInstanceSpec>,
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

fn validate_policy(task: &str, tier: &TierInstanceId, policy: &TierPolicy) -> Result<(), Box<dyn Error>> {
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
    let (definitions, mut tiers) = tier_specs_from(raw.tier_definition, raw.tier_instance)?;
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
    for (tier_key, task_policies) in raw.tier_policy {
        let spec = find_tier(&tiers, &tier_key).ok_or_else(|| {
            format!(
                "tier_policy: unknown tier {tier_key:?}; registered tiers: {}",
                known_tier_names(&tiers)
            )
        })?;
        let tier = spec.id.clone();

        let by_task = tier_policies.entry(tier.clone()).or_default();
        for (task_name, policy) in task_policies {
            let task = TaskId::new(task_name.clone());
            let profile = profiles.get(&task).ok_or_else(|| {
                format!(
                    "tier_policy.{tier}.{task_name}: no matching [task.{task_name}] definition"
                )
            })?;
            validate_policy(&task_name, &tier, &policy)?;
            if let Some(labels) = &profile.labels {
                for label in policy
                    .label_min_confidence
                    .keys()
                    .chain(policy.never_resolve.iter())
                {
                    if !labels.contains(label) {
                        return Err(format!(
                            "tier_policy.{tier}.{task_name}: policy references unknown label {label:?}"
                        )
                        .into());
                    }
                }
            }
            if by_task.insert(task, policy).is_some() {
                return Err(format!(
                    "tier_policy.{tier}.{task_name}: defined twice (a tier name and its tierN alias both appear)"
                )
                .into());
            }
        }
    }

    for reference in &raw.pipeline.joint_tiers {
        let instance_id = TierInstanceId::new(reference.clone())
            .map_err(|error| format!("pipeline.joint_tiers: {error}"))?;
        let Some(index) = tiers.iter().position(|instance| instance.id == instance_id) else {
            return Err(format!(
                "pipeline.joint_tiers: unknown instance {instance_id:?}; configured instances: {}",
                known_tier_names(&tiers)
            ).into());
        };
        let instance = &mut tiers[index];
        let definition = definitions.iter().find(|definition| definition.id == instance.tier_id)
            .expect("validated instance references a definition");
        if definition.kind != TierKind::Grpc {
            return Err(format!("pipeline.joint_tiers: {instance_id} is not a grpc instance").into());
        }
        instance.joint = true;
    }

    Ok(PipelineConfig {
        settings: raw.pipeline,
        definitions,
        tiers,
        profiles,
        tier_policies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
        [task.risk]
        output_kind = "multilabel"
        labels = ["fraud", "abuse"]
        fallback = "defer"

        [[tier_definition]]
        name = "encoder"
        kind = "grpc"

        [[tier_instance]]
        name = "second"
        tier = "encoder"
        model_id = "harrier-270m"
        endpoint = "http://127.0.0.1:50052"
        joint = true

        [[tier_instance]]
        name = "first"
        tier = "encoder"
        model_id = "modernbert-base"
        endpoint = "http://127.0.0.1:50053"

        [tier_policy.second.risk]
        min_confidence = 0.8
    "#;

    #[test]
    fn definitions_and_instances_are_distinct_and_order_is_configuration_order() {
        let cfg = parse(CONFIG).unwrap();
        assert_eq!(cfg.definitions.len(), 1);
        assert_eq!(cfg.definitions[0].id.as_str(), "encoder");
        assert_eq!(cfg.tiers.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), ["second", "first"]);
        assert!(cfg.tiers.iter().all(|i| i.tier_id.as_str() == "encoder"));
        assert_eq!(cfg.tiers[0].model_id, "harrier-270m");
        assert_eq!(cfg.tiers[1].model_id, "modernbert-base");
    }

    #[test]
    fn policies_and_joint_flags_are_instance_scoped() {
        let cfg = parse(CONFIG).unwrap();
        let second = cfg.tiers.iter().find(|i| i.id.as_str() == "second").unwrap();
        let first = cfg.tiers.iter().find(|i| i.id.as_str() == "first").unwrap();
        assert!(second.joint);
        assert!(!first.joint);
        assert!(cfg.tier_policies.contains_key(&second.id));
        assert!(!cfg.tier_policies.contains_key(&first.id));
    }

    #[test]
    fn arbitrary_task_contracts_remain_independent_of_tier_definitions() {
        let cfg = parse(CONFIG).unwrap();
        let profile = cfg.profile(&TaskId::new("risk")).unwrap();
        assert_eq!(profile.output_kind, OutputKind::Multilabel);
        assert_eq!(profile.labels.as_deref(), Some(&["fraud".to_string(), "abuse".to_string()][..]));
    }

    #[test]
    fn model_substitution_keeps_instance_and_definition_ids() {
        let mut cfg = parse(CONFIG).unwrap();
        let instance = &mut cfg.tiers[0];
        let id = instance.id.clone();
        let tier_id = instance.tier_id.clone();
        instance.model_id = "oryn".into();
        assert_eq!(instance.id, id);
        assert_eq!(instance.tier_id, tier_id);
        assert_eq!(instance.model_id, "oryn");
    }

    #[test]
    fn invalid_definition_and_instance_configuration_is_rejected() {
        let cases = [
            r#"[[tier_definition]]
name = "encoder"
kind = "grpc"
[[tier_definition]]
name = "encoder"
kind = "grpc""#,
            r#"[[tier_definition]]
name = "encoder"
kind = "grpc"
[[tier_instance]]
name = "a"
tier = "missing"
model_id = "m""#,
            r#"[[tier_definition]]
name = "encoder"
kind = "grpc"
[[tier_instance]]
name = "a"
tier = "encoder"
model_id = """#,
            r#"[[tier_definition]]
name = "rules"
kind = "rules"
[[tier_instance]]
name = "a"
tier = "rules"
model_id = "m"
endpoint = "http://localhost""#,
            r#"[[tier_definition]]
name = "encoder"
kind = "grpc"
[[tier_instance]]
name = "a"
tier = "encoder"
model_id = "m"
timeout_ms = 0"#,
        ];
        for raw in cases {
            assert!(parse(raw).is_err(), "should reject {raw}");
        }
    }

    #[test]
    fn unknown_instance_policy_and_non_grpc_joint_flag_are_rejected() {
        assert!(parse(r#"
            [task.risk]
            fallback = "defer"
            [tier_policy.missing.risk]
            min_confidence = 0.8
        "#).is_err());
        assert!(parse(r#"
            [[tier_definition]]
            name = "rules"
            kind = "rules"
            [[tier_instance]]
            name = "rules_one"
            tier = "rules"
            model_id = "rules-v1"
            [pipeline]
            joint_tiers = ["rules_one"]
        "#).is_err());
    }

    #[test]
    fn defaults_use_shared_grpc_definition_and_named_instances() {
        let cfg = parse("").unwrap();
        assert_eq!(cfg.definitions.len(), 3);
        assert_eq!(
            cfg.tiers.iter().map(|i| (i.id.as_str(), i.tier_id.as_str())).collect::<Vec<_>>(),
            [("rules", "rules"), ("fasttext", "fasttext"), ("modernbert", "grpc"), ("harrier", "grpc")]
        );
    }
}
