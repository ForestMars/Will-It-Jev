// src/pipeline/config.rs
//
// Loads task definitions, the tier registry, and tier/task resolution policies
// as separate concerns.
//
//   [task.<id>]               what a task is
//   [[tier]]                  the available named tiers; no execution order
//   [tier_policy.<tier>.<id>] when a tier may resolve a task
//   [legacy_tiers]            compatibility: tier name -> its old number (optional)
//
// `<tier>` is a tier name. For configs written before tiers had names, the key
// `tierN` is accepted as an alias for the tier numbered N in `[legacy_tiers]`.
// The numbers are read here, at the boundary, and handed on as a
// `LegacyTierNumbers`; no tier, instance or cascade type carries them.

use std::collections::BTreeMap;
use std::{error::Error, fs, path::Path};

use serde::Deserialize;

use crate::legacy_tiers::LegacyTierNumbers;

use super::tier::TierName;
use super::tier::{ModelFormat, Params, RequestShape, Role, Tier, Transport};

type TierConfig = (TierName, Tier, Params);
use super::task::{Cascade, FallbackPolicy, OutputKind, TaskLabelMetadata, Task, TierPolicy};

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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTier {
    name: String,
    executor: Role,
    transport: Transport,
    request_shape: Option<RequestShape>,
    model_format: Option<ModelFormat>,
    model: Option<String>,
    config: Option<String>,
    enabled: Option<bool>,
    endpoint: Option<String>,
    timeout_ms: Option<u64>,
}

/// The available tier instances. Their vector order has no scheduling meaning.
fn default_tier_configs() -> Vec<TierConfig> {
    [
        ("rules", Role::Rules, Transport::InProcess, None, None),
        ("fasttext", Role::FastText, Transport::InProcess, None, None),
        ("modernbert", Role::Encoder, Transport::Grpc, Some(RequestShape::Score), Some(ModelFormat::Onnx)),
        ("harrier", Role::Encoder, Transport::Grpc, Some(RequestShape::Score), Some(ModelFormat::Gguf)),
    ].into_iter().map(|(name, role, transport, request_shape, model_format)| (
        TierName::new(name).expect("built-in tier names are valid"),
        Tier { role, transport, request_shape },
        Params { model_format, model: None, config: None, endpoint: None, timeout_ms: None, enabled: None },
    )).collect()

}

fn tier_configs_from(raw: Vec<RawTier>) -> Result<Vec<TierConfig>, Box<dyn Error>> {
    if raw.is_empty() { return Ok(default_tier_configs()); }
    let mut tiers: Vec<TierConfig> = Vec::with_capacity(raw.len());
    for tier in raw {
        let name = TierName::new(tier.name.clone()).map_err(|error| format!("[[tier]]: {error}"))?;
        let label = name.as_str().to_string();
        if tiers.iter().any(|existing| existing.0 == name) {
            return Err(format!("[[tier]] {label:?} is defined twice").into());
        }
        let definition = Tier { role: tier.executor, transport: tier.transport, request_shape: tier.request_shape };
        definition.validate().map_err(|reason| format!("[[tier]] {label:?}: {reason}"))?;
        if definition.role == Role::Encoder && tier.model_format.is_none() {
            return Err(format!("[[tier]] {label:?}: encoder must declare model_format (onnx or gguf)").into());
        }
        if !definition.accepts(tier.model_format) {
            return Err(format!("[[tier]] {label:?}: model_format {:?} is incompatible with {} executor", tier.model_format.map(ModelFormat::as_str), definition.role.as_str()).into());
        }
        if definition.role != Role::Encoder && (tier.endpoint.is_some() || tier.timeout_ms.is_some()) {
            return Err(format!("[[tier]] {label:?}: endpoint/timeout_ms require grpc transport").into());
        }
        match definition.role {
            Role::Rules if tier.model.is_some() => {
                return Err(format!("[[tier]] {label:?}: a rules tier takes `config`, not `model`").into());
            }
            Role::FastText | Role::Encoder if tier.config.is_some() => {
                return Err(format!("[[tier]] {label:?}: `config` is only valid on a rules tier").into());
            }
            _ => {}
        }
        for (field, value) in [("endpoint", &tier.endpoint), ("model", &tier.model), ("config", &tier.config)] {
            if value.as_deref().is_some_and(|value| value.trim().is_empty()) {
                return Err(format!("[[tier]] {label:?}: {field} must be non-empty").into());
            }
        }
        if tier.timeout_ms == Some(0) {
            return Err(format!("[[tier]] {label:?}: timeout_ms must be greater than zero").into());
        }
        tiers.push((
            name,
            definition,
            Params {
                model_format: tier.model_format, model: tier.model, config: tier.config, endpoint: tier.endpoint,
                timeout_ms: tier.timeout_ms, enabled: tier.enabled,
            },
        ));
    }
    Ok(tiers)
}

fn find_tier<'a>(tiers: &'a [TierConfig], legacy: &LegacyTierNumbers, reference: &str) -> Option<&'a TierConfig> {
    if let Some(tier) = tiers.iter().find(|tier| tier.0.as_str() == reference) {
        return Some(tier);
    }
    reference
        .strip_prefix("tier")
        .and_then(|number| number.parse::<u8>().ok())
        .and_then(|ordinal| legacy.name_of(ordinal))
        .and_then(|name| tiers.iter().find(|tier| &tier.0 == name))
}

/// The compatibility numbering: `[legacy_tiers]` if present; otherwise the
/// original four-tier numbering when no `[[tier]]` is declared; otherwise none.
fn legacy_numbers_from(
    declared: Option<BTreeMap<String, u8>>,
    tiers: &[TierConfig],
    has_explicit_tier_registry: bool,
) -> Result<LegacyTierNumbers, Box<dyn Error>> {
    let Some(declared) = declared else {
        return Ok(if has_explicit_tier_registry {
            LegacyTierNumbers::default()
        } else {
            LegacyTierNumbers::original_four()
        });
    };
    let mut pairs = Vec::with_capacity(declared.len());
    for (name, ordinal) in declared {
        let name = TierName::new(name.clone()).map_err(|error| format!("[legacy_tiers]: {error}"))?;
        if !tiers.iter().any(|tier| tier.0 == name) {
            return Err(format!(
                "[legacy_tiers]: unknown tier {name}; registered tiers: {}", known_tier_names(tiers)
            ).into());
        }
        pairs.push((name, ordinal));
    }
    Ok(LegacyTierNumbers::new(pairs).map_err(|error| format!("[legacy_tiers]: {error}"))?)
}

fn known_tier_names(tiers: &[TierConfig]) -> String {
    tiers.iter().map(|tier| tier.0.as_str()).collect::<Vec<_>>().join(", ")
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
    /// Ordered path of tier names for this task.
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
    legacy_tiers: Option<BTreeMap<String, u8>>,
    #[serde(default)]
    tier_policy: BTreeMap<String, BTreeMap<String, TierPolicy>>,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub settings: PipelineSettings,
    /// Available configured tiers; vector order carries no scheduling semantics.
    pub(crate) tiers: Vec<TierConfig>,
    profiles: BTreeMap<String, Task<TierName>>,
    /// Configuration policies are authored by tier name and resolve to TierId
    /// after tier instances have been constructed.
    pub tier_policies: BTreeMap<TierName, BTreeMap<String, TierPolicy>>,
    /// Compatibility numbering for numbered telemetry and `[router.tierN]`.
    pub legacy: LegacyTierNumbers,
}

impl PipelineConfig {
    pub fn profile(&self, task: &str) -> Option<Task<TierName>> {
        self.profiles.get(task).cloned()
    }

    /// All configured task contracts, returned in stable task-ID order.
    pub fn profiles(&self) -> Vec<Task<TierName>> {
        self.profiles.values().cloned().collect()
    }

    pub fn has_policy(&self, task: &str) -> bool {
        self.tier_policies
            .values()
            .any(|by_task| by_task.contains_key(task))
    }
}

fn validate_policy(task: &str, tier: &TierName, policy: &TierPolicy) -> Result<(), Box<dyn Error>> {
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
    let has_explicit_tier_registry = !raw.tier.is_empty();
    let tiers = tier_configs_from(raw.tier)?;
    let legacy = legacy_numbers_from(raw.legacy_tiers, &tiers, has_explicit_tier_registry)?;
    let mut profiles = BTreeMap::new();

    for (name, profile) in raw.task {
        let task = name.clone();
        let path = match profile.cascade {
            Some(names) => names.into_iter().map(|tier_name| {
                TierName::new(tier_name.clone()).map_err(|error| format!("task.{name}.cascade: {error}"))
            }).collect::<Result<Vec<_>, _>>()?,
            None if has_explicit_tier_registry => {
                return Err(format!("task.{name}: cascade is required when [[tier]] entries are configured").into());
            }
            None => default_tier_configs().into_iter().map(|tier| tier.0).collect(),
        };

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
        profiles.insert(task.clone(), Task {
            name: task, cascade: Cascade { tiers: path }, output_kind: profile.output_kind, labels, label_metadata,
            taxonomy_version, fallback,
        });
    }

    let mut tier_policies: BTreeMap<TierName, BTreeMap<String, TierPolicy>> = BTreeMap::new();
    for (tier_key, task_policies) in raw.tier_policy {
        let decl = find_tier(&tiers, &legacy, &tier_key).ok_or_else(|| format!(
            "tier_policy: unknown tier {tier_key:?}; registered tiers: {}", known_tier_names(&tiers)
        ))?;
        let tier = decl.0.clone();
        let by_task = tier_policies.entry(tier.clone()).or_default();
        for (task_name, policy) in task_policies {
            let task = task_name.clone();
            let profile = profiles.get(&task).ok_or_else(|| format!(
                "tier_policy.{tier}.{task_name}: no matching [task.{task_name}] definition"
            ))?;
            if !profile.cascade.tiers.contains(&tier) {
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
    Ok(PipelineConfig { settings: raw.pipeline, tiers, profiles, tier_policies, legacy })
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
            model_format = "onnx"
            transport = "grpc"
            request_shape = "score_tasks"
            endpoint = "http://127.0.0.1:50052"
            [[tier]]
            name = "harrier"
            executor = "scorer"
            model_format = "onnx"
            transport = "grpc"
            request_shape = "score_tasks"
            endpoint = "http://127.0.0.1:50051"
        "#).unwrap();
        assert_eq!(cfg.profile("route").unwrap().cascade.tiers.iter().map(TierName::as_str).collect::<Vec<_>>(), ["rules", "modernbert", "harrier"]);
        assert_eq!(cfg.profile("tag").unwrap().cascade.tiers.iter().map(TierName::as_str).collect::<Vec<_>>(), ["modernbert", "harrier"]);
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
            model_format = "onnx"
            transport = "grpc"
            request_shape = "score"
            endpoint = "http://127.0.0.1:50051"
            [[tier]]
            name = "b"
            executor = "scorer"
            model_format = "onnx"
            transport = "grpc"
            request_shape = "score"
            endpoint = "http://127.0.0.1:50052"
        "#);
        assert!(matches!(result, Err(error) if error.to_string().contains("cycle")));
    }

    #[test]
    fn explicit_tier_registry_requires_each_task_to_declare_its_cascade() {
        let result = parse(r#"
            [task.route]
            fallback_label = "needs_llm"
            [[tier]]
            name = "zeta"
            executor = "scorer"
            model_format = "onnx"
            transport = "grpc"
            request_shape = "score"
            endpoint = "http://127.0.0.1:50051"
        "#);
        assert!(result.is_err_and(|error| error.to_string().contains("cascade is required")));
    }

    #[test]
    fn legacy_config_without_a_named_registry_keeps_the_original_four_tier_path() {
        let cfg = parse("[task.route]\nfallback_label = \"needs_llm\"").unwrap();
        assert_eq!(
            cfg.profile("route").unwrap().cascade.tiers.iter().map(TierName::as_str).collect::<Vec<_>>(),
            ["rules", "fasttext", "modernbert", "harrier"]
        );
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
        let tag = cfg.profile("tag").unwrap();
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
        let available: std::collections::BTreeSet<_> = cfg.tiers.iter().map(|tier| tier.0.clone()).collect();
        for task in cfg.profiles() {
            assert!(task.cascade.tiers.iter().all(|tier| available.contains(tier)));
        }
        assert!(cfg.has_policy(&String::from("route")));
        assert!(!cfg.profiles().is_empty());
    }

    fn with_tiers(tiers: &str) -> String {
        format!(
            r#"
            [task.route]
            labels = ["jev_capable", "needs_llm"]
            fallback_label = "needs_llm"
            cascade = ["a", "b"]
            {tiers}
            "#
        )
    }

    #[test]
    fn two_instances_of_one_tier_declare_their_own_model_format_and_model() {
        let cfg = parse(&with_tiers(r#"
            [[tier]]
            name = "a"
            executor = "encoder"
            model_format = "onnx"
            transport = "grpc"
            request_shape = "score"
            model = "modernbert-base"
            [[tier]]
            name = "b"
            role = "encoder"
            model_format = "gguf"
            transport = "grpc"
            request_shape = "score"
            model = "harrier-270m"
        "#)).unwrap();
        assert_eq!(cfg.tiers[0].tier, cfg.tiers[1].tier);
        assert_eq!(cfg.tiers[0].params.model_format, Some(ModelFormat::Onnx));
        assert_eq!(cfg.tiers[1].params.model_format, Some(ModelFormat::Gguf));
        assert_eq!(cfg.tiers[1].params.model.as_deref(), Some("harrier-270m"));
    }

    #[test]
    fn several_instances_of_the_in_process_roles_are_allowed() {
        let cfg = parse(&with_tiers(r#"
            [[tier]]
            name = "a"
            executor = "fasttext"
            transport = "in_process"
            model = "models/small.bin"
            [[tier]]
            name = "b"
            role = "fasttext"
            transport = "in_process"
            model = "models/large.bin"
        "#)).unwrap();
        assert_eq!(cfg.tiers.len(), 2);
        assert_eq!(cfg.tiers[0].params.model_format, None);
    }

    #[test]
    fn the_older_executor_vocabulary_is_still_read() {
        let cfg = parse(&with_tiers(r#"
            [[tier]]
            name = "a"
            executor = "rules"
            transport = "in_process"
            [[tier]]
            name = "b"
            executor = "scorer"
            model_format = "onnx"
            transport = "grpc"
            request_shape = "score"
        "#)).unwrap();
        assert_eq!(cfg.tiers[0].tier.role, Role::Rules);
        assert_eq!(cfg.tiers[1].tier.role, Role::Encoder);
    }

    #[test]
    fn an_encoder_must_declare_a_model_format_and_other_roles_reject_one() {
        let missing = with_tiers(r#"
            [[tier]]
            name = "a"
            role = "encoder"
            transport = "grpc"
            request_shape = "score"
            [[tier]]
            name = "b"
            executor = "rules"
            transport = "in_process"
        "#);
        assert!(parse(&missing).is_err_and(|e| e.to_string().contains("must declare model_format")));
        let misfit = with_tiers(r#"
            [[tier]]
            name = "a"
            role = "rules"
            model_format = "onnx"
            transport = "in_process"
            [[tier]]
            name = "b"
            role = "rules"
            transport = "in_process"
        "#);
        assert!(parse(&misfit).is_err_and(|e| e.to_string().contains("model_format")));
    }

    #[test]
    fn instance_parameters_are_checked_against_the_role() {
        let rules_model = with_tiers(r#"
            [[tier]]
            name = "a"
            role = "rules"
            transport = "in_process"
            model = "x.bin"
            [[tier]]
            name = "b"
            role = "rules"
            transport = "in_process"
        "#);
        assert!(parse(&rules_model).is_err_and(|e| e.to_string().contains("takes `config`")));
        let fasttext_config = with_tiers(r#"
            [[tier]]
            name = "a"
            role = "fasttext"
            transport = "in_process"
            config = "x.toml"
            [[tier]]
            name = "b"
            role = "rules"
            transport = "in_process"
        "#);
        assert!(parse(&fasttext_config).is_err_and(|e| e.to_string().contains("only valid on a rules tier")));
        let empty_model = with_tiers(r#"
            [[tier]]
            name = "a"
            role = "fasttext"
            transport = "in_process"
            model = "  "
            [[tier]]
            name = "b"
            role = "rules"
            transport = "in_process"
        "#);
        assert!(parse(&empty_model).is_err_and(|e| e.to_string().contains("model must be non-empty")));
    }

    const TWO_TIERS: &str = r#"
        [task.route]
        labels = ["jev_capable", "needs_llm"]
        fallback_label = "needs_llm"
        cascade = ["a", "b"]
        [[tier]]
        name = "a"
        role = "rules"
        transport = "in_process"
        [[tier]]
        name = "b"
        role = "fasttext"
        transport = "in_process"
        [tier_policy.a.route]
        min_confidence = 0.9
    "#;

    #[test]
    fn legacy_numbers_live_in_their_own_table_and_not_on_the_tier() {
        let cfg = parse(&format!("{TWO_TIERS}\n[legacy_tiers]\na = 1\nb = 2\n")).unwrap();
        let name = |value: &str| TierName::new(value).unwrap();
        assert_eq!(cfg.legacy.ordinal_of(&name("a")), Some(1));
        assert_eq!(cfg.legacy.name_of(2), Some(&name("b")));
        // The old per-tier key no longer exists.
        let on_the_tier = TWO_TIERS.replace("role = \"rules\"", "role = \"rules\"\n        legacy_ordinal = 1");
        assert!(parse(&on_the_tier).is_err());
    }

    #[test]
    fn legacy_tiers_must_name_declared_tiers_with_unique_positive_numbers() {
        assert!(parse(&format!("{TWO_TIERS}\n[legacy_tiers]\nmissing = 1\n"))
            .is_err_and(|e| e.to_string().contains("unknown tier")));
        assert!(parse(&format!("{TWO_TIERS}\n[legacy_tiers]\na = 1\nb = 1\n")).is_err());
        assert!(parse(&format!("{TWO_TIERS}\n[legacy_tiers]\na = 0\n")).is_err());
    }

    #[test]
    fn a_tiern_policy_key_resolves_through_the_legacy_table() {
        let cfg = parse(&TWO_TIERS.replace("[tier_policy.a.route]", "[tier_policy.tier1.route]")
            .replace("min_confidence = 0.9\n    ", "min_confidence = 0.9\n        [legacy_tiers]\n        a = 1\n    ")).unwrap();
        assert!(cfg.tier_policies.contains_key(&TierName::new("a").unwrap()));
        // Without the table the alias has nothing to resolve through.
        assert!(parse(&TWO_TIERS.replace("[tier_policy.a.route]", "[tier_policy.tier1.route]"))
            .is_err_and(|e| e.to_string().contains("unknown tier")));
    }

    #[test]
    fn configuration_without_a_tier_registry_keeps_the_original_numbering() {
        let cfg = parse("[task.route]\nlabels = [\"jev_capable\", \"needs_llm\"]\nfallback_label = \"needs_llm\"\n").unwrap();
        assert_eq!(cfg.legacy, LegacyTierNumbers::original_four());
        let explicit = parse(TWO_TIERS).unwrap();
        assert_eq!(explicit.legacy, LegacyTierNumbers::default());
    }

    #[test]
    fn the_shipped_config_numbers_the_original_four() {
        let cfg = load_or_default("config/tasks.toml").unwrap();
        assert_eq!(cfg.legacy, LegacyTierNumbers::original_four());
    }
}
