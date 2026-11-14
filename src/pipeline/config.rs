// src/pipeline/config.rs
//
// Loads task definitions, the tier registry, and tier/task resolution policies
// as separate concerns.
//
//   [pipeline].cascade       the one shared ordered execution cascade
//   [task.<id>]               what a task is
//   [[tier]]                  the available named tiers; no execution order
//   [tier_policy.<tier>.<id>] whether a task runs/resolves at a tier
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

use super::cascade::Cascade;
use super::registry::TierRegistry;
use super::tier::TierName;
use super::tier::{ModelFormat, Params, RequestShape, Role, Tier, Transport};

type TierConfig = (TierName, Tier, Params);
use super::task::{FallbackPolicy, OutputKind, TaskLabelMetadata, Task, TierPolicy};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSettings {
    pub tag_via_cascade: bool,
    /// Include sampled tagging as a best-effort task in the route request.
    pub tag_in_request: bool,
    pub tag_budget_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPipelineSettings {
    #[serde(default)]
    tag_via_cascade: bool,
    #[serde(default)]
    tag_in_request: bool,
    #[serde(default = "default_tag_budget_ms")]
    tag_budget_ms: u64,
}

impl Default for RawPipelineSettings {
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

/// Available tier declarations. Their vector order has no scheduling meaning.
fn default_tier_configs() -> Vec<TierConfig> {
    [
        ("rules", Role::Rules, Transport::InProcess, None, None),
        ("fasttext", Role::FastText, Transport::InProcess, None, None),
        ("modernbert", Role::Encoder, Transport::Grpc, Some(RequestShape::Score), Some(ModelFormat::Onnx)),
        ("harrier", Role::Encoder, Transport::Grpc, Some(RequestShape::Score), Some(ModelFormat::Gguf)),
    ].into_iter().map(|(name, role, transport, request_shape, model_format)| (
        TierName::new(name).expect("built-in tier names are valid"),
        Tier { exex: role, transport, request_shape },
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
        let definition = Tier { exex: tier.executor, transport: tier.transport, request_shape: tier.request_shape };
        definition.validate().map_err(|reason| format!("[[tier]] {label:?}: {reason}"))?;
        if definition.exex == Role::Encoder && tier.model_format.is_none() {
            return Err(format!("[[tier]] {label:?}: encoder must declare model_format (onnx or gguf)").into());
        }
        if !definition.accepts(tier.model_format) {
            return Err(format!("[[tier]] {label:?}: model_format {:?} is incompatible with {} executor", tier.model_format.map(ModelFormat::as_str), definition.exex.as_str()).into());
        }
        if definition.exex != Role::Encoder && (tier.endpoint.is_some() || tier.timeout_ms.is_some()) {
            return Err(format!("[[tier]] {label:?}: endpoint/timeout_ms require grpc transport").into());
        }
        match definition.exex {
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
        Self { tag_via_cascade: false, tag_in_request: false, tag_budget_ms: default_tag_budget_ms() }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProfile {
    labels: Option<Vec<String>>,
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
    pipeline: RawPipelineSettings,
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
    profiles: BTreeMap<String, Task>,
    /// Configuration policies are authored by tier name and resolve to TierId
    /// after tier instances have been constructed.
    pub tier_policies: BTreeMap<TierName, BTreeMap<String, TierPolicy>>,
    /// Compatibility numbering for numbered telemetry and `[router.tierN]`.
    pub legacy: LegacyTierNumbers,
}

impl PipelineConfig {
    /// Resolve task paths against enabled tier definitions. The ordered Cascade
    /// remains TierNames; CascadeGraph assigns instance IDs during compilation.
    pub fn tasks(&self, registry: &TierRegistry) -> Result<Vec<Task>, Box<dyn Error>> {
        self.profiles.iter().map(|(name, definition)| {
            let enabled: Vec<TierName> = definition.cascade.tiers().iter()
                .filter(|tier| registry.get(tier).is_some())
                .cloned()
                .collect();
            let mut task = definition.clone();
            task.cascade = Cascade::from_names(&enabled)
                .map_err(|error| format!("task.{name}.cascade: {error}"))?;
            Ok(task)
        }).collect()
    }

    pub fn has_policy(&self, task: &str) -> bool {
        self.tier_policies.values().any(|by_task| by_task.contains_key(task))
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

// Each task's ordered Cascade is authored by TierName and compiled to
// TierIds once concrete TierInstances have been built.
fn parse_with_base(contents: &str, base: &Path) -> Result<PipelineConfig, Box<dyn Error>> {
    let raw: RawFile = toml::from_str(contents)?;
    let has_explicit_tier_registry = !raw.tier.is_empty();
    let tiers = tier_configs_from(raw.tier)?;
    let legacy = legacy_numbers_from(raw.legacy_tiers, &tiers, has_explicit_tier_registry)?;
    let settings = PipelineSettings {
        tag_via_cascade: raw.pipeline.tag_via_cascade,
        tag_in_request: raw.pipeline.tag_in_request,
        tag_budget_ms: raw.pipeline.tag_budget_ms,
    };

    let mut profiles = BTreeMap::new();
    for (name, profile) in raw.task {
        let authored_names = match profile.cascade {
            Some(names) => names,
            None if has_explicit_tier_registry =>
                return Err(format!("task.{name}.cascade is required when [[tier]] entries are configured").into()),
            None => default_tier_configs().into_iter().map(|tier| tier.0.to_string()).collect(),
        };
        if authored_names.is_empty() {
            return Err(format!("task.{name}.cascade must contain at least one tier").into());
        }

        let mut cascade = Vec::with_capacity(authored_names.len());
        let mut seen = std::collections::BTreeSet::new();
        for raw_name in authored_names {
            let tier_name = TierName::new(raw_name.clone())
                .map_err(|error| format!("task.{name}.cascade: {error}"))?;
            if !seen.insert(tier_name.clone()) {
                return Err(format!("task.{name}.cascade lists tier {tier_name} more than once").into());
            }
            if !tiers.iter().any(|tier| tier.0 == tier_name) {
                return Err(format!(
                    "task.{name}.cascade references unavailable tier {tier_name}; registered tiers: {}",
                    known_tier_names(&tiers),
                ).into());
            }
            cascade.push(tier_name);
        }
        let cascade = Cascade::from_names(&cascade)?;

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
        profiles.insert(name.clone(), Task {
            name,
            cascade,
            output_kind: profile.output_kind,
            labels,
            label_metadata,
            taxonomy_version,
            fallback,
        });
    }

    let mut tier_policies: BTreeMap<TierName, BTreeMap<String, TierPolicy>> = BTreeMap::new();
    for (tier_key, task_policies) in raw.tier_policy {
        let declaration = find_tier(&tiers, &legacy, &tier_key).ok_or_else(|| format!(
            "tier_policy: unknown tier {tier_key:?}; registered tiers: {}", known_tier_names(&tiers)
        ))?;
        let tier = declaration.0.clone();
        let by_task = tier_policies.entry(tier.clone()).or_default();
        for (task_name, policy) in task_policies {
            let profile = profiles.get(&task_name).ok_or_else(|| format!(
                "tier_policy.{tier}.{task_name}: no matching [task.{task_name}] definition"
            ))?;
            if !profile.cascade.tiers().contains(&tier) {
                return Err(format!(
                    "tier_policy.{tier}.{task_name}: tier is not in task.{task_name}.cascade"
                ).into());
            }
            validate_policy(&task_name, &tier, &policy)?;
            if let Some(labels) = &profile.labels {
                for label in policy.label_min_confidence.keys().chain(policy.never_resolve.iter()) {
                    if !labels.contains(label) {
                        return Err(format!("tier_policy.{tier}.{task_name}: policy references unknown label {label:?}").into());
                    }
                }
            }
            if by_task.insert(task_name.clone(), policy).is_some() {
                return Err(format!(
                    "tier_policy.{tier}.{task_name}: defined twice (a tier name and its tierN alias both appear)"
                ).into());
            }
        }
    }

    Ok(PipelineConfig { settings, tiers, profiles, tier_policies, legacy })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::cascade::CascadeGraph;
    use crate::pipeline::tier::{FakeEvaluate, TierContext, TierOutput};
    use async_trait::async_trait;
    use std::sync::Arc;

    struct Silent;
    #[async_trait]
    impl FakeEvaluate for Silent {
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            TierOutput::abstain_all("silent", ctx.tasks, "test")
        }
    }
    fn registry(cfg: &PipelineConfig) -> TierRegistry {
        let entries = cfg.tiers.iter().map(|(name, tier, params)| (
            name.clone(),
            *tier,
            params.clone(),
            Arc::new(Silent) as Arc<dyn FakeEvaluate>,
        )).collect();
        TierRegistry::of(entries)
    }
    fn named_config() -> &'static str {
        r#"
            [task.route]
            cascade = ["rules", "modernbert", "harrier"]
            labels = ["jev_capable", "needs_llm"]
            fallback_label = "needs_llm"
            [task.tag]
            cascade = ["modernbert", "harrier"]
            labels = ["engineering", "help_desk"]
            fallback = "defer"

            [[tier]]
            name = "rules"
            executor = "rules"
            transport = "in_process"
            [[tier]]
            name = "modernbert"
            executor = "encoder"
            model_format = "onnx"
            transport = "grpc"
            request_shape = "score_tasks"
            endpoint = "http://127.0.0.1:50052"
            [[tier]]
            name = "harrier"
            executor = "encoder"
            model_format = "gguf"
            transport = "grpc"
            request_shape = "score_tasks"
            endpoint = "http://127.0.0.1:50051"

            [tier_policy.rules.route]
            min_confidence = 0.78
            [tier_policy.modernbert.route]
            min_confidence = 0.95
            [tier_policy.modernbert.tag]
            min_confidence = 0.85
        "#
    }

    #[test]
    fn each_task_owns_its_ordered_cascade_and_graph_compiles_the_union() {
        let cfg = parse(named_config()).unwrap();
        let registry = registry(&cfg);
        let tasks = cfg.tasks(&registry).unwrap();
        let graph = CascadeGraph::from_tasks(&tasks, &registry).unwrap();
        let route = tasks.iter().find(|task| task.name == "route").unwrap();
        let tag = tasks.iter().find(|task| task.name == "tag").unwrap();
        let route_names: Vec<_> = route.cascade.tiers().iter().map(TierName::as_str).collect();
        let tag_names: Vec<_> = tag.cascade.tiers().iter().map(TierName::as_str).collect();
        let graph_names: Vec<_> = graph.order().iter()
            .map(|id| graph.tier_name(id).unwrap().as_str()).collect();
        assert_eq!(route_names, ["rules", "modernbert", "harrier"]);
        assert_eq!(tag_names, ["modernbert", "harrier"]);
        assert_eq!(graph_names, ["rules", "modernbert", "harrier"]);
        let modernbert = graph.id_for_name(&TierName::new("modernbert").unwrap()).unwrap();
        assert!(graph.task_uses("route", &modernbert));
        assert!(graph.task_uses("tag", &modernbert));
    }

    #[test]
    fn explicit_tier_registry_requires_each_task_cascade() {
        let result = parse(r#"
            [task.route]
            fallback_label = "needs_llm"
            [[tier]]
            name = "rules"
            executor = "rules"
            transport = "in_process"
        "#);
        assert!(result.is_err_and(|e| e.to_string().contains("task.route.cascade is required")));
    }

    #[test]
    fn legacy_config_defaults_each_task_to_the_original_four_tiers() {
        let cfg = parse("[task.route]\nfallback_label = \"needs_llm\"").unwrap();
        let names: Vec<_> = cfg.profiles["route"].cascade.tiers().iter().map(TierName::as_str).collect();
        assert_eq!(names, ["rules", "fasttext", "modernbert", "harrier"]);
    }

    #[test]
    fn task_cascade_rejects_empty_unknown_and_duplicate_tier_names() {
        let tier = "[[tier]]\nname = \"rules\"\nexecutor = \"rules\"\ntransport = \"in_process\"\n";
        for (path, expected) in [
            ("cascade = []", "at least one tier"),
            ("cascade = [\"missing\"]", "unavailable tier missing"),
            ("cascade = [\"rules\", \"rules\"]", "more than once"),
        ] {
            let cfg = format!("[task.route]\n{path}\nfallback_label = \"needs_llm\"\n{tier}");
            assert!(parse(&cfg).is_err_and(|e| e.to_string().contains(expected)), "{path}");
        }
    }

    #[test]
    fn tier_policy_must_be_in_that_tasks_cascade() {
        let result = parse(r#"
            [task.route]
            cascade = ["rules"]
            labels = ["jev_capable", "needs_llm"]
            fallback_label = "needs_llm"
            [[tier]]
            name = "rules"
            executor = "rules"
            transport = "in_process"
            [[tier]]
            name = "fasttext"
            executor = "fasttext"
            transport = "in_process"
            [tier_policy.fasttext.route]
            min_confidence = 0.9
        "#);
        assert!(result.is_err_and(|e| e.to_string().contains(
            "tier_policy.fasttext.route: tier is not in task.route.cascade"
        )));
    }

    #[test]
    fn disabled_instances_are_omitted_from_task_paths_without_reordering() {
        let cfg = parse(named_config()).unwrap();
        let entries = cfg.tiers.iter()
            .filter(|(name, _, _)| name.as_str() == "harrier" || name.as_str() == "rules")
            .map(|(name, tier, params)| (
                name.clone(), *tier, params.clone(), Arc::new(Silent) as Arc<dyn FakeEvaluate>,
            ))
            .collect();
        let registry = TierRegistry::of(entries);
        let tasks = cfg.tasks(&registry).unwrap();
        let route = tasks.iter().find(|task| task.name == "route").unwrap();
        let names: Vec<_> = route.cascade.tiers().iter().map(TierName::as_str).collect();
        assert_eq!(names, ["rules", "harrier"]);
    }

    #[test]
    fn preserves_taxonomy_metadata() {
        let cfg = parse_with_base(
            "[task.tag]\ntaxonomy = \"../taskonomies/help_desk.json\"\noutput_kind = \"multiclass\"\nfallback = \"defer\"",
            Path::new("config"),
        ).unwrap();
        let tag = cfg.profiles.get("tag").unwrap();
        assert_eq!(tag.output_kind, OutputKind::Multiclass);
        assert_eq!(tag.labels.as_ref().unwrap().len(), 20);
        assert_eq!(tag.taxonomy_version.as_deref(), Some("1.1.0"));
        assert!(tag.label_metadata.iter().any(|row| row.id == "app_access_req"));
    }

    #[test]
    fn shipped_config_has_valid_task_paths_and_policies() {
        let cfg = load_or_default("config/tasks.toml").unwrap();
        let available: std::collections::BTreeSet<_> = cfg.tiers.iter().map(|tier| tier.0.clone()).collect();
        assert!(cfg.profiles.values().flat_map(|task| task.cascade.tiers()).all(|tier| available.contains(tier)));
        assert!(cfg.has_policy("route"));
        assert!(!cfg.profiles.is_empty());
    }
}
