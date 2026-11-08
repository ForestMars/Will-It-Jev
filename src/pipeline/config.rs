// src/pipeline/config.rs
//
// Loads task definitions, the tier registry, and tier/task resolution policies
// as separate concerns.
//
//   [task.<id>]               what a task is
//   [[tier]]                  the named execution tiers; array order is run order
//   [tier_policy.<tier>.<id>] when a tier may resolve a task
//
// `<tier>` is a tier name. For configs written before tiers had names, the key
// `tierN` is accepted as an alias for the tier whose `legacy_ordinal` is N.

use std::collections::BTreeMap;
use std::{error::Error, fs, path::Path};

use serde::Deserialize;

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
    /// gRPC tiers whose sidecars implement the joint ScoreTasks RPC, by tier
    /// name (or, for older configs, by legacy ordinal). A tier can also set
    /// `joint = true` on its own `[[tier]]` entry.
    #[serde(default)]
    pub joint_tiers: Vec<TierRef>,
}

/// A reference to a tier from older and newer configs alike.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum TierRef {
    Ordinal(u8),
    Name(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierKind {
    /// Pattern rules (Aho-Corasick plus fuzzy matching). At most one.
    Rules,
    /// fastText classifier. At most one.
    Fasttext,
    /// Any gRPC sidecar speaking the scorer protocol. As many as needed.
    Grpc,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTier {
    name: String,
    kind: TierKind,
    #[serde(default)]
    joint: bool,
    /// The 1..=4 number this tier carried before tiers had names. Only needed
    /// to keep `tierN` config keys and the numbered telemetry columns working.
    legacy_ordinal: Option<u8>,
    /// Defaults to `router.tier<legacy_ordinal>.enabled` when that exists,
    /// otherwise true.
    enabled: Option<bool>,
    /// gRPC tiers only. Defaults to `router.tier<legacy_ordinal>`.
    endpoint: Option<String>,
    timeout_ms: Option<u64>,
}

/// One entry of the tier registry. Position in `PipelineConfig::tiers` is the
/// execution order.
#[derive(Debug, Clone)]
pub struct TierSpec {
    pub id: TierId,
    pub kind: TierKind,
    pub joint: bool,
    pub legacy_ordinal: Option<u8>,
    pub enabled: Option<bool>,
    pub endpoint: Option<String>,
    pub timeout_ms: Option<u64>,
}

/// Registry used when a config has no `[[tier]]` entries: the original four
/// tiers, in their original order.
fn default_tier_specs() -> Vec<TierSpec> {
    [
        ("rules", TierKind::Rules, 1),
        ("fasttext", TierKind::Fasttext, 2),
        ("modernbert", TierKind::Grpc, 3),
        ("harrier", TierKind::Grpc, 4),
    ]
    .into_iter()
    .map(|(name, kind, ordinal)| TierSpec {
        id: TierId::new(name).expect("built-in tier names are valid"),
        kind,
        joint: false,
        legacy_ordinal: Some(ordinal),
        enabled: None,
        endpoint: None,
        timeout_ms: None,
    })
    .collect()
}

fn tier_specs_from(raw: Vec<RawTier>) -> Result<Vec<TierSpec>, Box<dyn Error>> {
    if raw.is_empty() {
        return Ok(default_tier_specs());
    }
    let mut specs: Vec<TierSpec> = Vec::with_capacity(raw.len());
    for tier in raw {
        let id = TierId::new(tier.name.clone()).map_err(|error| format!("[[tier]]: {error}"))?;
        let name = id.as_str().to_string();
        if specs.iter().any(|existing| existing.id == id) {
            return Err(format!("[[tier]] {name:?} is defined twice").into());
        }
        if let Some(ordinal) = tier.legacy_ordinal {
            if ordinal == 0 {
                return Err(format!("[[tier]] {name:?}: legacy_ordinal must be positive").into());
            }
            if specs.iter().any(|existing| existing.legacy_ordinal == Some(ordinal)) {
                return Err(format!(
                    "[[tier]] {name:?}: legacy_ordinal {ordinal} is already used by another tier"
                )
                .into());
            }
        }
        if tier.kind != TierKind::Grpc {
            if tier.endpoint.is_some() || tier.timeout_ms.is_some() {
                return Err(format!(
                    "[[tier]] {name:?}: endpoint and timeout_ms apply to kind = \"grpc\" only"
                )
                .into());
            }
            if tier.joint {
                return Err(format!("[[tier]] {name:?}: joint applies to kind = \"grpc\" only").into());
            }
            if specs.iter().any(|existing| existing.kind == tier.kind) {
                return Err(format!(
                    "[[tier]] {name:?}: only one tier of kind {:?} may be configured",
                    tier.kind
                )
                .into());
            }
        }
        if tier.endpoint.as_deref().is_some_and(|endpoint| endpoint.trim().is_empty()) {
            return Err(format!("[[tier]] {name:?}: endpoint must be non-empty").into());
        }
        if tier.timeout_ms == Some(0) {
            return Err(format!("[[tier]] {name:?}: timeout_ms must be greater than zero").into());
        }
        specs.push(TierSpec {
            id,
            kind: tier.kind,
            joint: tier.joint,
            legacy_ordinal: tier.legacy_ordinal,
            enabled: tier.enabled,
            endpoint: tier.endpoint,
            timeout_ms: tier.timeout_ms,
        });
    }
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
    /// The tier registry in execution order.
    pub tiers: Vec<TierSpec>,
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
    let mut tiers = tier_specs_from(raw.tier)?;
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
        let key = match reference {
            TierRef::Name(name) => name.clone(),
            TierRef::Ordinal(ordinal) => format!("tier{ordinal}"),
        };
        let id = find_tier(&tiers, &key).map(|spec| spec.id.clone()).ok_or_else(|| {
            format!(
                "pipeline.joint_tiers: unknown tier {key:?}; registered tiers: {}",
                known_tier_names(&tiers)
            )
        })?;
        let spec = tiers.iter_mut().find(|spec| spec.id == id).expect("tier was just found");
        if spec.kind != TierKind::Grpc {
            return Err(format!("pipeline.joint_tiers: {id} is not a grpc tier").into());
        }
        spec.joint = true;
    }

    Ok(PipelineConfig {
        settings: raw.pipeline,
        tiers,
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
        assert!(!cfg.settings.tag_in_request);
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

        // No [[tier]] entries: the original four tiers, with `tier4` an alias
        // for harrier and `joint_tiers = [4]` marking it joint.
        let harrier = cfg.tiers.iter().find(|tier| tier.id.as_str() == "harrier").unwrap();
        assert!(harrier.joint);
        assert!(!cfg.tiers.iter().find(|tier| tier.id.as_str() == "modernbert").unwrap().joint);
        let tag = cfg.profile(&TaskId::tag()).unwrap();
        assert_eq!(tag.fallback, FallbackPolicy::Defer);
        assert_eq!(
            cfg.tier_policies[&harrier.id][&TaskId::tag()].label_min_confidence["engineering"],
            0.9
        );
    }

    #[test]
    fn taxonomy_supplies_label_metadata_and_version() {
        let cfg = parse_with_base(
            "[task.tag]\ntaxonomy = \"../taskonomies/help_desk.json\"\noutput_kind = \"multiclass\"\nfallback = \"defer\"",
            Path::new("config"),
        ).unwrap();
        let tag = cfg.profile(&TaskId::tag()).unwrap();
        assert_eq!(tag.output_kind, OutputKind::Multiclass);
        assert_eq!(tag.labels.as_ref().unwrap().len(), 20);
        assert_eq!(tag.taxonomy_version.as_deref(), Some("1.1.0"));
        assert_eq!(tag.label_metadata[0].id, "app_access_req");
        assert!(!tag.label_metadata[0].classifier_criterion.is_empty());
    }

    #[test]
    fn rejects_unknown_tiers_bad_thresholds_and_unknown_task_references() {
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n\n[tier_policy.tier0.route]\nmin_confidence = 0.5").is_err());
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n\n[tier_policy.nonesuch.route]\nmin_confidence = 0.5").is_err());
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n\n[tier_policy.tier3.route]\nmin_confidence = 1.5").is_err());
        assert!(parse("[task.route]\nfallback_label = \"needs_llm\"\n\n[tier_policy.tier3.tag]\nmin_confidence = 0.5").is_err());
        assert!(parse("[task.tag]\nfallback = \"defer\"\nfallback_label = \"x\"").is_err());
        assert!(parse("[task.tag]\nlabels = [\"engineering\"]\nfallback = \"defer\"\n\n[tier_policy.tier3.tag]\nmin_confidence = 0.5\nnever_resolve = [\"unknown\"]").is_err());
    }

    const TIERS: &str = r#"
        [task.route]
        fallback_label = "needs_llm"
        labels = ["jev_capable", "needs_llm"]

        [[tier]]
        name = "fasttext"
        kind = "fasttext"
        legacy_ordinal = 2

        [[tier]]
        name = "triage"
        kind = "grpc"
        endpoint = "http://127.0.0.1:50061"
        timeout_ms = 150
        joint = true

        [[tier]]
        name = "modernbert"
        kind = "grpc"
        legacy_ordinal = 3
    "#;

    #[test]
    fn tier_array_order_is_execution_order() {
        let cfg = parse(TIERS).unwrap();
        let names: Vec<_> = cfg.tiers.iter().map(|tier| tier.id.as_str()).collect();
        assert_eq!(names, ["fasttext", "triage", "modernbert"]);
        let triage = &cfg.tiers[1];
        assert_eq!(triage.kind, TierKind::Grpc);
        assert!(triage.joint);
        assert_eq!(triage.legacy_ordinal, None);
        assert_eq!(triage.endpoint.as_deref(), Some("http://127.0.0.1:50061"));
        assert_eq!(triage.timeout_ms, Some(150));
    }

    #[test]
    fn missing_tier_section_yields_the_original_four_in_order() {
        let cfg = parse("").unwrap();
        let tiers: Vec<_> = cfg.tiers.iter().map(|t| (t.id.as_str(), t.legacy_ordinal)).collect();
        assert_eq!(
            tiers,
            [("rules", Some(1)), ("fasttext", Some(2)), ("modernbert", Some(3)), ("harrier", Some(4))]
        );
    }

    #[test]
    fn policies_resolve_by_name_or_legacy_alias_and_reject_unknown_tiers() {
        let by_name = format!("{TIERS}\n[tier_policy.triage.route]\nmin_confidence = 0.8\n");
        let cfg = parse(&by_name).unwrap();
        assert!(cfg.tier_policies.keys().any(|id| id.as_str() == "triage"));

        let by_alias = format!("{TIERS}\n[tier_policy.tier3.route]\nmin_confidence = 0.9\n");
        let cfg = parse(&by_alias).unwrap();
        assert!(cfg.tier_policies.keys().any(|id| id.as_str() == "modernbert"));

        // triage has no legacy ordinal, so no tierN alias reaches it.
        let unreachable = format!("{TIERS}\n[tier_policy.tier1.route]\nmin_confidence = 0.9\n");
        assert!(parse(&unreachable).is_err());

        let both = format!(
            "{TIERS}\n[tier_policy.modernbert.route]\nmin_confidence = 0.9\n[tier_policy.tier3.route]\nmin_confidence = 0.8\n"
        );
        assert!(parse(&both).is_err());
    }

    #[test]
    fn tier_registry_entries_are_validated() {
        let base = "[task.route]\nfallback_label = \"needs_llm\"\n";
        let cases = [
            "[[tier]]\nname = \"a\"\nkind = \"grpc\"\n[[tier]]\nname = \"a\"\nkind = \"grpc\"",
            "[[tier]]\nname = \"Bad Name\"\nkind = \"grpc\"",
            "[[tier]]\nname = \"fallback\"\nkind = \"grpc\"",
            "[[tier]]\nname = \"tier3\"\nkind = \"grpc\"",
            "[[tier]]\nname = \"a\"\nkind = \"rules\"\n[[tier]]\nname = \"b\"\nkind = \"rules\"",
            "[[tier]]\nname = \"a\"\nkind = \"rules\"\nendpoint = \"http://x\"",
            "[[tier]]\nname = \"a\"\nkind = \"fasttext\"\njoint = true",
            "[[tier]]\nname = \"a\"\nkind = \"grpc\"\ntimeout_ms = 0",
            "[[tier]]\nname = \"a\"\nkind = \"grpc\"\nlegacy_ordinal = 2\n[[tier]]\nname = \"b\"\nkind = \"grpc\"\nlegacy_ordinal = 2",
            "[[tier]]\nname = \"a\"\nkind = \"grpc\"\nsurprise = 1",
        ];
        for case in cases {
            assert!(parse(&format!("{base}\n{case}")).is_err(), "should reject:\n{case}");
        }
    }

    #[test]
    fn joint_tiers_accept_names_and_legacy_numbers_and_reject_non_grpc_tiers() {
        let by_name = format!("{TIERS}\n[pipeline]\njoint_tiers = [\"modernbert\"]\n");
        // [pipeline] must precede [[tier]] only in TOML table order; append is fine
        // because the tables are independent.
        let cfg = parse(&by_name).unwrap();
        assert!(cfg.tiers.iter().find(|t| t.id.as_str() == "modernbert").unwrap().joint);
        assert!(cfg.tiers.iter().find(|t| t.id.as_str() == "triage").unwrap().joint);

        let by_number = format!("{TIERS}\n[pipeline]\njoint_tiers = [3]\n");
        assert!(parse(&by_number).unwrap().tiers.iter().find(|t| t.id.as_str() == "modernbert").unwrap().joint);

        assert!(parse(&format!("{TIERS}\n[pipeline]\njoint_tiers = [\"fasttext\"]\n")).is_err());
        assert!(parse(&format!("{TIERS}\n[pipeline]\njoint_tiers = [\"ghost\"]\n")).is_err());
    }

    #[test]
    fn shipped_tasks_toml_declares_the_registry_and_names_every_policy_tier() {
        let cfg = load_or_default("config/tasks.toml").unwrap();
        let order: Vec<_> = cfg.tiers.iter().map(|tier| tier.id.as_str()).collect();
        assert_eq!(order, ["rules", "fasttext", "modernbert", "harrier"]);
        for tier in cfg.tier_policies.keys() {
            assert!(
                cfg.tiers.iter().any(|spec| &spec.id == tier),
                "policy for unregistered tier {tier}"
            );
        }
        assert!(cfg.has_policy(&TaskId::route()));
    }
}
