// src/pipeline/registry.rs
//
// Architecture definitions and configured instances are distinct.
// Definitions identify reusable execution contracts. Instances select a
// definition, model, and executor. The ordered instance ID vector is the
// cascade sequence; positions are derived by enumeration only.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use serde::Deserialize;

use super::tier::Tier;

pub const FALLBACK_TIER_NAME: &str = "fallback";
const MAX_TIER_NAME_LEN: usize = 32;

fn validate_name(name: String, kind: &str) -> Result<String, TierIdError> {
    let mut chars = name.chars();
    let valid = !name.is_empty()
        && name.len() <= MAX_TIER_NAME_LEN
        && chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
    if !valid {
        return Err(TierIdError(format!(
            "invalid {kind} name {name:?}: use 1 to {MAX_TIER_NAME_LEN} characters, starting with a lowercase letter, then lowercase letters, digits, '_' or '-'"
        )));
    }
    if name == FALLBACK_TIER_NAME {
        return Err(TierIdError(format!(
            "{kind} name {FALLBACK_TIER_NAME:?} is reserved for task fallbacks"
        )));
    }
    Ok(name)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
#[serde(try_from = "String")]
pub struct TierId(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
#[serde(try_from = "String")]
pub struct TierInstanceId(String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierIdError(String);

impl fmt::Display for TierIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}
impl Error for TierIdError {}

impl TierId {
    pub fn new(name: impl Into<String>) -> Result<Self, TierIdError> {
        let name = validate_name(name.into(), "tier")?;
        if name.strip_prefix("tier").is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())) {
            return Err(TierIdError(format!(
                "tier name {name:?} is reserved; numeric tier aliases are not supported"
            )));
        }
        Ok(Self(name))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}
impl TierInstanceId {
    pub fn new(name: impl Into<String>) -> Result<Self, TierIdError> {
        let name = validate_name(name.into(), "tier instance")?;
        if name.strip_prefix("tier").is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())) {
            return Err(TierIdError(format!(
                "tier instance name {name:?} is reserved; numeric tier aliases are not supported"
            )));
        }
        Ok(Self(name))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}
impl TryFrom<String> for TierId {
    type Error = TierIdError;
    fn try_from(value: String) -> Result<Self, Self::Error> { Self::new(value) }
}
impl TryFrom<String> for TierInstanceId {
    type Error = TierIdError;
    fn try_from(value: String) -> Result<Self, Self::Error> { Self::new(value) }
}
impl fmt::Display for TierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}
impl fmt::Display for TierInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}

#[cfg(test)]
pub fn tid(name: &str) -> TierId { TierId::new(name).expect("test tier name must be valid") }
#[cfg(test)]
pub fn iid(name: &str) -> TierInstanceId {
    TierInstanceId::new(name).expect("test instance name must be valid")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierKind { Rules, Fasttext, Grpc }

/// Immutable definition of the execution contract and its implementation kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierDefinition {
    pub id: TierId,
    pub kind: TierKind,
}
impl TierDefinition {
    pub fn new(id: TierId, kind: TierKind) -> Self { Self { id, kind } }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefinitionRegistryError {
    Duplicate(TierId),
}
impl fmt::Display for DefinitionRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Duplicate(id) => write!(f, "tier definition {id} is registered twice") }
    }
}
impl Error for DefinitionRegistryError {}

/// Catalog of reusable tier architecture definitions.
#[derive(Debug, Clone, Default)]
pub struct TierDefinitionRegistry {
    definitions: BTreeMap<TierId, TierDefinition>,
}
impl TierDefinitionRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn register(&mut self, definition: TierDefinition) -> Result<(), DefinitionRegistryError> {
        let id = definition.id.clone();
        if self.definitions.contains_key(&id) {
            return Err(DefinitionRegistryError::Duplicate(id));
        }
        self.definitions.insert(id, definition);
        Ok(())
    }
    pub fn get(&self, id: &TierId) -> Option<&TierDefinition> { self.definitions.get(id) }
    pub fn contains(&self, id: &TierId) -> bool { self.definitions.contains_key(id) }
}

/// One configured realization of a reusable tier definition.
#[derive(Clone)]
pub struct TierInstance {
    pub id: TierInstanceId,
    pub tier_id: TierId,
    /// Configured model reference. The executor may report a resolved model ID
    /// in TierOutput; both refer to this instance's selected model.
    pub model_id: String,
    pub(crate) executor: Arc<dyn Tier>,
}
impl TierInstance {
    pub fn new(
        id: TierInstanceId,
        tier_id: TierId,
        model_id: impl Into<String>,
        executor: Arc<dyn Tier>,
    ) -> Result<Self, InstanceError> {
        if executor.id() != &tier_id {
            return Err(InstanceError::DefinitionMismatch {
                instance: id,
                configured: tier_id,
                executor: executor.id().clone(),
            });
        }
        let model_id = model_id.into();
        if model_id.trim().is_empty() {
            return Err(InstanceError::EmptyModelId(id));
        }
        Ok(Self { id, tier_id, model_id, executor })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstanceError {
    Duplicate(TierInstanceId),
    UnknownDefinition(TierId),
    DefinitionMismatch { instance: TierInstanceId, configured: TierId, executor: TierId },
    EmptyModelId(TierInstanceId),
}
impl fmt::Display for InstanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Duplicate(id) => write!(f, "tier instance {id} is registered twice"),
            Self::UnknownDefinition(id) => write!(f, "tier instance references unknown definition {id}"),
            Self::DefinitionMismatch { instance, configured, executor } =>
                write!(f, "tier instance {instance} references definition {configured}, but its executor implements {executor}"),
            Self::EmptyModelId(id) => write!(f, "tier instance {id} has an empty model_id"),
        }
    }
}
impl Error for InstanceError {}

#[derive(Debug, Clone)]
pub struct TierInfo {
    pub id: TierInstanceId,
    pub tier_id: TierId,
    /// Zero-based position in the configured execution vector.
    pub position: usize,
}

#[derive(Debug, Clone, Default)]
pub struct TierDirectory {
    infos: Vec<TierInfo>,
}
impl TierDirectory {
    pub fn infos(&self) -> &[TierInfo] { &self.infos }
    pub fn position(&self, id: &TierInstanceId) -> Option<usize> {
        self.infos.iter().find(|info| &info.id == id).map(|info| info.position)
    }
    pub fn id_at_position(&self, position: usize) -> Option<&TierInstanceId> {
        self.infos.get(position).map(|info| &info.id)
    }
    pub fn tier_id(&self, id: &TierInstanceId) -> Option<&TierId> {
        self.infos.iter().find(|info| &info.id == id).map(|info| &info.tier_id)
    }
    /// Compatibility projection for the four historical numbered telemetry
    /// columns. It is derived from vector position, never configured identity.
    pub fn legacy_ordinal(&self, id: &TierInstanceId) -> Option<u8> {
        self.position(id).and_then(|position| u8::try_from(position + 1).ok())
    }
}

/// Ordered registry of configured tier instances. Definitions are catalogued
/// separately; repeated instances may reference the same TierId.
#[derive(Default)]
pub struct TierRegistry {
    definitions: TierDefinitionRegistry,
    order: Vec<TierInstanceId>,
    instances: BTreeMap<TierInstanceId, TierInstance>,
}
impl TierRegistry {
    pub fn new(definitions: TierDefinitionRegistry) -> Self {
        Self { definitions, order: Vec::new(), instances: BTreeMap::new() }
    }
    pub fn register(&mut self, instance: TierInstance) -> Result<(), InstanceError> {
        if !self.definitions.contains(&instance.tier_id) {
            return Err(InstanceError::UnknownDefinition(instance.tier_id));
        }
        if self.instances.contains_key(&instance.id) {
            return Err(InstanceError::Duplicate(instance.id));
        }
        self.order.push(instance.id.clone());
        self.instances.insert(instance.id.clone(), instance);
        Ok(())
    }
    pub fn is_empty(&self) -> bool { self.order.is_empty() }
    pub fn directory(&self) -> TierDirectory {
        TierDirectory {
            infos: self.order.iter().enumerate().filter_map(|(position, id)| {
                self.instances.get(id).map(|instance| TierInfo {
                    id: id.clone(),
                    tier_id: instance.tier_id.clone(),
                    position,
                })
            }).collect(),
        }
    }
    pub(crate) fn into_parts(self) -> (Vec<TierInstanceId>, BTreeMap<TierInstanceId, TierInstance>, TierDirectory) {
        let directory = self.directory();
        (self.order, self.instances, directory)
    }
    /// Test helper: each executor gets a definition and a same-named instance.
    #[cfg(test)]
    pub fn of(tiers: Vec<Arc<dyn Tier>>) -> Self {
        let mut definitions = TierDefinitionRegistry::new();
        for tier in &tiers {
            let kind = match tier.id().as_str() {
                "rules" => TierKind::Rules,
                "fasttext" => TierKind::Fasttext,
                _ => TierKind::Grpc,
            };
            definitions.register(TierDefinition::new(tier.id().clone(), kind)).expect("test definitions must be distinct");
        }
        let mut registry = Self::new(definitions);
        for tier in tiers {
            let id = TierInstanceId::new(tier.id().as_str()).expect("test instance name must be valid");
            let tier_id = tier.id().clone();
            registry.register(TierInstance::new(id, tier_id, tier.name(), tier).unwrap())
                .expect("test instances must be distinct");
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::tier::{TierContext, TierOutput};
    use async_trait::async_trait;

    struct Named(TierId);
    #[async_trait]
    impl Tier for Named {
        fn id(&self) -> &TierId { &self.0 }
        fn name(&self) -> &str { "named" }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            TierOutput::abstain_all("named", ctx.tasks, "test")
        }
    }
    fn named(name: &str) -> Arc<dyn Tier> { Arc::new(Named(tid(name))) }

    #[test]
    fn names_are_validated_separately_for_definitions_and_instances() {
        for ok in ["rules", "fasttext", "grpc", "modernbert", "harrier", "oryn-2", "a"] {
            assert!(TierId::new(ok).is_ok());
            assert!(TierInstanceId::new(ok).is_ok());
        }
        for bad in ["", "Rules", "2fast", "has space", "x.y", "fallback", "tier3"] {
            assert!(TierId::new(bad).is_err());
            assert!(TierInstanceId::new(bad).is_err());
        }
    }

    #[test]
    fn multiple_instances_can_share_one_definition_and_keep_vector_order() {
        let grpc = tid("grpc");
        let mut definitions = TierDefinitionRegistry::new();
        definitions.register(TierDefinition::new(grpc.clone(), TierKind::Grpc)).unwrap();
        let mut registry = TierRegistry::new(definitions);
        let first = Arc::new(Named(grpc.clone())) as Arc<dyn Tier>;
        let second = Arc::new(Named(grpc.clone())) as Arc<dyn Tier>;
        registry.register(TierInstance::new(iid("zulu"), grpc.clone(), "model-a", first).unwrap()).unwrap();
        registry.register(TierInstance::new(iid("alpha"), grpc.clone(), "model-b", second).unwrap()).unwrap();
        let directory = registry.directory();
        assert_eq!(directory.infos().iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), ["zulu", "alpha"]);
        assert_eq!(directory.infos()[0].tier_id, grpc);
        assert_eq!(directory.position(&iid("alpha")), Some(1));
    }

    #[test]
    fn instances_reject_unknown_definitions_and_empty_models() {
        let mut definitions = TierDefinitionRegistry::new();
        definitions.register(TierDefinition::new(tid("rules"), TierKind::Rules)).unwrap();
        let mut registry = TierRegistry::new(definitions.clone());
        let error = registry.register(TierInstance::new(iid("missing"), tid("grpc"), "m", named("grpc")).unwrap()).unwrap_err();
        assert_eq!(error, InstanceError::UnknownDefinition(tid("grpc")));
        assert!(TierInstance::new(iid("bad"), tid("rules"), " ", named("rules")).is_err());
    }

    #[test]
    fn instance_identity_is_independent_of_model_selection() {
        let id = iid("encoder_a");
        let original = TierInstance::new(id.clone(), tid("grpc"), "modernbert-base", named("grpc")).unwrap();
        let substituted = TierInstance::new(id.clone(), tid("grpc"), "harrier-270m", named("grpc")).unwrap();
        assert_eq!(original.id, substituted.id);
        assert_eq!(original.tier_id, substituted.tier_id);
        assert_ne!(original.model_id, substituted.model_id);
    }
}
