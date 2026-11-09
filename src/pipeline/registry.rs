// src/pipeline/registry.rs
//
// Tier definitions and configured tier instances are distinct:
//
// * TierId identifies an execution contract (rules, fasttext, grpc).
// * TierInstanceId identifies one configured realization of that contract.
// * TierDefinitionRegistry is the catalog of available contracts.
// * TierRegistry is the ordered sequence of configured instances. Its vector
//   order is the execution order; names and model IDs never imply a position.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use serde::Deserialize;

use super::tier::Tier;

pub const FALLBACK_TIER_NAME: &str = "fallback";

const MAX_ID_LEN: usize = 64;

fn validate_id(kind: &str, value: &str) -> Result<(), String> {
    let mut chars = value.chars();
    let valid = !value.is_empty()
        && value.len() <= MAX_ID_LEN
        && chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
    if !valid {
        return Err(format!(
            "invalid {kind} {value:?}: use 1 to {MAX_ID_LEN} characters, starting with a lowercase letter, then lowercase letters, digits, '_' or '-'"
        ));
    }
    if value == FALLBACK_TIER_NAME {
        return Err(format!("{kind} name {FALLBACK_TIER_NAME:?} is reserved"));
    }
    Ok(())
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
    pub fn new(id: impl Into<String>) -> Result<Self, TierIdError> {
        let id = id.into();
        validate_id("tier definition", &id).map_err(TierIdError)?;
        Ok(Self(id))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}
impl TierInstanceId {
    pub fn new(id: impl Into<String>) -> Result<Self, TierIdError> {
        let id = id.into();
        validate_id("tier instance", &id).map_err(TierIdError)?;
        Ok(Self(id))
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

/// The built-in execution contracts available to configured tier instances.
/// Adding another instance of an existing definition does not add a definition.
#[derive(Debug, Clone, Default)]
pub struct TierDefinitionRegistry {
    definitions: BTreeSet<TierId>,
}

impl TierDefinitionRegistry {
    pub fn builtins() -> Self {
        let mut registry = Self::default();
        for id in ["rules", "fasttext", "grpc"] {
            registry.register(TierId::new(id).expect("built-in tier definition is valid"));
        }
        registry
    }

    pub fn register(&mut self, id: TierId) -> bool {
        self.definitions.insert(id)
    }

    pub fn contains(&self, id: &TierId) -> bool {
        self.definitions.contains(id)
    }

    pub fn ids(&self) -> impl Iterator<Item = &TierId> {
        self.definitions.iter()
    }
}

/// One configured realization of a tier definition. The executor implements
/// the shared contract; endpoint, selected model and other per-instance
/// settings belong to this value, not to TierId.
pub struct TierInstance {
    pub id: TierInstanceId,
    pub tier_id: TierId,
    pub model_id: Option<String>,
    executor: Arc<dyn Tier>,
}

impl TierInstance {
    pub fn new(
        id: TierInstanceId,
        tier_id: TierId,
        model_id: Option<String>,
        executor: Arc<dyn Tier>,
    ) -> Result<Self, RegistryError> {
        if executor.id() != &tier_id {
            return Err(RegistryError::ImplementationMismatch {
                instance: id,
                expected: tier_id,
                actual: executor.id().clone(),
            });
        }
        Ok(Self { id, tier_id, model_id, executor })
    }

    pub fn executor(&self) -> &Arc<dyn Tier> {
        &self.executor
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateInstance(TierInstanceId),
    UnknownDefinition(TierId),
    ImplementationMismatch {
        instance: TierInstanceId,
        expected: TierId,
        actual: TierId,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateInstance(id) => write!(f, "tier instance {id:?} is registered twice"),
            Self::UnknownDefinition(id) => write!(f, "tier definition {id:?} is not registered"),
            Self::ImplementationMismatch { instance, expected, actual } => write!(
                f,
                "tier instance {instance:?} declares definition {expected:?}, but its executor implements {actual:?}"
            ),
        }
    }
}
impl Error for RegistryError {}

#[derive(Debug, Clone)]
pub struct TierInstanceInfo {
    pub id: TierInstanceId,
    pub tier_id: TierId,
    /// Zero-based position derived from the configured vector.
    pub position: usize,
}

#[derive(Debug, Clone, Default)]
pub struct TierDirectory {
    infos: Vec<TierInstanceInfo>,
}

impl TierDirectory {
    pub fn infos(&self) -> &[TierInstanceInfo] { &self.infos }

    pub fn position(&self, id: &TierInstanceId) -> Option<usize> {
        self.infos.iter().find(|info| &info.id == id).map(|info| info.position)
    }

    pub fn get(&self, id: &TierInstanceId) -> Option<&TierInstanceInfo> {
        self.infos.iter().find(|info| &info.id == id)
    }
}

/// Ordered collection of configured instances. The execution sequence is
/// explicitly a vector of IDs; instance values live in a separate lookup map.
pub struct TierRegistry {
    definitions: TierDefinitionRegistry,
    sequence: Vec<TierInstanceId>,
    instances: BTreeMap<TierInstanceId, TierInstance>,
}

impl Default for TierRegistry {
    fn default() -> Self { Self::new() }
}

impl TierRegistry {
    pub fn new() -> Self {
        Self::with_definitions(TierDefinitionRegistry::builtins())
    }

    pub fn with_definitions(definitions: TierDefinitionRegistry) -> Self {
        Self {
            definitions,
            sequence: Vec::new(),
            instances: BTreeMap::new(),
        }
    }

    pub fn definitions(&self) -> &TierDefinitionRegistry { &self.definitions }

    pub fn register(&mut self, instance: TierInstance) -> Result<(), RegistryError> {
        if !self.definitions.contains(&instance.tier_id) {
            return Err(RegistryError::UnknownDefinition(instance.tier_id));
        }
        if self.instances.contains_key(&instance.id) {
            return Err(RegistryError::DuplicateInstance(instance.id));
        }
        self.sequence.push(instance.id.clone());
        self.instances.insert(instance.id.clone(), instance);
        Ok(())
    }

    pub fn instance(&self, id: &TierInstanceId) -> Option<&TierInstance> {
        self.instances.get(id)
    }

    pub fn sequence(&self) -> &[TierInstanceId] {
        &self.sequence
    }

    pub fn is_empty(&self) -> bool { self.sequence.is_empty() }

    pub fn directory(&self) -> TierDirectory {
        TierDirectory {
            infos: self.sequence.iter().enumerate().filter_map(|(position, id)| {
                self.instances.get(id).map(|instance| TierInstanceInfo {
                    id: instance.id.clone(),
                    tier_id: instance.tier_id.clone(),
                    position,
                })
            }).collect(),
        }
    }

    pub(crate) fn into_ordered(self) -> (Vec<TierInstanceId>, BTreeMap<TierInstanceId, TierInstance>) {
        (self.sequence, self.instances)
    }

    #[cfg(test)]
    pub fn of(instances: Vec<TierInstance>) -> Self {
        let mut registry = Self::new();
        for instance in instances {
            registry.register(instance).expect("test tier instances must be valid");
        }
        registry
    }
}

#[cfg(test)]
pub fn tid(name: &str) -> TierId {
    TierId::new(name).expect("test tier ID must be valid")
}

#[cfg(test)]
pub fn iid(name: &str) -> TierInstanceId {
    TierInstanceId::new(name).expect("test tier instance ID must be valid")
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
        fn name(&self) -> &str { self.0.as_str() }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            TierOutput::abstain_all(self.0.as_str(), ctx.tasks, "test")
        }
    }

    fn instance(name: &str, definition: &str, model: &str) -> TierInstance {
        let tier_id = tid(definition);
        TierInstance::new(
            iid(name),
            tier_id.clone(),
            Some(model.into()),
            Arc::new(Named(tier_id)),
        ).unwrap()
    }

    #[test]
    fn identifiers_are_validated_independently() {
        assert!(TierId::new("grpc").is_ok());
        assert!(TierInstanceId::new("encoder_blue").is_ok());
        assert!(TierId::new("tier3").is_ok());
        assert!(TierInstanceId::new("tier3").is_ok());
        for bad in ["", "Bad Name", "2fast", "fallback", &"a".repeat(65)] {
            assert!(TierId::new(bad).is_err());
            assert!(TierInstanceId::new(bad).is_err());
        }
    }

    #[test]
    fn multiple_instances_share_one_registered_definition() {
        let mut registry = TierRegistry::new();
        registry.register(instance("encoder_a", "grpc", "modernbert-base")).unwrap();
        registry.register(instance("encoder_b", "grpc", "harrier-v1")).unwrap();
        assert!(registry.definitions().contains(&tid("grpc")));
        let infos = registry.directory().infos().to_vec();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].tier_id, tid("grpc"));
        assert_eq!(infos[1].tier_id, tid("grpc"));
        assert_eq!(infos[0].id, iid("encoder_a"));
        assert_eq!(infos[1].id, iid("encoder_b"));
    }

    #[test]
    fn vector_order_is_independent_of_instance_names() {
        let registry = TierRegistry::of(vec![
            instance("zeta", "grpc", "m1"),
            instance("alpha", "grpc", "m2"),
            instance("middle", "grpc", "m3"),
        ]);
        let directory = registry.directory();
        let order: Vec<_> = directory.infos().iter().map(|info| info.id.as_str()).collect();
        assert_eq!(order, ["zeta", "alpha", "middle"]);
        assert_eq!(directory.position(&iid("alpha")), Some(1));
    }

    #[test]
    fn duplicate_instance_ids_are_rejected_but_shared_definitions_are_not() {
        let mut registry = TierRegistry::new();
        registry.register(instance("encoder", "grpc", "m1")).unwrap();
        assert_eq!(
            registry.register(instance("encoder", "grpc", "m2")),
            Err(RegistryError::DuplicateInstance(iid("encoder")))
        );
    }
}
