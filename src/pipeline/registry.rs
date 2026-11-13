// src/pipeline/registry.rs
//
// Unordered inventory of available tier instances, addressed only by authored name.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::tier::{TierInstance, TierName};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateName(TierName),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "tier {name} is registered twice"),
        }
    }
}
impl Error for RegistryError {}

/// The complete inventory of available tier instances. It has no execution
/// order and no knowledge of graph identities.
#[derive(Default)]
pub struct TierRegistry {
    entries: BTreeMap<TierName, Arc<TierInstance>>,
}

impl TierRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(&mut self, tier: Arc<TierInstance>) -> Result<(), RegistryError> {
        let name = tier.name().clone();
        if self.entries.contains_key(&name) {
            return Err(RegistryError::DuplicateName(name));
        }
        self.entries.insert(name, tier);
        Ok(())
    }

    pub fn get(&self, name: &TierName) -> Option<Arc<TierInstance>> {
        self.entries.get(name).map(Arc::clone)
    }

    pub fn names(&self) -> BTreeSet<TierName> {
        self.entries.keys().cloned().collect()
    }

    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    #[cfg(test)]
    pub fn len(&self) -> usize { self.entries.len() }

    #[cfg(test)]
    pub fn of(tiers: Vec<Arc<TierInstance>>) -> Self {
        let mut registry = Self::new();
        for tier in tiers {
            registry.register(tier).expect("test tier names must be unique");
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::tier::tname;
    use crate::pipeline::tier::{FakeEvaluate, TierContext, TierOutput};
    use async_trait::async_trait;

    struct Silent;
    #[async_trait]
    impl FakeEvaluate for Silent {
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            TierOutput::abstain_all("silent", ctx.tasks, "test")
        }
    }

    fn fake(name: &str) -> Arc<TierInstance> {
        Arc::new(TierInstance::fake(name, Arc::new(Silent)))
    }

    #[test]
    fn registry_is_an_unordered_inventory_addressed_by_name() {
        let registry = TierRegistry::of(vec![fake("modernbert"), fake("rules"), fake("unused")]);
        assert_eq!(registry.len(), 3);
        assert!(registry.get(&tname("modernbert")).is_some());
        assert!(registry.get(&tname("rules")).is_some());
        assert!(registry.get(&tname("unused")).is_some());
        assert_eq!(
            registry.names(),
            [tname("modernbert"), tname("rules"), tname("unused")].into()
        );
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut registry = TierRegistry::new();
        registry.register(fake("rules")).unwrap();
        assert_eq!(registry.register(fake("rules")), Err(RegistryError::DuplicateName(tname("rules"))));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn different_tier_names_can_represent_separate_instances() {
        let registry = TierRegistry::of(vec![
            Arc::new(TierInstance::fake_with_model("first", "same-model", Arc::new(Silent))),
            Arc::new(TierInstance::fake_with_model("second", "same-model", Arc::new(Silent))),
        ]);
        assert_eq!(registry.len(), 2);
    }
}
