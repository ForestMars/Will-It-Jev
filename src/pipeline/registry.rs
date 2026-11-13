// src/pipeline/registry.rs
//
// Unordered inventory of available, initialized tier instances. The registry
// resolves authored TierNames only; execution identity and ordering belong to
// CascadeGraph.

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

/// The complete inventory of available tier instances, keyed only by their
/// authored configuration name. It has no execution-graph identity or order.
#[derive(Default)]
pub struct TierRegistry {
    entries: BTreeMap<TierName, Arc<TierInstance>>,
}

impl TierRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(&mut self, tier: Arc<TierInstance>) -> Result<(), RegistryError> {
        if self.entries.contains_key(tier.name()) {
            return Err(RegistryError::DuplicateName(tier.name().clone()));
        }
        self.entries.insert(tier.name().clone(), tier);
        Ok(())
    }

    pub fn get(&self, name: &TierName) -> Option<Arc<TierInstance>> {
        self.entries.get(name).map(Arc::clone)
    }

    pub fn names(&self) -> BTreeSet<TierName> {
        self.entries.keys().cloned().collect()
    }

    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    pub fn iter(&self) -> impl Iterator<Item = (&TierName, &Arc<TierInstance>)> {
        self.entries.iter()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize { self.entries.len() }

    #[cfg(test)]
    pub fn of(tiers: Vec<Arc<TierInstance>>) -> Self {
        let mut registry = Self::new();
        for tier in tiers {
            registry.register(tier).expect("test tiers must have unique names");
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::tier::{tname, FakeEvaluate, TierContext, TierOutput};
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
    fn registry_is_an_unordered_inventory_indexed_only_by_name() {
        let mut registry = TierRegistry::new();
        registry.register(fake("modernbert")).unwrap();
        registry.register(fake("rules")).unwrap();
        assert_eq!(registry.len(), 2);
        assert!(registry.get(&tname("modernbert")).is_some());
        assert!(registry.get(&tname("rules")).is_some());
        assert_eq!(registry.names(), BTreeSet::from([tname("modernbert"), tname("rules")]));
    }

    #[test]
    fn duplicate_names_are_rejected_without_replacing_the_registered_tier() {
        let mut registry = TierRegistry::new();
        registry.register(fake("rules")).unwrap();
        assert_eq!(registry.register(fake("rules")).err(), Some(RegistryError::DuplicateName(tname("rules"))));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn registry_retains_available_tiers_independently_of_execution_membership() {
        let registry = TierRegistry::of(vec![fake("rules"), fake("modernbert"), fake("harrier")]);
        assert_eq!(registry.len(), 3);
        assert!(registry.get(&tname("harrier")).is_some());
    }
}
