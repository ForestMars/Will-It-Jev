// src/pipeline/registry.rs
//
// Inventory of initialized tier instances. The registry is unordered; task
// cascades and the compiled CascadeGraph determine execution.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::identity::{Slot, TierId, TierName};
use super::task::TierPolicyMap;
use super::tier::TierInstance;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateName(TierName),
    DuplicateId { id: TierId, existing: TierName, duplicate: TierName },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "tier {name} is registered twice"),
            Self::DuplicateId { id, existing, duplicate } => write!(
                f,
                "tiers {existing} and {duplicate} unexpectedly share generated TierId {id}"
            ),
        }
    }
}
impl Error for RegistryError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierInfo {
    pub name: TierName,
    pub id: TierId,
    pub slot: Slot,
}

#[derive(Debug, Clone, Default)]
pub struct TierDirectory {
    infos: Vec<TierInfo>,
}

impl TierDirectory {
    pub fn infos(&self) -> &[TierInfo] { &self.infos }

    pub fn info(&self, id: &TierId) -> Option<&TierInfo> {
        self.infos.iter().find(|info| &info.id == id)
    }

    pub fn info_by_name(&self, name: &TierName) -> Option<&TierInfo> {
        self.infos.iter().find(|info| &info.name == name)
    }

    pub fn slot(&self, id: &TierId) -> Option<Slot> {
        self.info(id).map(|info| info.slot)
    }

    pub fn slot_by_name(&self, name: &TierName) -> Option<Slot> {
        self.info_by_name(name).map(|info| info.slot)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationMismatch {
    pub tier: TierName,
    pub tier_id: TierId,
    pub task: String,
    pub calibrated_for: TierId,
    pub running: TierId,
}

/// The complete inventory of available instances, indexed by authored name and
/// generated identity. Neither index implies an execution order.
#[derive(Default)]
pub struct TierRegistry {
    entries: BTreeMap<TierName, Arc<TierInstance>>,
    by_id: BTreeMap<TierId, TierName>,
}

impl TierRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(&mut self, tier: Arc<TierInstance>) -> Result<(), RegistryError> {
        if self.entries.contains_key(tier.name()) {
            return Err(RegistryError::DuplicateName(tier.name().clone()));
        }
        if let Some(existing) = self.by_id.get(tier.id()) {
            return Err(RegistryError::DuplicateId {
                id: *tier.id(),
                existing: existing.clone(),
                duplicate: tier.name().clone(),
            });
        }
        self.by_id.insert(*tier.id(), tier.name().clone());
        self.entries.insert(tier.name().clone(), tier);
        Ok(())
    }

    pub fn get(&self, name: &TierName) -> Option<Arc<TierInstance>> {
        self.entries.get(name).map(Arc::clone)
    }

    pub fn get_by_id(&self, id: &TierId) -> Option<Arc<TierInstance>> {
        self.by_id.get(id).and_then(|name| self.get(name))
    }

    pub fn names(&self) -> BTreeSet<TierName> {
        self.entries.keys().cloned().collect()
    }

    pub fn ids(&self) -> BTreeSet<TierId> {
        self.by_id.keys().copied().collect()
    }

    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    pub fn calibration_mismatches(&self, policies: &TierPolicyMap) -> Vec<CalibrationMismatch> {
        let mut mismatches = Vec::new();
        for (tier_id, by_task) in policies {
            let Some(instance) = self.get_by_id(tier_id) else { continue };
            for (task, policy) in by_task {
                if let Some(calibrated_for) = policy.calibrated_for {
                    if calibrated_for != *instance.id() {
                        mismatches.push(CalibrationMismatch {
                            tier: instance.name().clone(),
                            tier_id: *tier_id,
                            task: task.clone(),
                            calibrated_for,
                            running: *instance.id(),
                        });
                    }
                }
            }
        }
        mismatches
    }

    #[cfg(test)]
    pub fn len(&self) -> usize { self.entries.len() }

    #[cfg(test)]
    pub fn of(tiers: Vec<Arc<TierInstance>>) -> Self {
        let mut registry = Self::new();
        for tier in tiers {
            registry.register(tier).expect("test tiers must have unique names and ids");
        }
        registry
    }
}

impl TierDirectory {
    /// Project graph order to positional slots and stable identities.
    pub fn from_order(registry: &TierRegistry, order: &[TierId]) -> Self {
        Self {
            infos: order.iter().enumerate().filter_map(|(index, id)| {
                let tier = registry.get_by_id(id)?;
                Some(TierInfo {
                    name: tier.name().clone(),
                    id: *id,
                    slot: Slot::from_index(index)?,
                })
            }).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::identity::tname;
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
    fn registry_is_an_unordered_inventory_indexed_by_name_and_id() {
        let mut registry = TierRegistry::new();
        let modernbert = fake("modernbert");
        let modernbert_id = *modernbert.id();
        let rules = fake("rules");
        let rules_id = *rules.id();
        registry.register(modernbert).unwrap();
        registry.register(rules).unwrap();
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.get_by_id(&modernbert_id).unwrap().name(), &tname("modernbert"));
        let directory = TierDirectory::from_order(&registry, &[rules_id, modernbert_id]);
        assert_eq!(directory.slot(&rules_id).map(Slot::index), Some(0));
        assert_eq!(directory.slot(&modernbert_id).map(Slot::index), Some(1));
        assert_eq!(directory.info(&modernbert_id).map(|info| info.name.clone()), Some(tname("modernbert")));
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut registry = TierRegistry::new();
        registry.register(fake("rules")).unwrap();
        assert_eq!(registry.register(fake("rules")), Err(RegistryError::DuplicateName(tname("rules"))));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn different_tier_names_produce_distinct_instance_ids() {
        let mut registry = TierRegistry::new();
        let first = Arc::new(TierInstance::fake_with_model("first", "same-model", Arc::new(Silent)));
        let second = Arc::new(TierInstance::fake_with_model("second", "same-model", Arc::new(Silent)));
        let first_id = *first.id();
        let second_id = *second.id();
        registry.register(first).unwrap();
        registry.register(second).unwrap();
        assert_ne!(first_id, second_id);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn a_policy_calibrated_for_another_instance_is_reported() {
        use crate::pipeline::task::TierPolicy;
        let mut registry = TierRegistry::new();
        let running_instance = fake("rules");
        let running = *running_instance.id();
        let other = *fake("other").id();
        registry.register(running_instance).unwrap();
        let mut policy = TierPolicy::new(0.9);
        let mut policies = TierPolicyMap::new();
        policy.calibrated_for = Some(running);
        policies.insert(running, [("route".to_string(), policy.clone())].into());
        assert!(registry.calibration_mismatches(&policies).is_empty());
        policy.calibrated_for = Some(other);
        policies.insert(running, [("route".to_string(), policy)].into());
        assert_eq!(
            registry.calibration_mismatches(&policies),
            vec![CalibrationMismatch {
                tier: tname("rules"),
                tier_id: running,
                task: "route".to_string(),
                calibrated_for: other,
                running,
            }],
        );
    }
}
