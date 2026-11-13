// src/pipeline/registry.rs
//
// Lookup of instantiated tiers, and the projection used by telemetry.
//
// * `TierName`, `TierId` and `Slot` (see `identity.rs`) are not defined here.
// * `TierInstance` (see `tier.rs`) keeps everything it was built from.
// * `TierRegistry` finds every available instance by its user-authored name. It
//   imposes no execution order and stores nothing about an instance that the
//   instance does not already carry.
// * `CascadeGraph` owns task-specific ordering. `CascadeRunner` turns graph order
//   into slots. `TierDirectory` projects slots and ids for telemetry.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::identity::{Slot, TierId, TierName};
use super::task::{TaskId, TierPolicyMap};
use super::tier::TierInstance;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateName(TierName),
    /// Two names for instances with identical behavior-determining content. They
    /// are one instance: give them a different model, config or computation.
    DuplicateId { id: TierId, existing: TierName, duplicate: TierName },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "tier {name} is registered twice"),
            Self::DuplicateId { id, existing, duplicate } => write!(
                f,
                "tiers {existing} and {duplicate} have the same id {id}: identical role, computation, \
                 model and rule file make them one instance, so declare what differs (for example `model`)"
            ),
        }
    }
}

impl Error for RegistryError {}

/// A tier's place in the cascade, as telemetry sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierInfo {
    pub name: TierName,
    pub id: TierId,
    pub slot: Slot,
}

/// Slot and id lookup by name, built from the compiled cascade.
#[derive(Debug, Clone, Default)]
pub struct TierDirectory {
    infos: Vec<TierInfo>,
}

impl TierDirectory {
    #[allow(dead_code)]
    pub fn infos(&self) -> &[TierInfo] { &self.infos }
    pub fn info(&self, name: &TierName) -> Option<&TierInfo> {
        self.infos.iter().find(|info| &info.name == name)
    }
    pub fn slot(&self, name: &TierName) -> Option<Slot> { self.info(name).map(|info| info.slot) }
}

/// A policy that records the instance it was calibrated for, and the instance
/// now running under that name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationMismatch {
    pub tier: TierName,
    pub task: TaskId,
    pub calibrated_for: TierId,
    pub running: TierId,
}

/// Every available instance, looked up by name. Registration does not establish
/// execution order.
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
        if let Some(existing) = self.entries.values().find(|existing| existing.id() == tier.id()) {
            return Err(RegistryError::DuplicateId {
                id: *tier.id(), existing: existing.name().clone(), duplicate: tier.name().clone(),
            });
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

    /// Policies that name the instance they were calibrated for, where the
    /// instance now running under that name is a different one.
    pub fn calibration_mismatches(&self, policies: &TierPolicyMap) -> Vec<CalibrationMismatch> {
        let mut mismatches = Vec::new();
        for (name, by_task) in policies {
            let Some(instance) = self.entries.get(name) else { continue };
            for (task, policy) in by_task {
                if let Some(calibrated_for) = policy.calibrated_for {
                    if calibrated_for != *instance.id() {
                        mismatches.push(CalibrationMismatch {
                            tier: name.clone(), task: task.clone(), calibrated_for, running: *instance.id(),
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
    /// Construct the directory from graph order, never registry insertion order.
    pub fn from_order(registry: &TierRegistry, order: &[TierName]) -> Self {
        Self {
            infos: order.iter().enumerate().filter_map(|(index, name)| {
                let tier = registry.entries.get(name)?;
                Some(TierInfo {
                    name: name.clone(), id: *tier.id(),
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
    fn registry_is_a_lookup_not_an_execution_sequence() {
        let mut registry = TierRegistry::new();
        registry.register(fake("modernbert")).unwrap();
        registry.register(fake("rules")).unwrap();
        assert_eq!(registry.len(), 2);
        let directory = TierDirectory::from_order(&registry, &[tname("rules"), tname("modernbert")]);
        assert_eq!(directory.slot(&tname("rules")).map(Slot::index), Some(0));
        assert_eq!(directory.slot(&tname("modernbert")).map(Slot::index), Some(1));
        assert_eq!(directory.info(&tname("rules")).map(|info| info.id), registry.get(&tname("rules")).map(|tier| *tier.id()));
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut registry = TierRegistry::new();
        registry.register(fake("rules")).unwrap();
        assert_eq!(registry.register(fake("rules")), Err(RegistryError::DuplicateName(tname("rules"))));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn identical_content_under_two_names_is_one_instance() {
        let mut registry = TierRegistry::new();
        let first = Arc::new(TierInstance::fake_with_model("first", "same-model", Arc::new(Silent)));
        let second = Arc::new(TierInstance::fake_with_model("second", "same-model", Arc::new(Silent)));
        let id = *first.id();
        registry.register(first).unwrap();
        assert_eq!(
            registry.register(second),
            Err(RegistryError::DuplicateId { id, existing: tname("first"), duplicate: tname("second") }),
        );
    }

    #[test]
    fn a_policy_calibrated_for_another_instance_is_reported() {
        use crate::pipeline::task::TierPolicy;
        let mut registry = TierRegistry::new();
        registry.register(fake("rules")).unwrap();
        let running = *registry.get(&tname("rules")).unwrap().id();
        let other = *fake("other").id();
        let mut policy = TierPolicy::new(0.9);
        let mut policies = TierPolicyMap::new();
        policy.calibrated_for = Some(running);
        policies.insert(tname("rules"), [(TaskId::route(), policy.clone())].into());
        assert!(registry.calibration_mismatches(&policies).is_empty());
        policy.calibrated_for = Some(other);
        policies.insert(tname("rules"), [(TaskId::route(), policy)].into());
        assert_eq!(
            registry.calibration_mismatches(&policies),
            vec![CalibrationMismatch { tier: tname("rules"), task: TaskId::route(), calibrated_for: other, running }],
        );
    }
}
