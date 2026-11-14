// src/pipeline/cascade.rs
//
// The single ordered execution cascade. Authored TierNames resolve against the
// unordered TierRegistry here; TierIds belong to this compiled cascade.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use super::registry::TierRegistry;
use super::tier::{TierId, TierInstance, TierName};

/// Positional address of a TierId in the configured cascade.
/// A slot is derived from the cascade and can change when its order changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Slot(u16);

impl Slot {
    pub fn from_index(index: usize) -> Option<Self> {
        u16::try_from(index).ok().map(Self)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeError {
    UnknownTier(TierName),
    DuplicateTier(TierName),
    TierIdCollision {
        id: TierId,
        first: TierName,
        second: TierName,
    },
}

impl fmt::Display for CascadeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTier(tier) => {
                write!(f, "pipeline cascade references unavailable tier {tier}")
            }
            Self::DuplicateTier(tier) => {
                write!(f, "pipeline cascade lists tier {tier} more than once")
            }
            Self::TierIdCollision { id, first, second } => {
                write!(f, "tiers {first} and {second} generated the same TierId {id}")
            }
        }
    }
}

impl Error for CascadeError {}

/// One ordered sequence shared by every task. A task participates at a tier
/// only when that tier has a policy for the task.
#[derive(Debug, Clone, Default)]
pub struct Cascade {
    order: Vec<TierId>,
    names_by_id: BTreeMap<TierId, TierName>,
    ids_by_name: BTreeMap<TierName, TierId>,
}

impl Cascade {
    /// Resolve the authored order against registered instances and assign the
    /// corresponding TierIds. Registry entries absent from this list remain
    /// available but are not executable by this cascade.
    pub fn from_names(
        names: &[TierName],
        registry: &TierRegistry,
    ) -> Result<Self, CascadeError> {
        let mut cascade = Self::default();

        for name in names {
            if cascade.ids_by_name.contains_key(name) {
                return Err(CascadeError::DuplicateTier(name.clone()));
            }

            let instance = registry
                .get(name)
                .ok_or_else(|| CascadeError::UnknownTier(name.clone()))?;
            let id = TierInstance::derive_id(
                instance.name(),
                instance.tier(),
                instance.params(),
                instance.artifact_hash(),
            );

            if let Some(existing) = cascade.names_by_id.get(&id) {
                if existing != name {
                    return Err(CascadeError::TierIdCollision {
                        id,
                        first: existing.clone(),
                        second: name.clone(),
                    });
                }
            }

            cascade.order.push(id);
            cascade.names_by_id.insert(id, name.clone());
            cascade.ids_by_name.insert(name.clone(), id);
        }

        Ok(cascade)
    }

    pub fn order(&self) -> &[TierId] {
        &self.order
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn tier_name(&self, id: &TierId) -> Option<&TierName> {
        self.names_by_id.get(id)
    }

    pub fn id_for_name(&self, name: &TierName) -> Option<TierId> {
        self.ids_by_name.get(name).copied()
    }

    pub fn slot(&self, id: &TierId) -> Option<Slot> {
        self.order
            .iter()
            .position(|candidate| candidate == id)
            .and_then(Slot::from_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::tier::tname;
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

    fn registry(names: &[&str]) -> TierRegistry {
        TierRegistry::of(
            names
                .iter()
                .map(|name| Arc::new(TierInstance::fake(name, Arc::new(Silent))))
                .collect(),
        )
    }

    fn name(value: &str) -> TierName {
        tname(value)
    }

    #[test]
    fn a_slot_is_an_index_and_nothing_more() {
        assert_eq!(Slot::from_index(3).map(Slot::index), Some(3));
        assert_eq!(Slot::from_index(70_000), None);
    }

    #[test]
    fn cascade_preserves_the_authored_order_and_ignores_unused_registry_entries() {
        let registry = registry(&["alpha", "zulu", "unused"]);
        let cascade = Cascade::from_names(&[name("zulu"), name("alpha")], &registry).unwrap();
        let zulu = cascade.id_for_name(&name("zulu")).unwrap();
        let alpha = cascade.id_for_name(&name("alpha")).unwrap();

        assert_eq!(cascade.order(), &[zulu, alpha]);
        assert_eq!(cascade.slot(&zulu).map(Slot::index), Some(0));
        assert_eq!(cascade.slot(&alpha).map(Slot::index), Some(1));
        assert_eq!(cascade.id_for_name(&name("unused")), None);
    }

    #[test]
    fn cascade_requires_available_unique_tier_names() {
        let registry = registry(&["a"]);
        assert!(matches!(
            Cascade::from_names(&[name("a"), name("missing")], &registry),
            Err(CascadeError::UnknownTier(_))
        ));
        assert!(matches!(
            Cascade::from_names(&[name("a"), name("a")], &registry),
            Err(CascadeError::DuplicateTier(_))
        ));
    }
}
