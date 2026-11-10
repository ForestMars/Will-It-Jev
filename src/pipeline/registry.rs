// src/pipeline/registry.rs
//
// Tier identity and execution order are separate concerns.
//
// * `TierId` is a stable name ("rules", "fasttext", "modernbert", ...). Policies,
//   traces, resolutions, and telemetry refer to tiers by name.
// * `TierRegistry` catalogs every available configured tier by TierId. It does
//   not impose execution order.
// * `CascadeGraph` owns task-specific ordering and dependencies. `TierDirectory`
//   projects graph order for execution/telemetry and retains optional legacy ordinals
//   that keeps the numbered telemetry columns (`tier1_outcome` ..
//   `tier4_outcome`, `decided_by_tier`) meaningful for the original four tiers.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use serde::Deserialize;

use super::tier::Tier;

/// Name recorded when a task takes its fallback instead of a tier's reading.
/// Reserved: no tier may use it.
pub const FALLBACK_TIER_NAME: &str = "fallback";

const MAX_TIER_NAME_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
#[serde(try_from = "String")]
pub struct TierId(String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierIdError(String);

impl fmt::Display for TierIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for TierIdError {}

impl TierId {
    /// A tier name is 1 to 32 characters: a lowercase ASCII letter followed by
    /// lowercase letters, digits, `_` or `-`.
    pub fn new(name: impl Into<String>) -> Result<Self, TierIdError> {
        let name = name.into();
        let mut chars = name.chars();
        let valid = !name.is_empty()
            && name.len() <= MAX_TIER_NAME_LEN
            && chars.next().is_some_and(|c| c.is_ascii_lowercase())
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
        if !valid {
            return Err(TierIdError(format!(
                "invalid tier name {name:?}: use 1 to {MAX_TIER_NAME_LEN} characters, starting with a \
                 lowercase letter, then lowercase letters, digits, '_' or '-'"
            )));
        }
        if name == FALLBACK_TIER_NAME {
            return Err(TierIdError(format!(
                "tier name {FALLBACK_TIER_NAME:?} is reserved for task fallbacks"
            )));
        }
        if name.strip_prefix("tier").is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())) {
            return Err(TierIdError(format!(
                "tier name {name:?} is reserved: `tierN` is the alias for legacy ordinal N"
            )));
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for TierId {
    type Error = TierIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl fmt::Display for TierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Shorthand for tests, where names are literals.
#[cfg(test)]
pub fn tid(name: &str) -> TierId {
    TierId::new(name).expect("test tier name must be valid")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateTier(TierId),
    DuplicateOrdinal(u8),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateTier(id) => write!(f, "tier {id:?} is registered twice"),
            Self::DuplicateOrdinal(n) => write!(f, "legacy ordinal {n} is assigned to two tiers"),
        }
    }
}

impl Error for RegistryError {}

#[derive(Debug, Clone)]
pub struct TierInfo {
    pub id: TierId,
    /// Zero-based position in the graph's deterministic topological order.
    pub position: usize,
    /// Legacy ordinal retained only for old telemetry projections and tierN aliases.
    pub legacy_ordinal: Option<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct TierDirectory {
    infos: Vec<TierInfo>,
}

impl TierDirectory {
    pub fn infos(&self) -> &[TierInfo] { &self.infos }
    pub fn position(&self, id: &TierId) -> Option<usize> {
        self.infos.iter().find(|info| &info.id == id).map(|info| info.position)
    }
    pub fn legacy_ordinal(&self, id: &TierId) -> Option<u8> {
        self.infos.iter().find(|info| &info.id == id).and_then(|info| info.legacy_ordinal)
    }
    pub fn id_for_ordinal(&self, ordinal: u8) -> Option<&TierId> {
        self.infos.iter().find(|info| info.legacy_ordinal == Some(ordinal)).map(|info| &info.id)
    }
}

/// All available configured tiers. Registration does not establish execution order.
#[derive(Default)]
pub struct TierRegistry {
    entries: BTreeMap<TierId, (Arc<dyn Tier>, Option<u8>)>,
}

impl TierRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn register(&mut self, tier: Arc<dyn Tier>, legacy_ordinal: Option<u8>) -> Result<(), RegistryError> {
        let id = tier.id().clone();
        if self.entries.contains_key(&id) {
            return Err(RegistryError::DuplicateTier(id));
        }
        if let Some(ordinal) = legacy_ordinal {
            if self.entries.values().any(|(_, existing)| *existing == Some(ordinal)) {
                return Err(RegistryError::DuplicateOrdinal(ordinal));
            }
        }
        self.entries.insert(id, (tier, legacy_ordinal));
        Ok(())
    }
    pub fn get(&self, id: &TierId) -> Option<Arc<dyn Tier>> {
        self.entries.get(id).map(|(tier, _)| Arc::clone(tier))
    }
    pub fn ids(&self) -> BTreeSet<TierId> {
        self.entries.keys().cloned().collect()
    }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    /// Construct the directory from graph order, never registry insertion order.
    pub fn directory(&self, order: &[TierId]) -> TierDirectory {
        TierDirectory {
            infos: order.iter().enumerate().filter_map(|(position, id)| {
                self.entries.get(id).map(|(_, legacy_ordinal)| TierInfo {
                    id: id.clone(), position, legacy_ordinal: *legacy_ordinal,
                })
            }).collect(),
        }
    }
    #[cfg(test)]
    pub fn len(&self) -> usize { self.entries.len() }
    #[cfg(test)]
    pub fn of(tiers: Vec<Arc<dyn Tier>>) -> Self {
        let mut registry = Self::new();
        for tier in tiers {
            registry.register(tier, None).expect("test tiers must have unique IDs");
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
    fn tier_names_are_validated() {
        for ok in ["rules", "fasttext", "modernbert", "harrier", "oryn-2", "a", "prefilter_v2"] {
            assert!(TierId::new(ok).is_ok(), "{ok} should be valid");
        }
        for bad in ["", "Rules", "2fast", "has space", "x.y", "fallback", "tier3", &"a".repeat(33)] {
            assert!(TierId::new(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn registry_is_an_available_tier_lookup_not_an_execution_sequence() {
        let mut registry = TierRegistry::new();
        registry.register(named("modernbert"), Some(3)).unwrap();
        registry.register(named("rules"), Some(1)).unwrap();
        assert_eq!(registry.len(), 2);
        assert!(registry.get(&tid("modernbert")).is_some());
        assert!(registry.get(&tid("rules")).is_some());
        let directory = registry.directory(&[tid("rules"), tid("modernbert")]);
        assert_eq!(directory.position(&tid("rules")), Some(0));
        assert_eq!(directory.position(&tid("modernbert")), Some(1));
        assert_eq!(directory.legacy_ordinal(&tid("modernbert")), Some(3));
    }

    #[test]
    fn duplicates_and_duplicate_legacy_ordinals_are_rejected() {
        let mut registry = TierRegistry::new();
        registry.register(named("rules"), Some(1)).unwrap();
        assert_eq!(registry.register(named("rules"), None), Err(RegistryError::DuplicateTier(tid("rules"))));
        assert_eq!(registry.register(named("other"), Some(1)), Err(RegistryError::DuplicateOrdinal(1)));
        assert_eq!(registry.len(), 1);
    }
}
