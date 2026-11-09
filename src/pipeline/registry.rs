// src/pipeline/registry.rs
//
// Tier identity and execution order are separate concerns.
//
// * `TierId` is a stable name ("rules", "fasttext", "modernbert", ...). Policies,
//   traces, resolutions, and telemetry refer to tiers by name.
// * `TierRegistry` is the ordered list of tiers. Execution order is the order
//   in which tiers are registered, which `main` takes from the `[[tier]]`
//   entries in `tasks.toml`. Nothing derives order from a number.
// * `TierDirectory` is the read-only view of that registry that the rest of the
//   router uses to look up a tier's position, and the optional legacy ordinal
//   that keeps the numbered telemetry columns (`tier1_outcome` ..
//   `tier4_outcome`, `decided_by_tier`) meaningful for the original four tiers.

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
    /// Definition this tier is an instance of (`rules`, `fasttext`, `grpc`, ...).
    pub kind: &'static str,
    /// Zero-based execution position.
    pub position: usize,
    /// The 1..=4 number the original tiers carried, kept only so numbered
    /// telemetry columns and `tierN` config keys continue to resolve.
    pub legacy_ordinal: Option<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct TierDirectory {
    infos: Vec<TierInfo>,
}

impl TierDirectory {
    pub fn infos(&self) -> &[TierInfo] {
        &self.infos
    }

    pub fn position(&self, id: &TierId) -> Option<usize> {
        self.infos.iter().find(|info| &info.id == id).map(|info| info.position)
    }

    pub fn kind(&self, id: &TierId) -> Option<&'static str> {
        self.infos.iter().find(|info| &info.id == id).map(|info| info.kind)
    }

    pub fn legacy_ordinal(&self, id: &TierId) -> Option<u8> {
        self.infos
            .iter()
            .find(|info| &info.id == id)
            .and_then(|info| info.legacy_ordinal)
    }

    pub fn id_for_ordinal(&self, ordinal: u8) -> Option<&TierId> {
        self.infos
            .iter()
            .find(|info| info.legacy_ordinal == Some(ordinal))
            .map(|info| &info.id)
    }
}

/// Ordered collection of execution tiers.
#[derive(Default)]
pub struct TierRegistry {
    entries: Vec<(Arc<dyn Tier>, Option<u8>)>,
}

impl TierRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a tier. It runs after every tier registered before it.
    pub fn register(
        &mut self,
        tier: Arc<dyn Tier>,
        legacy_ordinal: Option<u8>,
    ) -> Result<(), RegistryError> {
        if self.entries.iter().any(|(existing, _)| existing.id() == tier.id()) {
            return Err(RegistryError::DuplicateTier(tier.id().clone()));
        }
        if let Some(ordinal) = legacy_ordinal {
            if self.entries.iter().any(|(_, existing)| *existing == Some(ordinal)) {
                return Err(RegistryError::DuplicateOrdinal(ordinal));
            }
        }
        self.entries.push((tier, legacy_ordinal));
        Ok(())
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn directory(&self) -> TierDirectory {
        TierDirectory {
            infos: self
                .entries
                .iter()
                .enumerate()
                .map(|(position, (tier, legacy_ordinal))| TierInfo {
                    id: tier.id().clone(),
                    kind: tier.kind(),
                    position,
                    legacy_ordinal: *legacy_ordinal,
                })
                .collect(),
        }
    }

    pub(crate) fn into_tiers(self) -> Vec<Arc<dyn Tier>> {
        self.entries.into_iter().map(|(tier, _)| tier).collect()
    }

    /// Test helper: register tiers in the given order, no legacy ordinals.
    #[cfg(test)]
    pub fn of(tiers: Vec<Arc<dyn Tier>>) -> Self {
        let mut registry = Self::new();
        for tier in tiers {
            registry.register(tier, None).expect("test tiers must be distinct");
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

    fn named(name: &str) -> Arc<dyn Tier> {
        Arc::new(Named(tid(name)))
    }

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
    fn execution_order_is_registration_order_not_name_order() {
        let mut registry = TierRegistry::new();
        for name in ["zeta", "alpha", "mid"] {
            registry.register(named(name), None).unwrap();
        }
        let order: Vec<_> = registry.directory().infos().iter().map(|i| i.id.to_string()).collect();
        assert_eq!(order, ["zeta", "alpha", "mid"]);
        assert_eq!(registry.directory().position(&tid("alpha")), Some(1));
        let tiers = registry.into_tiers();
        assert_eq!(tiers[0].id().as_str(), "zeta");
    }

    #[test]
    fn duplicates_are_rejected() {
        let mut registry = TierRegistry::new();
        registry.register(named("rules"), Some(1)).unwrap();
        assert_eq!(
            registry.register(named("rules"), None),
            Err(RegistryError::DuplicateTier(tid("rules")))
        );
        assert_eq!(
            registry.register(named("other"), Some(1)),
            Err(RegistryError::DuplicateOrdinal(1))
        );
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn directory_maps_between_names_and_legacy_ordinals() {
        let mut registry = TierRegistry::new();
        registry.register(named("rules"), Some(1)).unwrap();
        registry.register(named("prefilter"), None).unwrap();
        registry.register(named("modernbert"), Some(3)).unwrap();
        let directory = registry.directory();
        assert_eq!(directory.legacy_ordinal(&tid("modernbert")), Some(3));
        assert_eq!(directory.legacy_ordinal(&tid("prefilter")), None);
        assert_eq!(directory.id_for_ordinal(3), Some(&tid("modernbert")));
        assert_eq!(directory.id_for_ordinal(2), None);
        assert_eq!(directory.position(&tid("modernbert")), Some(2));
        assert_eq!(directory.position(&tid("missing")), None);
    }

    #[test]
    fn several_instances_can_share_one_definition() {
        let mut registry = TierRegistry::new();
        registry.register(named("encoder-fast"), None).unwrap();
        registry.register(named("encoder-accurate"), None).unwrap();
        let directory = registry.directory();
        let fast = directory.kind(&tid("encoder-fast")).unwrap();
        let accurate = directory.kind(&tid("encoder-accurate")).unwrap();
        assert_eq!(fast, accurate, "both instances are the same definition");
        assert_ne!(tid("encoder-fast"), tid("encoder-accurate"), "but distinct instances");
        assert_eq!(directory.position(&tid("encoder-accurate")), Some(1));
    }
}
