// src/pipeline/tier_types.rs
//
// Shared tier-domain value types used by configuration and execution:
//
//   TierName  authored configuration name. Task cascades and tier-policy keys
//             refer to tiers by name; the runtime resolves names through the registry.
//   TierId    stable generated identifier derived from behavior-determining instance
//             properties (see `TierInstance::derive_id`). Configuration can record
//             it in `calibrated_for`; policies, the registry, graph, and execution
//             traces also use it. A changed model therefore produces a changed id.
//   Slot      positional address in the compiled execution directory. It is derived
//             from graph order, changes when that order changes, and is not persisted.

use std::error::Error;
use std::fmt;

use serde::Deserialize;

/// Name recorded when a task takes its fallback instead of a tier's reading.
/// Reserved: no tier may use it.
pub const FALLBACK_TIER_NAME: &str = "fallback";

const MAX_TIER_NAME_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
#[serde(try_from = "String")]
pub struct TierName(String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierTypeError(String);

impl fmt::Display for TierTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for TierTypeError {}

impl TierName {
    /// A tier name is 1 to 32 characters: a lowercase ASCII letter followed by
    /// lowercase letters, digits, `_` or `-`.
    pub fn new(name: impl Into<String>) -> Result<Self, TierTypeError> {
        let name = name.into();
        let mut chars = name.chars();
        let valid = !name.is_empty()
            && name.len() <= MAX_TIER_NAME_LEN
            && chars.next().is_some_and(|c| c.is_ascii_lowercase())
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
        if !valid {
            return Err(TierTypeError(format!(
                "invalid tier name {name:?}: use 1 to {MAX_TIER_NAME_LEN} characters, starting with a \
                 lowercase letter, then lowercase letters, digits, '_' or '-'"
            )));
        }
        if name == FALLBACK_TIER_NAME {
            return Err(TierTypeError(format!(
                "tier name {FALLBACK_TIER_NAME:?} is reserved for task fallbacks"
            )));
        }
        if name.strip_prefix("tier").is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())) {
            return Err(TierTypeError(format!(
                "tier name {name:?} is reserved: `tierN` is the alias for a tier numbered in [legacy_tiers]"
            )));
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for TierName {
    type Error = TierTypeError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl fmt::Display for TierName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Shorthand for tests, where names are literals.
#[cfg(test)]
pub fn tname(name: &str) -> TierName {
    TierName::new(name).expect("test tier name must be valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tier_id_round_trips_through_hex() {
        let id = TierId::from_digest(&[0xab; 32]);
        assert_eq!(id.to_string(), "ab".repeat(16));
        assert_eq!(TierId::parse(&id.to_string()), Ok(id));
        assert!(TierId::parse("xyz").is_err());
        assert!(TierId::parse(&"g".repeat(32)).is_err());
    }

    #[test]
    fn a_slot_is_an_index_and_nothing_more() {
        assert_eq!(Slot::from_index(3).map(Slot::index), Some(3));
        assert_eq!(Slot::from_index(70_000), None);
    }

    #[test]
    fn tier_names_are_validated() {
        for ok in ["rules", "fasttext", "modernbert", "harrier", "oryn-2", "a", "prefilter_v2"] {
            assert!(TierName::new(ok).is_ok(), "{ok} should be valid");
        }
        for bad in ["", "Rules", "2fast", "has space", "x.y", "fallback", "tier3", &"a".repeat(33)] {
            assert!(TierName::new(bad).is_err(), "{bad:?} should be rejected");
        }
    }
}

/// Generated identity of a tier instance. Equal ids mean equal behavior-determining
/// content, whatever the instances are called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TierId([u8; 16]);

impl TierId {
    pub const LEN: usize = 16;

    pub fn from_digest(digest: &[u8]) -> Self {
        let mut bytes = [0u8; Self::LEN];
        bytes.copy_from_slice(&digest[..Self::LEN]);
        Self(bytes)
    }

    pub fn parse(hex: &str) -> Result<Self, TierTypeError> {
        let bad = || TierTypeError(format!("invalid tier id {hex:?}: expected {} hex characters", Self::LEN * 2));
        if hex.len() != Self::LEN * 2 || !hex.is_ascii() {
            return Err(bad());
        }
        let mut bytes = [0u8; Self::LEN];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for TierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for TierId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

/// Position of an instance in the compiled cascade.
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

