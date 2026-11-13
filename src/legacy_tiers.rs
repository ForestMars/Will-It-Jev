// src/legacy_tiers.rs
//
// Compatibility with numbered tiers, kept at the boundary.
//
// Before tiers had names, the four original tiers were numbered 1 to 4. Those
// numbers still appear in places that outlive the rewrite: the numbered
// telemetry fields (`tier1_outcome` .. `tier4_outcome`, `cascade_decision_tier`,
// `resolving_tier`), `[router.tierN]` settings, the numbered environment
// variables and `tierN` aliases in `[tier_policy.*]` keys.
//
// This map is the only place a name and a legacy number meet. Tiers, instances,
// the registry, the cascade and the runner know nothing about it; only the code
// that reads those compatibility surfaces consults it.

use std::collections::BTreeMap;

use crate::pipeline::identity::TierName;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LegacyTierNumbers {
    by_name: BTreeMap<TierName, u8>,
}

impl LegacyTierNumbers {
    pub fn new(pairs: impl IntoIterator<Item = (TierName, u8)>) -> Result<Self, String> {
        let mut by_name = BTreeMap::new();
        for (name, ordinal) in pairs {
            if ordinal == 0 {
                return Err(format!("legacy tier number for {name} must be positive"));
            }
            if let Some((existing, _)) = by_name.iter().find(|(_, taken)| **taken == ordinal) {
                return Err(format!("legacy tier number {ordinal} is used by both {existing} and {name}"));
            }
            by_name.insert(name, ordinal);
        }
        Ok(Self { by_name })
    }

    /// The numbering of the original four-tier path, for configuration that
    /// declares no `[[tier]]` entries.
    pub fn original_four() -> Self {
        Self::new(
            [("rules", 1), ("fasttext", 2), ("modernbert", 3), ("harrier", 4)]
                .map(|(name, ordinal)| (TierName::new(name).expect("built-in tier names are valid"), ordinal)),
        )
        .expect("built-in numbering is valid")
    }

    pub fn ordinal_of(&self, name: &TierName) -> Option<u8> {
        self.by_name.get(name).copied()
    }

    pub fn name_of(&self, ordinal: u8) -> Option<&TierName> {
        self.by_name.iter().find(|(_, number)| **number == ordinal).map(|(name, _)| name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::identity::tname;

    #[test]
    fn a_name_and_a_number_look_each_other_up() {
        let map = LegacyTierNumbers::new([(tname("rules"), 1), (tname("modernbert"), 3)]).unwrap();
        assert_eq!(map.ordinal_of(&tname("modernbert")), Some(3));
        assert_eq!(map.ordinal_of(&tname("prefilter")), None);
        assert_eq!(map.name_of(1), Some(&tname("rules")));
        assert_eq!(map.name_of(2), None);
    }

    #[test]
    fn numbers_are_positive_and_unique() {
        assert!(LegacyTierNumbers::new([(tname("rules"), 0)]).is_err());
        assert!(LegacyTierNumbers::new([(tname("rules"), 1), (tname("other"), 1)]).is_err());
    }

    #[test]
    fn the_original_numbering_is_the_four_shipped_tiers() {
        let map = LegacyTierNumbers::original_four();
        assert_eq!(map.name_of(4), Some(&tname("harrier")));
        assert_eq!(map.ordinal_of(&tname("fasttext")), Some(2));
    }
}

