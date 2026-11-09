use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub router: RouterConfig,
}

#[derive(Debug, Default, Deserialize)]
pub struct RouterConfig {
    #[serde(default)]
    pub tier1: TierConfig,
    #[serde(default)]
    pub tier2: TierConfig,
    #[serde(default)]
    pub tier3: TierConfig,
    #[serde(default)]
    pub tier4: TierConfig,
    #[serde(default)]
    pub tier5: TierConfig,
    #[serde(default)]
    pub tier6: TierConfig,
}

impl RouterConfig {
    /// Settings block for a tier that still carries its original 1..=6 number.
    pub fn legacy_tier(&self, ordinal: u8) -> Option<&TierConfig> {
        match ordinal {
            1 => Some(&self.tier1),
            2 => Some(&self.tier2),
            3 => Some(&self.tier3),
            4 => Some(&self.tier4),
            5 => Some(&self.tier5),
            6 => Some(&self.tier6),
            _ => None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct TierConfig {
    #[serde(default)]
    pub enabled: bool,
    /// gRPC scorer endpoint for tiers backed by the shared scoring protocol.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Per-call timeout cap in milliseconds for gRPC scorer tiers.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}
