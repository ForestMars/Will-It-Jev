use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub router: RouterConfig,
}

#[derive(Debug, Default, Deserialize)]
pub struct RouterConfig {
    #[serde(default)]
    pub tier5: TierConfig,
    #[serde(default)]
    pub tier6: TierConfig,
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
