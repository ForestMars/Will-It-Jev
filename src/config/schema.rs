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
    pub tier3: GrpcTierConfig,
    #[serde(default)]
    pub tier4: GrpcTierConfig,
    #[serde(default)]
    pub tier5: TierConfig,
    #[serde(default)]
    pub tier6: TierConfig,
}

#[derive(Debug, Default, Deserialize)]
pub struct TierConfig {
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct GrpcTierConfig {
    pub enabled: bool,
    pub endpoint: Option<String>,
    pub jev_threshold: f32,
    pub llm_threshold: f32,
    pub timeout_ms: u64,
}

impl Default for GrpcTierConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            jev_threshold: 0.95,
            llm_threshold: 0.70,
            timeout_ms: 500,
        }
    }
}
