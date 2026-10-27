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
}

#[derive(Debug, Default, Deserialize)]
pub struct TierConfig {
	#[serde(default)]
	pub enabled: bool,
}
