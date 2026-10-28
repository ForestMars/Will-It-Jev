mod schema;

use std::{error::Error, fs, path::Path};

pub use schema::AppConfig;

pub fn load_from_path(path: impl AsRef<Path>) -> Result<AppConfig, Box<dyn Error>> {
	let contents = fs::read_to_string(path)?;
	parse(&contents)
}

fn parse(contents: &str) -> Result<AppConfig, Box<dyn Error>> {
	let config: AppConfig = toml::from_str(contents)?;
	if ![
		config.router.tier1.enabled,
		config.router.tier2.enabled,
		config.router.tier3.enabled,
		config.router.tier4.enabled,
		config.router.tier5.enabled,
		config.router.tier6.enabled,
	]
	.into_iter()
	.any(|enabled| enabled)
	{
		return Err("at least one router tier must be enabled".into());
	}
	Ok(config)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn omitted_tiers_are_disabled() {
		let config = parse(
			"[router.tier1]\nenabled = true\n[router.tier3]\nenabled = false\n",
		)
		.unwrap();

		assert!(config.router.tier1.enabled);
		assert!(!config.router.tier2.enabled);
		assert!(!config.router.tier3.enabled);
		assert!(!config.router.tier4.enabled);
		assert!(!config.router.tier5.enabled);
		assert!(!config.router.tier6.enabled);
	}

	#[test]
	fn rejects_config_with_no_enabled_tiers() {
		assert!(parse("[router.tier1]\nenabled = false\n").is_err());
	}
}
