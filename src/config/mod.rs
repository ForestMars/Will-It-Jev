mod schema;

use std::{error::Error, fs, path::Path};

pub use schema::{AppConfig, RouterConfig};

pub fn load_from_path(path: impl AsRef<Path>) -> Result<AppConfig, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    parse(&contents)
}

fn parse(contents: &str) -> Result<AppConfig, Box<dyn Error>> {
    let config: AppConfig = toml::from_str(contents)?;

    let any_tier_enabled = [
        config.router.tier5.enabled,
        config.router.tier6.enabled,
    ]
    .into_iter()
    .any(|enabled| enabled);

    if !any_tier_enabled {
        return Err("at least one router tier must be enabled".into());
    }



    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn downstream_tiers_default_to_disabled() {
        let config = parse("[router.tier5]\nenabled = true\n").unwrap();
        assert!(config.router.tier5.enabled);
        assert!(!config.router.tier6.enabled);
    }
    #[test]
    fn rejects_configuration_with_no_enabled_downstream_tier() {
        assert!(parse("[router.tier5]\nenabled = false\n").is_err());
    }
}
