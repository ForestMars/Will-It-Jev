mod schema;

use std::{error::Error, fs, path::Path};

pub use schema::{AppConfig, RouterConfig};

pub fn load_from_path(path: impl AsRef<Path>) -> Result<AppConfig, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    parse(&contents)
}

fn parse(contents: &str) -> Result<AppConfig, Box<dyn Error>> {
    let config: AppConfig = toml::from_str(contents)?;

    // Whether any cascade tier is enabled is decided against the tier registry
    // ([[tier]] in tasks.toml), not against the numbered [router.tierN] blocks:
    // a tier defined only in the registry has no such block. main.rs rejects a
    // registry with no enabled tier.

    // Task confidence thresholds now live in config/tasks.toml under
    // [tier_policy.tierN.task]. router.toml contains only execution-tier
    // settings, so this module validates endpoints and timeout caps only.
    for (name, tier) in [
        ("tier3", &config.router.tier3),
        ("tier4", &config.router.tier4),
    ] {
        if !tier.enabled {
            continue;
        }

        if tier
            .endpoint
            .as_deref()
            .is_some_and(|endpoint| endpoint.trim().is_empty())
        {
            return Err(format!(
                "router.{name}.endpoint must be non-empty when configured"
            )
            .into());
        }

        if tier.timeout_ms == Some(0) {
            return Err(format!(
                "router.{name}.timeout_ms must be greater than zero"
            )
            .into());
        }
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
    fn parses_grpc_execution_settings_without_task_thresholds() {
        let config = parse(
            r#"
            [router.tier3]
            enabled = true
            endpoint = "http://127.0.0.1:50052"
            timeout_ms = 650

            [router.tier4]
            enabled = true
            endpoint = "http://127.0.0.1:50051"
            timeout_ms = 750
            "#,
        )
        .unwrap();

        assert_eq!(
            config.router.tier3.endpoint.as_deref(),
            Some("http://127.0.0.1:50052")
        );
        assert_eq!(config.router.tier3.timeout_ms, Some(650));
        assert_eq!(
            config.router.tier4.endpoint.as_deref(),
            Some("http://127.0.0.1:50051")
        );
        assert_eq!(config.router.tier4.timeout_ms, Some(750));
    }

    #[test]
    fn rejects_blank_configured_scorer_endpoint() {
        assert!(parse(
            "[router.tier3]\nenabled = true\nendpoint = \"   \"\n"
        )
        .is_err());
    }

    #[test]
    fn rejects_zero_configured_timeout() {
        assert!(parse(
            "[router.tier4]\nenabled = true\ntimeout_ms = 0\n"
        )
        .is_err());
    }

    #[test]
    fn enablement_is_not_judged_from_the_numbered_blocks_alone() {
        let config = parse("[router.tier1]\nenabled = false\n").unwrap();
        assert!(!config.router.tier1.enabled);
    }
}
