// src/main.rs

mod config;
mod legacy_tiers;
mod pipeline;
mod router;
mod tagging;
mod telemetry;

use std::{path::PathBuf, sync::Arc, time::Duration};

use crate::tagging::{ClickHouseRawTable, OvenV1Tagger, TagConfig};
use pipeline::tagger::{CascadeTagger, JointTagStore};
use legacy_tiers::LegacyTierNumbers;
use pipeline::cascade::Cascade;
use pipeline::task::TierPolicyMap;
use pipeline::tier::{hex, Params, Role, Tier, TierInstance, TierName};
use pipeline::TierRegistry;
use pipeline::CascadeRunner;
use router::dispatch::DownstreamDispatcher;
use router::service;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

fn default_fasttext_model_path() -> String {
    const CANDIDATES: [&str; 2] = ["models/tier2_small.bin", "models/tier2.bin"];
    for candidate in CANDIDATES {
        if std::path::Path::new(candidate).exists() {
            return candidate.to_string();
        }
    }
    "models/tier2.bin".to_string()
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

/// A tier is enabled by its own `enabled`, else by the `[router.tierN]` block of
/// the legacy ordinal it carries, else it is on.
fn tier_enabled(name: &TierName, params: &Params, router: &config::RouterConfig, legacy: &LegacyTierNumbers) -> bool {
    params.enabled
        .or_else(|| {
            legacy.ordinal_of(name)
                .and_then(|ordinal| router.legacy_tier(ordinal))
                .map(|tier| tier.enabled)
        })
        .unwrap_or(true)
}

/// Endpoint and per-call timeout for a gRPC tier. The original sidecars keep
/// their historical environment variables; other names use TIER_<NAME>_*.
fn grpc_settings(
    name: &TierName,
    params: &Params,
    router: &config::RouterConfig,
    numbers: &LegacyTierNumbers,
) -> Result<(String, Duration), Box<dyn std::error::Error>> {
    let ordinal = numbers.ordinal_of(name);
    let legacy = ordinal.and_then(|ordinal| router.legacy_tier(ordinal));
    let endpoint = params.endpoint.as_deref()
        .or_else(|| legacy.and_then(|tier| tier.endpoint.as_deref()));
    let timeout_ms = params.timeout_ms.or_else(|| legacy.and_then(|tier| tier.timeout_ms));
    let upper = name.as_str().to_ascii_uppercase().replace('-', "_");
    let (endpoint_env, endpoint_aliases, timeout_env): (String, &[&str], String) =
        match ordinal {
            Some(3) => ("TIER3_SCORER_ENDPOINT".into(), &["MODERNBERT_ENDPOINT"], "TIER3_MAX_CALL_MS".into()),
            Some(4) => ("TIER4_SCORER_ENDPOINT".into(), &["TIER4_HARRIER_ENDPOINT", "TIER3_ENDPOINT"], "TIER4_MAX_CALL_MS".into()),
            _ => (format!("TIER_{upper}_ENDPOINT"), &[], format!("TIER_{upper}_MAX_CALL_MS")),
        };
    Ok((
        configured_endpoint(&endpoint_env, endpoint_aliases, endpoint)?,
        configured_timeout(&timeout_env, timeout_ms)?,
    ))
}

/// Resolve deployment-specific values after parsing, without introducing a
/// declaration/domain type between Tier configuration and TierInstance.
fn resolve_instance_params(
    name: &TierName,
    tier: &Tier,
    params: &Params,
    router: &config::RouterConfig,
    legacy: &LegacyTierNumbers,
) -> Result<Params, Box<dyn std::error::Error>> {
    let mut resolved = params.clone();
    match tier.exex {
        Role::Rules => {
            if resolved.config.is_none() {
                resolved.config = Some(env_or("TIER1_CONFIG", "config/tier1.toml"));
            }
        }
        Role::FastText => {
            if resolved.model.is_none() {
                resolved.model = Some(
                    std::env::var("FASTTEXT_MODEL").unwrap_or_else(|_| default_fasttext_model_path()),
                );
            }
            if resolved.model.as_deref() == Some("models/tier2.bin")
                && !std::path::Path::new("models/tier2_small.bin").exists()
            {
                tracing::warn!(tier = %name, model_path = "models/tier2.bin", "using legacy fastText tier-2 model; prefer models/tier2_small.bin when available");
            }
        }
        Role::Encoder => {
            let (endpoint, timeout) = grpc_settings(name, &resolved, router, legacy)?;
            resolved.endpoint = Some(endpoint);
            resolved.timeout_ms = Some(u64::try_from(timeout.as_millis())?);
        }
    }
    Ok(resolved)
}

fn configured_endpoint(
    name: &str,
    legacy_names: &[&str],
    configured: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    for candidate in std::iter::once(name).chain(legacy_names.iter().copied()) {
        if let Ok(value) = std::env::var(candidate) {
            return Ok(value);
        }
    }
    configured
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "missing endpoint: set {name}, or configure `endpoint` on the tier ([[tier]] in config/tasks.toml or [router.tierN] in config/router.toml)"
                ),
            )
            .into()
        })
}

fn configured_timeout(
    name: &str,
    configured_ms: Option<u64>,
) -> Result<Duration, Box<dyn std::error::Error>> {
    let millis = match std::env::var(name) {
        Ok(value) => value.parse::<u64>()?,
        Err(std::env::VarError::NotPresent) => configured_ms.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "missing timeout: set {name}, or configure `timeout_ms` on the tier ([[tier]] in config/tasks.toml or [router.tierN] in config/router.toml)"
                ),
            )
        })?,
        Err(error) => return Err(error.into()),
    };
    if millis == 0 {
        return Err(format!("{name} must be greater than zero").into());
    }
    Ok(Duration::from_millis(millis))
}

/// Reads KEY=VALUE pairs from a dotenv-style file such as the `.auth` file devo writes.
/// A missing or unreadable file yields no entries.
fn read_env_file(path: &str) -> std::collections::HashMap<String, String> {
    let mut entries = std::collections::HashMap::new();
    let Ok(contents) = std::fs::read_to_string(path) else {
        return entries;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|inner| inner.strip_suffix('"'))
            .or_else(|| {
                value
                    .strip_prefix('\'')
                    .and_then(|inner| inner.strip_suffix('\''))
            })
            .unwrap_or(value);
        entries.insert(key.trim().to_string(), value.to_string());
    }
    entries
}

/// ClickHouse credentials: the process environment wins, and `.auth` fills in any
/// name the environment leaves unset.
fn clickhouse_credentials() -> (Option<String>, Option<String>) {
    let file = read_env_file(".auth");
    let pick = |name: &str| std::env::var(name).ok().or_else(|| file.get(name).cloned());
    (pick("CLICKHOUSE_USER"), pick("CLICKHOUSE_PASSWORD"))
}

/// Headers the ClickHouse HTTP interface accepts for authentication. Empty when no
/// credentials are configured.
fn clickhouse_auth_headers(
    user: Option<&str>,
    password: Option<&str>,
) -> Result<HeaderMap, Box<dyn std::error::Error>> {
    let mut headers = HeaderMap::new();
    if user.is_some() || password.is_some() {
        headers.insert(
            HeaderName::from_static("x-clickhouse-user"),
            HeaderValue::from_str(user.unwrap_or("default"))?,
        );
    }
    if let Some(password) = password {
        let mut value = HeaderValue::from_str(password)?;
        value.set_sensitive(true);
        headers.insert(HeaderName::from_static("x-clickhouse-key"), value);
    }
    Ok(headers)
}

/// Returns `base` with the credentials embedded as URL userinfo, which reqwest
/// sends as HTTP Basic auth. Unchanged when no credentials are configured.
fn clickhouse_url_with_auth(
    base: &str,
    user: Option<&str>,
    password: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut url = reqwest::Url::parse(base)?;
    if user.is_some() || password.is_some() {
        url.set_username(user.unwrap_or("default"))
            .map_err(|_| "ClickHouse URL cannot carry credentials")?;
        url.set_password(password)
            .map_err(|_| "ClickHouse URL cannot carry credentials")?;
    }
    Ok(url.as_str().trim_end_matches('/').to_string())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                if let Err(error) = result {
                    tracing::error!(%error, "failed to listen for Ctrl-C");
                }
            }
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::error!(%error, "failed to listen for Ctrl-C");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let verbose = std::env::args()
        .skip(1)
        .any(|argument| argument == "-v" || argument == "--verbose");
    let filter = match std::env::var("RUST_LOG") {
        Ok(filter) => EnvFilter::try_new(filter)?,
        Err(std::env::VarError::NotPresent) => {
            EnvFilter::new(if verbose { "debug" } else { "info" })
        }
        Err(error) => return Err(error.into()),
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(false).compact())
        .init();

    tracing::info!("Cascade Router initialized successfully");

    let config_path = env_or("ROUTER_CONFIG", "config/router.toml");
    let config = config::load_from_path(&config_path)?;
    tracing::info!(config_path, "loaded router configuration");
    tracing::info!(
        tier1_enabled = config.router.tier1.enabled,
        tier2_enabled = config.router.tier2.enabled,
        tier3_enabled = config.router.tier3.enabled,
        tier4_enabled = config.router.tier4.enabled,
        tier5_enabled = config.router.tier5.enabled,
        tier6_enabled = config.router.tier6.enabled,
        "configured tier status"
    );
    tracing::debug!(verbose, "startup verbosity configured");

    let pipeline_config =
        pipeline::config::load_or_default(env_or("TASKS_CONFIG", "config/tasks.toml"))?;
    // ---- Build the complete available-tier registry -----------------------
    // [[tier]] supplies instances; its array order has no execution meaning.
    let mut registry = TierRegistry::new();
    for (name, tier, params) in &pipeline_config.tiers {
        if !tier_enabled(name, params, &config.router, &pipeline_config.legacy) {
            tracing::info!(tier = %name, "tier disabled by configuration");
            continue;
        }
        let resolved = resolve_instance_params(name, tier, params, &config.router, &pipeline_config.legacy)?;
        let instance = TierInstance::build(name.clone(), tier.clone(), resolved)?;
        tracing::info!(
            tier = %instance.name(), exex = instance.tier().exex.as_str(),
            model_format = ?instance.params().model_format.map(|format| format.as_str()), model = %instance.model_label(),
            "tier instance built"
        );
        registry.register(Arc::new(instance))?;
    }
    if registry.is_empty() {
        return Err("no cascade tier is enabled; enable at least one [[tier]]".into());
    }

    // Resolve the one authored cascade against the complete available registry.
    let configured_tasks = pipeline_config.profiles();
    let enabled_cascade_names: Vec<_> = pipeline_config.cascade.iter().filter_map(|name| {
        if registry.get(name).is_some() {
            Some(name.clone())
        } else {
            tracing::info!(tier = %name, "disabled tier omitted from pipeline cascade");
            None
        }
    }).collect();
    let execution_cascade = Cascade::from_names(&enabled_cascade_names, &registry)?;
    let runtime_tasks = configured_tasks;

    // Policies are authored by TierName and become TierId-keyed only after the shared Cascade has resolved the corresponding node.
    let mut tier_policies = TierPolicyMap::new();
    for (name, by_task) in &pipeline_config.tier_policies {
        if let Some(id) = execution_cascade.id_for_name(name) {
            tier_policies.insert(id, by_task.clone());
        } else {
            tracing::info!(tier = %name, "policies for a disabled or unused tier are inactive");
        }
    }
    if execution_cascade.order().is_empty() {
        return Err("no task cascade contains an enabled tier".into());
    }
    tracing::info!(
        order = ?execution_cascade.order().iter().map(ToString::to_string).collect::<Vec<_>>(),
        "pipeline cascade order"
    );

    let artifact_hash = |exex: Role| {
        execution_cascade.order().iter()
            .filter_map(|id| execution_cascade.tier_name(id).and_then(|name| registry.get(name)))
            .find(|tier| tier.tier().exex == exex)
            .and_then(|tier| tier.artifact_hash().map(|hash| hex(hash)))
            .unwrap_or_else(|| "disabled".to_string())
    };
    let hashes = telemetry::Hashes {
        tier1_toml: artifact_hash(Role::Rules),
        tier2_bin: artifact_hash(Role::FastText),
    };

    for (tier_id, by_task) in &tier_policies {
        let Some(name) = execution_cascade.tier_name(tier_id) else { continue };
        for (task, policy) in by_task {
            if let Some(calibrated_for) = policy.calibrated_for {
                if calibrated_for != *tier_id {
                    tracing::warn!(
                        tier = %name, tier_id = %tier_id, task = %task,
                        calibrated_for = %calibrated_for, running = %tier_id,
                        "tier_policy was calibrated for a different instance than the one now running under this name"
                    );
                }
            }
        }
    }

    let route_profile = runtime_tasks.iter().find(|task| task.name == "route").cloned()
        .ok_or("tasks.toml: missing [task.route] definition")?;
    if !pipeline_config.has_policy("route") {
        return Err("tasks.toml: route task has no [tier_policy.<tier>.route] entries".into());
    }

    let tag_in_request = pipeline_config.settings.tag_in_request
        || pipeline_config.settings.tag_via_cascade;
    let tag_profile = if tag_in_request {
        let profile = runtime_tasks.iter().find(|task| task.name == "tag").cloned().ok_or(
            "tasks.toml: tag_in_request = true but there is no [task.tag] profile",
        )?;
        if !pipeline_config.has_policy("tag") {
            return Err("tasks.toml: tag_in_request = true but no [tier_policy.<tier>.tag] policy is configured".into());
        }
        let has_executable_tag_head = execution_cascade.order().iter().any(|id| {
            execution_cascade.tier_name(id).and_then(|name| registry.get(name)).is_some_and(|instance| {
                instance.tier().exex == Role::Encoder
                    && tier_policies.get(id).is_some_and(|by_task| by_task.contains_key("tag"))
            })
        });
        if !has_executable_tag_head {
            return Err("tasks.toml: joint tagging requires a tag policy on an enabled encoder tier".into());
        }
        Some(profile)
    } else {
        None
    };

    let tier_policies = Arc::new(tier_policies);
    let cascade = Arc::new(CascadeRunner::new(registry, execution_cascade)?);
    let legacy = Arc::new(pipeline_config.legacy.clone());

    let downstream_timeout =
        Duration::from_millis(env_or("DOWNSTREAM_TIMEOUT_MS", "120000").parse()?);
    let dispatcher = DownstreamDispatcher::new_with_enabled(
        env_or(
            "TIER5_ENDPOINT",
            &env_or("TIER4_ENDPOINT", "http://127.0.0.1:11435/api/decide"),
        ),
        env_or("OLLAMA_GENERATE_URL", "http://127.0.0.1:11434/api/generate"),
        env_or("GEMMA_MODEL", "gemma3:latest"),
        downstream_timeout,
        3,
        Duration::from_secs(30),
        Duration::from_secs(30),
        config.router.tier5.enabled,
        config.router.tier6.enabled,
    )?;

    let clickhouse_url = env_or("CLICKHOUSE_URL", "http://127.0.0.1:8123");
    let route_clickhouse_url = clickhouse_url.clone();
    let (clickhouse_user, clickhouse_password) = clickhouse_credentials();
    let clickhouse_headers =
        clickhouse_auth_headers(clickhouse_user.as_deref(), clickhouse_password.as_deref())?;
    let tag_clickhouse_url = clickhouse_url_with_auth(
        &clickhouse_url,
        clickhouse_user.as_deref(),
        clickhouse_password.as_deref(),
    )?;
    tracing::info!(
        clickhouse_user = clickhouse_user.as_deref().unwrap_or("none"),
        clickhouse_password_set = clickhouse_password.is_some(),
        "resolved ClickHouse credentials"
    );

    let (capture_tx, mut capture_rx) =
        tokio::sync::mpsc::channel::<telemetry::RouteRecord>(1024);
    let capture_task = tokio::spawn(async move {
        tokio::fs::create_dir_all("logs").await?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/routing_capture.jsonl")
            .await?;
        let clickhouse_client = reqwest::Client::builder()
            .default_headers(clickhouse_headers)
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(std::io::Error::other)?;
        let mut flush_interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            tokio::select! {
                recv = capture_rx.recv() => {
                    let Some(record) = recv else { break };
                    let json = serde_json::to_string(&record)?;
                    file.write_all(format!("{json}\n").as_bytes()).await?;
                    if let Err(error) = tagging::insert_clickhouse_raw_line(
                        &clickhouse_client,
                        &route_clickhouse_url,
                        ClickHouseRawTable::RouteRecords,
                        &json,
                    ).await {
                        tracing::warn!(
                            request_id = %record.request_id,
                            error = %error,
                            "failed to insert route record into ClickHouse; retained in local JSONL spool"
                        );
                    }
                }
                _ = flush_interval.tick() => file.flush().await?,
            }
        }
        file.flush().await?;
        Ok::<(), std::io::Error>(())
    });

    let ollama_tagger = OvenV1Tagger::new(
        env_or("TAGGER_OLLAMA_URL", "http://127.0.0.1:11434"),
        env_or("TAGGER_MODEL", "qwen3:1.7b"),
    );
    let tag_sample_prob: f32 = env_or("TAG_SAMPLE_PROB", "1.0").parse()?;
    if !tag_sample_prob.is_finite() || !(0.0..=1.0).contains(&tag_sample_prob) {
        return Err("TAG_SAMPLE_PROB must be between 0 and 1".into());
    }
    let tag_budget = Duration::from_millis(pipeline_config.settings.tag_budget_ms);
    let joint_tags = JointTagStore::default();
    let tag_config = TagConfig {
        base_path: PathBuf::from("logs/task_tags.jsonl"),
        clickhouse_url: Some(tag_clickhouse_url),
        queue_capacity: 1024,
        concurrency: 4,
        call_timeout: Duration::from_secs(5),
        max_age: Duration::from_secs(60),
        sample_prob: tag_sample_prob,
    };

    let (tag_queue, tag_handle) = if tag_in_request {
        let taxonomy_version = tag_profile.as_ref()
            .and_then(|profile| profile.taxonomy_version.clone())
            .unwrap_or_else(|| "unknown".into());
        let output_kind = tag_profile.as_ref()
            .map(|profile| profile.output_kind.as_str())
            .unwrap_or("multiclass");
        let tagger = Arc::new(
            CascadeTagger::new(
                joint_tags.clone(),
                tag_budget,
                taxonomy_version,
                output_kind,
                ollama_tagger,
            )
            .with_legacy_numbers(Arc::clone(&legacy)),
        );
        tracing::info!("sampled task tagging joins the route request's single cascade");
        tagging::spawn(tagger, tag_config)
    } else {
        tagging::spawn(Arc::new(ollama_tagger), tag_config)
    };

    let app = service::app(service::AppDependencies {
        cascade,
        task_profiles: runtime_tasks,
        route_profile,
        tier_policies,
        dispatcher,
        hashes,
        capture_tx: capture_tx.clone(),
        tag_queue: tag_queue.clone(),
        tag_profile,
        joint_tags,
        tag_in_request,
        tag_sample_prob,
        joint_tag_ttl: downstream_timeout + tag_budget + Duration::from_secs(90),
        budget: Duration::from_millis(env_or("ROUTER_BUDGET_MS", "600").parse()?),
        legacy,
    });
    let bind_address = env_or("ROUTER_BIND", "127.0.0.1:8080");
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;
    tracing::info!(address = %listener.local_addr()?, "gateway listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    drop(capture_tx);
    drop(tag_queue);
    let (capture_result, tag_result) = tokio::join!(capture_task, tag_handle);
    capture_result??;
    tag_result??;
    Ok(())
}
