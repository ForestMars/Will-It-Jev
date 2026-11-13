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
use pipeline::tier::{hex, RequestShape, Role, TierDecl, TierInstance};
use pipeline::{CascadeRunner, TaskId, TierRegistry};
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
fn tier_enabled(decl: &TierDecl, router: &config::RouterConfig, legacy: &LegacyTierNumbers) -> bool {
    decl.params.enabled
        .or_else(|| {
            legacy.ordinal_of(&decl.name)
                .and_then(|ordinal| router.legacy_tier(ordinal))
                .map(|tier| tier.enabled)
        })
        .unwrap_or(true)
}

/// Endpoint and per-call timeout for a gRPC tier. The original two sidecar
/// tiers keep their historical environment variables; every other tier uses
/// `TIER_<NAME>_ENDPOINT` and `TIER_<NAME>_MAX_CALL_MS`.
fn grpc_settings(
    decl: &TierDecl,
    router: &config::RouterConfig,
    numbers: &LegacyTierNumbers,
) -> Result<(String, Duration), Box<dyn std::error::Error>> {
    let ordinal = numbers.ordinal_of(&decl.name);
    let legacy = ordinal.and_then(|ordinal| router.legacy_tier(ordinal));
    let endpoint = decl.params
        .endpoint
        .as_deref()
        .or_else(|| legacy.and_then(|tier| tier.endpoint.as_deref()));
    let timeout_ms = decl.params.timeout_ms.or_else(|| legacy.and_then(|tier| tier.timeout_ms));
    let upper = decl.name.as_str().to_ascii_uppercase().replace('-', "_");
    let (endpoint_env, endpoint_aliases, timeout_env): (String, &[&str], String) =
        match ordinal {
            Some(3) => (
                "TIER3_SCORER_ENDPOINT".into(),
                &["MODERNBERT_ENDPOINT"],
                "TIER3_MAX_CALL_MS".into(),
            ),
            Some(4) => (
                "TIER4_SCORER_ENDPOINT".into(),
                &["TIER4_HARRIER_ENDPOINT", "TIER3_ENDPOINT"],
                "TIER4_MAX_CALL_MS".into(),
            ),
            _ => (
                format!("TIER_{upper}_ENDPOINT"),
                &[],
                format!("TIER_{upper}_MAX_CALL_MS"),
            ),
        };
    Ok((
        configured_endpoint(&endpoint_env, endpoint_aliases, endpoint)?,
        configured_timeout(&timeout_env, timeout_ms)?,
    ))
}

/// Fill in everything an instance needs that its declaration does not state, so
/// the instance is built from, and keeps, its final parameters.
///
/// * rules `config` and fasttext `model`: the instance's own value wins. The
///   process-wide `TIER1_CONFIG` / `FASTTEXT_MODEL` only supply a value for an
///   instance that declares none, so several instances never collapse into one.
/// * encoder `endpoint` and `timeout_ms`: per-instance environment variables
///   override the declaration, which overrides the legacy `[router.tierN]` block.
fn resolve_decl(
    decl: &TierDecl,
    router: &config::RouterConfig,
    legacy: &LegacyTierNumbers,
) -> Result<TierDecl, Box<dyn std::error::Error>> {
    let mut decl = decl.clone();
    match decl.tier.role {
        Role::Rules => {
            if decl.params.config.is_none() {
                decl.params.config = Some(env_or("TIER1_CONFIG", "config/tier1.toml"));
            }
        }
        Role::FastText => {
            if decl.params.model.is_none() {
                decl.params.model = Some(
                    std::env::var("FASTTEXT_MODEL").unwrap_or_else(|_| default_fasttext_model_path()),
                );
            }
            if decl.params.model.as_deref() == Some("models/tier2.bin")
                && !std::path::Path::new("models/tier2_small.bin").exists()
            {
                tracing::warn!(tier = %decl.name, model_path = "models/tier2.bin", "using legacy fastText tier-2 model; prefer models/tier2_small.bin when available");
            }
        }
        Role::Encoder => {
            let (endpoint, timeout) = grpc_settings(&decl, router, legacy)?;
            decl.params.endpoint = Some(endpoint);
            decl.params.timeout_ms = Some(u64::try_from(timeout.as_millis())?);
        }
    }
    Ok(decl)
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
    let route_profile = pipeline_config
        .profile(&TaskId::route())
        .ok_or("tasks.toml: missing [task.route] definition")?;
    if !pipeline_config.has_policy(&TaskId::route()) {
        return Err("tasks.toml: route task has no [tier_policy.<tier>.route] entries".into());
    }
    let tier_policies = Arc::new(pipeline_config.tier_policies.clone());
    // tag_via_cascade is retained as a compatibility alias. Both settings
    // now mean that sampled tag tasks join the route request's single cascade.
    let tag_in_request = pipeline_config.settings.tag_in_request
        || pipeline_config.settings.tag_via_cascade;
    let tag_profile = if tag_in_request {
        let profile = pipeline_config.profile(&TaskId::tag()).ok_or(
            "tasks.toml: tag_in_request = true but there is no [task.tag] profile",
        )?;
        if !pipeline_config.has_policy(&TaskId::tag()) {
            return Err("tasks.toml: tag_in_request = true but no [tier_policy.<tier>.tag] policy is configured".into());
        }
        let has_executable_tag_head = pipeline_config.tiers.iter().any(|decl| {
            decl.tier.role == Role::Encoder
                && decl.tier.request_shape == Some(RequestShape::ScoreTasks)
                && tier_enabled(decl, &config.router, &pipeline_config.legacy)
                && pipeline_config.cascades.task_uses(&TaskId::tag(), &decl.name)
                && pipeline_config.tier_policies.get(&decl.name)
                    .is_some_and(|by_task| by_task.contains_key(&TaskId::tag()))
        });
        if !has_executable_tag_head {
            return Err(r#"tasks.toml: joint tagging requires a tag policy on an enabled encoder tier configured with request_shape = "score_tasks""#.into());
        }
        Some(profile)
    } else {
        None
    };

    // ---- Build the available-tier registry -------------------------------
    // TierRegistry is the complete lookup of available instances by name. Task
    // cascades independently determine which registered tiers run and in what order.
    let mut registry = TierRegistry::new();
    for decl in &pipeline_config.tiers {
        if !tier_enabled(decl, &config.router, &pipeline_config.legacy) {
            tracing::info!(tier = %decl.name, "tier disabled by configuration");
            continue;
        }
        let instance = TierInstance::build(resolve_decl(decl, &config.router, &pipeline_config.legacy)?)?;
        tracing::info!(
            tier = %instance.name(), id = %instance.id(), role = instance.tier().role.as_str(),
            computation = instance.params().computation.as_str(), model = %instance.model_label(),
            "tier instance built"
        );
        registry.register(Arc::new(instance))?;
    }
    if registry.is_empty() {
        return Err("no cascade tier is enabled; enable at least one [[tier]]".into());
    }
    let execution_graph = pipeline_config.cascades.retain_available(&registry.names())?;
    if execution_graph.order().is_empty() {
        return Err("no task cascade contains an enabled tier".into());
    }
    tracing::info!(
        order = ?execution_graph.order().iter().map(|id| id.as_str()).collect::<Vec<_>>(),
        "cascade DAG topological order"
    );
    // The numbered telemetry columns predate named tiers and describe one rule
    // file and one fastText model: take them from the first such instance in
    // execution order.
    let artifact_hash = |role: Role| {
        execution_graph.order().iter()
            .filter_map(|name| registry.get(name))
            .find(|tier| tier.tier().role == role)
            .and_then(|tier| tier.artifact_hash().map(|hash| hex(hash)))
            .unwrap_or_else(|| "disabled".to_string())
    };
    let hashes = telemetry::Hashes {
        tier1_toml: artifact_hash(Role::Rules),
        tier2_bin: artifact_hash(Role::FastText),
    };
    for mismatch in registry.calibration_mismatches(&pipeline_config.tier_policies) {
        tracing::warn!(
            tier = %mismatch.tier, task = %mismatch.task,
            calibrated_for = %mismatch.calibrated_for, running = %mismatch.running,
            "tier_policy was calibrated for a different instance than the one now running under this name"
        );
    }
    for (task, path) in pipeline_config.cascades.paths() {
        if !pipeline_config.has_policy(task) {
            continue;
        }
        for tier in path.iter().filter(|tier| registry.get(tier).is_some()) {
            let has_policy = pipeline_config.tier_policies.get(tier)
                .is_some_and(|by_task| by_task.contains_key(task));
            if !has_policy {
                tracing::warn!(%task, %tier, "tier is in the task's cascade but has no tier_policy for it; it will never run for that task");
            }
        }
    }
    let cascade = Arc::new(CascadeRunner::new(registry, execution_graph)?);
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
        task_profiles: pipeline_config.profiles(),
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
