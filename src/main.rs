// src/main.rs

mod config;
mod pipeline;
mod router;
mod tagging;
mod telemetry;

use std::{path::PathBuf, sync::Arc, time::Duration};

use crate::tagging::{ClickHouseRawTable, OvenV1Tagger, TagConfig};
use pipeline::tagger::CascadeTagger;
use pipeline::tier::{Tier, Tier3Tier};
use pipeline::{CascadeRunner, TaskId};
use router::dispatch::DownstreamDispatcher;
use router::service;
use router::tier1_runner::{Tier1Automaton, Tier1Engine};
use router::tier2_runner::{Tier2Runner, DEFAULT_CONFIDENCE_THRESHOLD};
use router::tier3_runner::Tier3Runner;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use sha2::{Digest, Sha256};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

fn sha256_file(path: &str) -> Result<String, std::io::Error> {
    let contents = std::fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(contents)))
}

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

    let tier1_path = env_or("TIER1_CONFIG", "config/tier1.toml");
    let tier1 = if config.router.tier1.enabled {
        Some(Tier1Engine::new(Tier1Automaton::from_toml_file(&tier1_path)?)?)
    } else {
        tracing::info!("tier 1 disabled by configuration");
        None
    };
    let model_path = if config.router.tier2.enabled {
        Some(std::env::var("FASTTEXT_MODEL").unwrap_or_else(|_| default_fasttext_model_path()))
    } else {
        tracing::info!("tier 2 disabled by configuration");
        None
    };
    if let Some(model_path) = model_path.as_deref() {
        if model_path == "models/tier2.bin" && !std::path::Path::new("models/tier2_small.bin").exists() {
            tracing::warn!(model_path, "using legacy fastText tier-2 model; prefer models/tier2_small.bin when available");
        }
    }
    let tier2 = model_path
        .as_deref()
        .map(|path| Tier2Runner::new(path, DEFAULT_CONFIDENCE_THRESHOLD))
        .transpose()?;
    let hashes = telemetry::Hashes {
        tier1_toml: if tier1.is_some() { sha256_file(&tier1_path)? } else { "disabled".into() },
        tier2_bin: match model_path.as_deref() {
            Some(path) => sha256_file(path)?,
            None => "disabled".into(),
        },
    };
    let pipeline_config =
        pipeline::config::load_or_default(env_or("TASKS_CONFIG", "config/tasks.toml"))?;
    let modernbert = if config.router.tier3.enabled {
        Some(Tier3Runner::new(
            env_or("MODERNBERT_ENDPOINT", "http://[::1]:50052"),
            0.95,
            0.70,
            Duration::from_millis(500),
        )?)
    } else {
        tracing::info!("tier 3 disabled by configuration");
        None
    };
    let harrier = if config.router.tier4.enabled {
        Some(Tier3Runner::new(
            env_or(
                "TIER4_HARRIER_ENDPOINT",
                &env_or("TIER3_ENDPOINT", "http://[::1]:50051"),
            ),
            0.95,
            0.70,
            Duration::from_millis(500),
        )?)
    } else {
        tracing::info!("tier 4 disabled by configuration");
        None
    };

    // Handles for the tagging cascade; the originals move into the router below.
    let tag_tier3 = modernbert.clone();
    let tag_tier4 = harrier.clone();

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
    let (capture_tx, mut capture_rx) = tokio::sync::mpsc::channel::<telemetry::RouteRecord>(1024);
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
    let tag_config = TagConfig {
        base_path: PathBuf::from("logs/task_tags.jsonl"),
        clickhouse_url: Some(tag_clickhouse_url),
        queue_capacity: 1024,
        concurrency: 4,
        call_timeout: Duration::from_secs(5),
        max_age: Duration::from_secs(60),
        sample_prob: env_or("TAG_SAMPLE_PROB", "1.0").parse()?,
    };
    let (tag_queue, tag_handle) = if pipeline_config.settings.tag_via_cascade {
        // Tagging as the second use case on the shared tiers (tiers 3 and 4 are the
        // only ones that can carry a tagging head).
        let profile = pipeline_config.profile(&TaskId::tag()).ok_or(
            "tasks.toml: tag_via_cascade = true but there is no [task.tag] profile",
        )?;
        let joint = &pipeline_config.settings.joint_tiers;
        let mut tiers: Vec<Arc<dyn Tier>> = Vec::new();
        if let Some(runner) = tag_tier3.clone() {
            tiers.push(Arc::new(Tier3Tier::new(3, "modernbert", runner, joint.contains(&3))));
        }
        if let Some(runner) = tag_tier4.clone() {
            tiers.push(Arc::new(Tier3Tier::new(4, "harrier", runner, joint.contains(&4))));
        }
        let available: Vec<u8> = tiers.iter().map(|tier| tier.id()).collect();
        if let Some(missing) = profile.tiers.keys().find(|tier| !available.contains(tier)) {
            return Err(format!(
                "tasks.toml: task.tag lists tier{missing}, which is disabled or cannot serve tagging"
            )
            .into());
        }
        let tagger = Arc::new(CascadeTagger::new(
            Arc::new(CascadeRunner::new(tiers)),
            profile,
            Duration::from_millis(pipeline_config.settings.tag_budget_ms),
            ollama_tagger,
        )?);
        tracing::info!("task tagging runs through the shared cascade tiers");
        tagging::spawn(tagger, tag_config)
    } else {
        tagging::spawn(Arc::new(ollama_tagger), tag_config)
    };

    let app = service::app(
        tier1,
        tier2,
        modernbert,
        harrier,
        dispatcher,
        hashes,
        capture_tx.clone(),
        tag_queue.clone(),
        Duration::from_millis(env_or("ROUTER_BUDGET_MS", "600").parse()?),
    );
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
