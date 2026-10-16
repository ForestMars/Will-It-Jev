// src/main.rs

mod router;
mod tagging;
mod telemetry;

use std::{path::PathBuf, sync::Arc, time::Duration};

use crate::tagging::{ClickHouseRawTable, OvenV1Tagger, TagConfig};
use router::dispatch::DownstreamDispatcher;
use router::service;
use router::tier1_runner::{Tier1Automaton, Tier1Engine};
use router::tier2_runner::{Tier2Runner, DEFAULT_CONFIDENCE_THRESHOLD};
use router::tier3_runner::Tier3Runner;

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
    let filter = match std::env::var("RUST_LOG") {
        Ok(filter) => EnvFilter::try_new(filter)?,
        Err(std::env::VarError::NotPresent) => EnvFilter::new("info"),
        Err(error) => return Err(error.into()),
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(false).compact())
        .init();

    tracing::info!("Cascade Router initialized successfully");

    let tier1_path = env_or("TIER1_CONFIG", "config/tier1.toml");
    let tier1 = Tier1Engine::new(Tier1Automaton::from_toml_file(&tier1_path)?)?;
    let model_path = std::env::var("FASTTEXT_MODEL").unwrap_or_else(|_| default_fasttext_model_path());
    if model_path == "models/tier2.bin" && !std::path::Path::new("models/tier2_small.bin").exists() {
        tracing::warn!(model_path, "using legacy fastText tier-2 model; prefer models/tier2_small.bin when available");
    }
    let tier2 = Tier2Runner::new(&model_path, DEFAULT_CONFIDENCE_THRESHOLD)?;
    let hashes = telemetry::Hashes {
        tier1_toml: sha256_file(&tier1_path)?,
        tier2_bin: sha256_file(&model_path)?,
    };
    let tier3 = Tier3Runner::new(
        env_or("TIER3_ENDPOINT", "http://[::1]:50051"),
        0.95,
        0.70,
        Duration::from_millis(500),
    )?;

    let downstream_timeout =
        Duration::from_millis(env_or("DOWNSTREAM_TIMEOUT_MS", "120000").parse()?);
    let dispatcher = DownstreamDispatcher::new(
        env_or("TIER4_ENDPOINT", "http://127.0.0.1:11435/api/decide"),
        env_or("OLLAMA_GENERATE_URL", "http://127.0.0.1:11434/api/generate"),
        env_or("GEMMA_MODEL", "gemma3:latest"),
        downstream_timeout,
        3,
        Duration::from_secs(30),
        Duration::from_secs(30),
    )?;

    let clickhouse_url = env_or("CLICKHOUSE_URL", "http://127.0.0.1:8123");
    let route_clickhouse_url = clickhouse_url.clone();
    let (capture_tx, mut capture_rx) = tokio::sync::mpsc::channel::<telemetry::RouteRecord>(1024);
    let capture_task = tokio::spawn(async move {
        tokio::fs::create_dir_all("logs").await?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/routing_capture.jsonl")
            .await?;
        let clickhouse_client = reqwest::Client::builder()
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

    let tagger = Arc::new(OvenV1Tagger::new(
        env_or("TAGGER_OLLAMA_URL", "http://127.0.0.1:11434"),
        env_or("TAGGER_MODEL", "qwen3:1.7b"),
    ));
    let (tag_queue, tag_handle) = tagging::spawn(
        tagger,
        TagConfig {
            base_path: PathBuf::from("logs/task_tags.jsonl"),
            clickhouse_url: Some(clickhouse_url),
            queue_capacity: 1024,
            concurrency: 4,
            call_timeout: Duration::from_secs(5),
            max_age: Duration::from_secs(60),
            sample_prob: env_or("TAG_SAMPLE_PROB", "1.0").parse()?,
        },
    );

    let app = service::app(
        tier1,
        tier2,
        tier3,
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
