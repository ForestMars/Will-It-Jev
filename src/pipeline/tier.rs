// src/pipeline/tier.rs
//
// The tier ontology, and the code that evaluates it.
//
//   Tier          the reusable definition: a role, how it is reached, and, for an
//                 encoder, which RPC contract it speaks. No instance parameters.
//   Params        what distinguishes one instance of a Tier from another: the
//                 model format and model, rule file, endpoint, timeout.
//   TierInstance  a Tier plus resolved instance parameters, loaded artifacts, and runner.
//                 Cascade assigns its TierId using TierInstance::derive_id.
//
// A tier returns raw readings for the task IDs it can serve; all task-specific
// acceptance policy lives outside this module.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[cfg(test)]
use async_trait::async_trait;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::task::{OutputKind, Reading, Task, TierReading, LABEL_JEV, LABEL_LLM};
use crate::router::tier1_runner::{Tier1Automaton, Tier1Engine};
use crate::router::tier2_runner::{Tier2Outcome, Tier2Route, Tier2Runner};
use crate::router::tier3_runner::{TasksError, Tier3Runner};

// ---- authored tier name ----------------------------------------------------

/// Authored name of a declared tier. The registry uses it to retrieve the
/// concrete TierInstance built from that declaration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
#[serde(try_from = "String")]
pub struct TierName(String);

/// Reserved name used in route records when a task resolves through fallback.
pub const FALLBACK_TIER_NAME: &str = "fallback";

/// Error validating a TierName or parsing a TierId.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierTypeError(String);

impl fmt::Display for TierTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for TierTypeError {}

const MAX_TIER_NAME_LEN: usize = 32;

impl TierName {
    /// A tier name is 1 to 32 characters: a lowercase ASCII letter followed by
    /// lowercase letters, digits, `_` or `-`.
    pub fn new(name: impl Into<String>) -> Result<Self, TierTypeError> {
        let name = name.into();
        let mut chars = name.chars();
        let valid = !name.is_empty()
            && name.len() <= MAX_TIER_NAME_LEN
            && chars.next().is_some_and(|c| c.is_ascii_lowercase())
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
        if !valid {
            return Err(TierTypeError(format!(
                "invalid tier name {name:?}: use 1 to {MAX_TIER_NAME_LEN} characters, starting with a \
                 lowercase letter, then lowercase letters, digits, '_' or '-'"
            )));
        }
        if name == FALLBACK_TIER_NAME {
            return Err(TierTypeError(format!(
                "tier name {FALLBACK_TIER_NAME:?} is reserved for task fallbacks"
            )));
        }
        if name.strip_prefix("tier").is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())) {
            return Err(TierTypeError(format!(
                "tier name {name:?} is reserved: `tierN` is the alias for a tier numbered in [legacy_tiers]"
            )));
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for TierName {
    type Error = TierTypeError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl fmt::Display for TierName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Shorthand for tests, where tier names are literals.
#[cfg(test)]
pub fn tname(name: &str) -> TierName {
    TierName::new(name).expect("test tier name must be valid")
}

// ---- definition ------------------------------------------------------------

/// What a tier is for. `Encoder` is the tier that fronts a neural encoder served
/// by a sidecar; the same definition supports many instances, each running its
/// own model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Role {
    #[serde(rename = "rules")]
    Rules,
    #[serde(rename = "fasttext")]
    FastText,
    #[serde(rename = "encoder", alias = "scorer")]
    Encoder,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rules => "rules",
            Self::FastText => "fasttext",
            Self::Encoder => "encoder",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    InProcess,
    Grpc,
}


/// The RPC contract an encoder speaks: one route score, or one head per task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestShape {
    Score,
    ScoreTasks,
}


/// The on-disk representation of an encoder model artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ModelFormat {
    #[serde(rename = "onnx")]
    Onnx,
    #[serde(rename = "gguf")]
    Gguf,
}
impl ModelFormat {
    pub fn as_str(self) -> &'static str {
        match self { Self::Onnx => "onnx", Self::Gguf => "gguf" }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tier {
    pub exex: Role,
    pub transport: Transport,
    /// Encoder over gRPC only.
    pub request_shape: Option<RequestShape>,
}

impl Tier {
    /// A definition is coherent when its role, transport and request shape agree.
    pub fn validate(&self) -> Result<(), &'static str> {
        match (self.exex, self.transport, self.request_shape) {
            (Role::Rules | Role::FastText, Transport::InProcess, None) => Ok(()),
            (Role::Rules | Role::FastText, _, _) => {
                Err("rules and fasttext require transport = \"in_process\" and no request_shape")
            }
            (Role::Encoder, Transport::Grpc, Some(_)) => Ok(()),
            (Role::Encoder, _, _) => {
                Err("encoder requires transport = \"grpc\" and request_shape = \"score\" or \"score_tasks\"")
            }
        }
    }

    pub fn accepts(&self, model_format: Option<ModelFormat>) -> bool {
        matches!((self.exex, model_format),
            (Role::Rules | Role::FastText, None)
                | (Role::Encoder, Some(ModelFormat::Onnx | ModelFormat::Gguf)))
    }
}

// ---- declaration -----------------------------------------------------------

/// What distinguishes one instance of a Tier from another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Params {
    /// Encoder only: artifact representation, not inference runtime.
    pub model_format: Option<ModelFormat>,
    /// fasttext: path to the model file. encoder: the model name the sidecar is
    /// expected to report; a different `model_id` in a response is logged once.
    pub model: Option<String>,
    /// Rules only: path to the rule file.
    pub config: Option<String>,
    /// Encoder only.
    pub endpoint: Option<String>,
    /// Encoder only.
    pub timeout_ms: Option<u64>,
    pub enabled: Option<bool>,
}

// ---- evaluation contract ---------------------------------------------------

pub struct TierContext<'a> {
    pub prompt: &'a str,
    /// Tasks still open that this tier is eligible to answer.
    pub tasks: &'a [String],
    /// Task contracts are passed to tiers so heads can validate requested output shapes.
    pub task_profiles: &'a [Task],
    /// Time left on the run's global deadline.
    pub budget: Duration,
}

#[derive(Debug, Clone)]
pub struct TierOutput {
    pub model_id: String,
    pub readings: BTreeMap<String, TierReading>,
}

impl TierOutput {
    pub fn abstain_all(
        model_id: impl Into<String>,
        tasks: &[String],
        reason: &'static str,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            readings: tasks
                .iter()
                .map(|task| (task.clone(), TierReading::Abstain(reason)))
                .collect(),
        }
    }
}


/// Test double for a computation. Production code has no open extension point:
/// the set of runners is closed.
#[cfg(test)]
#[async_trait]
pub trait FakeEvaluate: Send + Sync {
    async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput;
}

enum Runner {
    Rules(Arc<Tier1Engine>),
    FastText(Arc<Tier2Runner>),
    Grpc(Tier3Runner),
    #[cfg(test)]
    Fake(Arc<dyn FakeEvaluate>),
}

fn unsupported_all(ctx: &TierContext<'_>, model_id: &str) -> TierOutput {
    TierOutput {
        model_id: model_id.to_string(),
        readings: ctx.tasks.iter()
            .map(|task| (task.clone(), TierReading::Unsupported))
            .collect(),
    }
}

/// A tier-local scorer for any non-route task. Resolution remains entirely in
/// Task and is therefore independent of this implementation.
pub trait TaskScorer: Send + Sync {
    fn score(&self, prompt: &str, profile: &Task) -> TierReading;
}

fn route_only(ctx: &TierContext<'_>, model_id: &str, route: TierReading) -> TierOutput {
    let route_id = String::from("route");
    let readings = ctx
        .tasks
        .iter()
        .map(|task| {
            let reading = if *task == route_id {
                route.clone()
            } else {
                TierReading::Unsupported
            };
            (task.clone(), reading)
        })
        .collect();
    TierOutput {
        model_id: model_id.to_string(),
        readings,
    }
}


fn sha256_file(path: &str) -> Result<[u8; 32], std::io::Error> {
    Ok(Sha256::digest(std::fs::read(path)?).into())
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

// ---- TierId ----------------------------------------------------------------

/// Generated TierId for a concrete TierInstance. `TierInstance::derive_id`
/// hashes its authored name, role, model format, model value and artifact hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TierId([u8; 16]);

impl TierId {
    pub const LEN: usize = 16;

    pub fn from_digest(digest: &[u8]) -> Self {
        let mut bytes = [0u8; Self::LEN];
        bytes.copy_from_slice(&digest[..Self::LEN]);
        Self(bytes)
    }

    pub fn parse(hex: &str) -> Result<Self, TierTypeError> {
        let bad = || TierTypeError(format!("invalid tier id {hex:?}: expected {} hex characters", Self::LEN * 2));
        if hex.len() != Self::LEN * 2 || !hex.is_ascii() {
            return Err(bad());
        }
        let mut bytes = [0u8; Self::LEN];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for TierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for TierId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

// ---- instance --------------------------------------------------------------

/// A Tier plus the parameters that distinguish this instance, loaded and ready to
/// evaluate. The registry stores instances by TierName; CascadeGraph derives TierId.
pub struct TierInstance {
    name: TierName,
    tier: Tier,
    params: Params,
    artifact_hash: Option<[u8; 32]>,
    runner: Runner,
    task_scorers: BTreeMap<String, Arc<dyn TaskScorer>>,
    request_shape_unimplemented: AtomicBool,
    model_mismatch_reported: AtomicBool,
}

impl TierInstance {
    /// Load what the declaration points at and connect the runner. `params` must
    /// already be resolved: every value the tier needs is on it.
    pub fn build(name: TierName, tier: Tier, params: Params) -> Result<Self, Box<dyn Error>> {
        tier.validate().map_err(|reason| format!("tier {name}: {reason}"))?;
        if !tier.accepts(params.model_format) {
            return Err(format!("tier {name}: model_format {:?} is incompatible with {} executor", params.model_format.map(ModelFormat::as_str), tier.exex.as_str()).into());
        }
        let (runner, artifact_hash) = match tier.exex {
            Role::Rules => {
                let path = params.config.as_deref()
                    .ok_or_else(|| format!("tier {name}: rules tier has no `config` path"))?;
                let hash = sha256_file(path)?;
                let engine = Arc::new(Tier1Engine::new(Tier1Automaton::from_toml_file(path)?)?);
                (Runner::Rules(engine), Some(hash))
            }
            Role::FastText => {
                let path = params.model.as_deref()
                    .ok_or_else(|| format!("tier {name}: fasttext tier has no `model` path"))?;
                let hash = sha256_file(path)?;
                (Runner::FastText(Arc::new(Tier2Runner::new(path)?)), Some(hash))
            }
            Role::Encoder => {
                let endpoint = params.endpoint.clone()
                    .ok_or_else(|| format!("tier {name}: encoder tier has no endpoint"))?;
                let timeout_ms = params.timeout_ms
                    .ok_or_else(|| format!("tier {name}: encoder tier has no timeout_ms"))?;
                (Runner::Grpc(Tier3Runner::new(endpoint, Duration::from_millis(timeout_ms))?), None)
            }
        };
        Ok(Self::assemble(name, tier, params, runner, artifact_hash))
    }

    fn assemble(
        name: TierName,
        tier: Tier,
        params: Params,
        runner: Runner,
        artifact_hash: Option<[u8; 32]>,
    ) -> Self {
        Self {
            name, tier, params, artifact_hash, runner,
            task_scorers: BTreeMap::new(),
            request_shape_unimplemented: AtomicBool::new(false),
            model_mismatch_reported: AtomicBool::new(false),
        }
    }

    /// Derive the TierId for this instance. The hash includes its authored name,
    /// role, model format, model value and artifact content hash. Endpoint,
    /// transport, invocation protocol, timeout and enabled state do not
    /// contribute to the ID.
    pub fn derive_id(
        name: &TierName,
        tier: &Tier,
        params: &Params,
        artifact_hash: Option<&[u8; 32]>,
    ) -> TierId {
        let mut digest = Sha256::new();
        let mut field = |key: &str, value: &str| {
            digest.update(key.as_bytes());
            digest.update(b"=");
            digest.update(value.as_bytes());
            digest.update(b"\0");
        };
        field("tier_name", name.as_str());
        field("role", tier.exex.as_str());
        field("model_format", params.model_format.map(ModelFormat::as_str).unwrap_or(""));
        field("model", params.model.as_deref().unwrap_or(""));
        field("artifact", &artifact_hash.map(|hash| hex(hash)).unwrap_or_default());
        TierId::from_digest(&digest.finalize())
    }

    pub fn name(&self) -> &TierName { &self.name }
    pub fn tier(&self) -> &Tier { &self.tier }
    pub fn params(&self) -> &Params { &self.params }
    pub fn artifact_hash(&self) -> Option<&[u8; 32]> { self.artifact_hash.as_ref() }

    /// Label recorded as `model_id` when the instance has no better answer, for
    /// example when a sidecar call fails. It comes from the model, never from the
    /// name: names are free to change.
    pub fn model_label(&self) -> String {
        match self.tier.exex {
            Role::Rules => "heuristic".to_string(),
            Role::FastText => self.params.model.as_deref()
                .and_then(|path| Path::new(path).file_name())
                .map(|file| file.to_string_lossy().into_owned())
                .unwrap_or_else(|| "fasttext".to_string()),
            Role::Encoder => self.params.model.clone().unwrap_or_else(|| "unknown".to_string()),
        }
    }

    #[allow(dead_code)]
    pub fn with_task_scorer(mut self, task: String, scorer: Arc<dyn TaskScorer>) -> Self {
        self.task_scorers.insert(task, scorer);
        self
    }

    /// A tier without a compatible head reports the task as unsupported.
    pub async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
        match &self.runner {
            Runner::Rules(engine) => self.evaluate_rules(engine, ctx),
            Runner::FastText(runner) => self.evaluate_fasttext(runner, ctx),
            Runner::Grpc(runner) => self.evaluate_grpc(runner, ctx).await,
            #[cfg(test)]
            Runner::Fake(fake) => fake.evaluate(ctx).await,
        }
    }

    /// Tasks other than route are served by an in-process scorer if one is wired.
    fn other_tasks(&self, ctx: &TierContext<'_>, readings: &mut BTreeMap<String, TierReading>) {
        for task in ctx.tasks.iter().filter(|task| task.as_str() != String::from("route").as_str()) {
            let reading = self.task_scorers.get(task)
                .and_then(|scorer| ctx.task_profiles.iter().find(|profile| profile.name.as_str() == task.as_str())
                    .map(|profile| scorer.score(ctx.prompt, profile)))
                .unwrap_or(TierReading::Unsupported);
            readings.insert(task.clone(), reading);
        }
    }

    fn evaluate_rules(&self, engine: &Tier1Engine, ctx: &TierContext<'_>) -> TierOutput {
        let mut readings = BTreeMap::new();
        if ctx.tasks.contains(&String::from("route")) {
            let result = engine.classify(ctx.prompt);
            let p_jev = if result.confidence.is_finite() {
                result.confidence.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (label, confidence) = if p_jev >= 0.5 {
                (LABEL_JEV, p_jev)
            } else {
                (LABEL_LLM, 1.0 - p_jev)
            };
            readings.insert(String::from("route"), TierReading::Scored(Reading {
                label: label.to_string(),
                confidence,
                scores: vec![
                    (LABEL_JEV.to_string(), p_jev),
                    (LABEL_LLM.to_string(), 1.0 - p_jev),
                ],
            }));
        }
        self.other_tasks(ctx, &mut readings);
        TierOutput { model_id: self.model_label(), readings }
    }

    fn evaluate_fasttext(&self, runner: &Tier2Runner, ctx: &TierContext<'_>) -> TierOutput {
        let mut readings = BTreeMap::new();
        if ctx.tasks.contains(&String::from("route")) {
            readings.insert(String::from("route"), tier2_reading(&runner.evaluate(ctx.prompt)));
        }
        self.other_tasks(ctx, &mut readings);
        TierOutput { model_id: self.model_label(), readings }
    }

    /// A sidecar serves whatever model it loaded. If the instance declares a model
    /// and the sidecar reports another, say so once.
    fn check_reported_model(&self, reported: &str) {
        if let Some(expected) = self.params.model.as_deref() {
            if expected != reported && !self.model_mismatch_reported.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    tier = %self.name, expected, reported,
                    "sidecar reported a different model than the instance declares"
                );
            }
        }
    }

    async fn route_via_score(&self, runner: &Tier3Runner, ctx: &TierContext<'_>) -> TierOutput {
        match runner.score_raw(ctx.prompt, ctx.budget).await {
            Err(reason) => TierOutput::abstain_all(self.model_label(), ctx.tasks, reason),
            Ok(raw) => {
                self.check_reported_model(&raw.model_id);
                route_only(
                    ctx,
                    &raw.model_id,
                    TierReading::Scored(Reading::new(raw.label, raw.confidence)),
                )
            }
        }
    }

    async fn evaluate_grpc(&self, runner: &Tier3Runner, ctx: &TierContext<'_>) -> TierOutput {
        match self.tier.request_shape {
            Some(RequestShape::ScoreTasks) => {}
            _ => {
                return if ctx.tasks.contains(&String::from("route")) {
                    self.route_via_score(runner, ctx).await
                } else {
                    unsupported_all(ctx, &self.model_label())
                };
            }
        }
        if self.request_shape_unimplemented.load(Ordering::Relaxed) {
            return TierOutput::abstain_all(self.model_label(), ctx.tasks, "score_tasks_unimplemented");
        }
        let requested_profiles: Vec<Task> = ctx.tasks.iter()
            .filter_map(|task| ctx.task_profiles.iter()
                .find(|profile| profile.name.as_str() == task.as_str()).cloned())
            .collect();
        match runner.score_tasks(ctx.prompt, &requested_profiles, ctx.budget).await {
            Ok(response) => {
                self.check_reported_model(&response.model_id);
                let readings = ctx.tasks.iter().map(|task| {
                    let reading = response.results.iter()
                        .find(|result| result.task == task.as_str())
                        .map(|result| {
                            let expected = ctx.task_profiles.iter()
                                .find(|profile| profile.name.as_str() == task.as_str())
                                .map(|profile| match profile.output_kind {
                                    OutputKind::Binary => 1,
                                    OutputKind::Multiclass => 2,
                                    OutputKind::Multilabel => 3,
                                    OutputKind::Ordinal => 4,
                                });
                            // Zero is the proto3 default; accept it for older sidecars.
                            let kind_matches = expected
                                .is_none_or(|kind| result.output_kind == 0 || result.output_kind == kind);
                            if !kind_matches { return TierReading::Unsupported; }
                            TierReading::Scored(Reading {
                                label: result.label.clone(),
                                confidence: result.confidence,
                                scores: result.scores.iter()
                                    .map(|score| (score.label.clone(), score.score))
                                    .collect(),
                            })
                        })
                        .unwrap_or(TierReading::Unsupported);
                    (task.clone(), reading)
                }).collect();
                TierOutput { model_id: response.model_id, readings }
            }
            Err(TasksError::Fault(reason)) => {
                TierOutput::abstain_all(self.model_label(), ctx.tasks, reason)
            }
            Err(TasksError::Unimplemented) => {
                self.request_shape_unimplemented.store(true, Ordering::Relaxed);
                TierOutput::abstain_all(self.model_label(), ctx.tasks, "score_tasks_unimplemented")
            }
        }
    }
}

#[cfg(test)]
impl TierInstance {
    /// An instance whose executor is a test double. The model is the name, so
    /// instances built from different names have different ids.
    pub fn fake(name: &str, fake: Arc<dyn FakeEvaluate>) -> Self {
        Self::fake_with_model(name, name, fake)
    }

    pub fn fake_with_model(name: &str, model: &str, fake: Arc<dyn FakeEvaluate>) -> Self {
        let tier = Tier { exex: Role::Rules, transport: Transport::InProcess, request_shape: None };
        let params = Params {
            model_format: None, model: Some(model.to_string()), config: None,
            endpoint: None, timeout_ms: None, enabled: None,
        };
        Self::assemble(
            TierName::new(name).expect("test tier name must be valid"), tier, params, Runner::Fake(fake), None,
        )
    }

    /// An encoder instance over an already-constructed gRPC runner.
    pub fn encoder(name: &str, shape: RequestShape, model: Option<&str>, runner: Tier3Runner) -> Self {
        let tier = Tier { exex: Role::Encoder, transport: Transport::Grpc, request_shape: Some(shape) };
        let params = Params {
            model_format: Some(ModelFormat::Onnx), model: model.map(str::to_string), config: None,
            endpoint: Some("http://127.0.0.1:9".into()), timeout_ms: Some(250), enabled: None,
        };
        Self::assemble(
            TierName::new(name).expect("test tier name must be valid"), tier, params, Runner::Grpc(runner), None,
        )
    }
}

/// Converts FastText's raw class evidence to a task reading. The confidence
/// used by route policy is the calibrated class-probability margin; the full
/// two-class distribution is retained for telemetry.
pub fn tier2_reading(outcome: &Tier2Outcome) -> TierReading {
    match outcome {
        Tier2Outcome::Abstain { reason } => TierReading::Abstain(reason),
        Tier2Outcome::Scored(score) => {
            let label = match score.route {
                Tier2Route::Jev => LABEL_JEV,
                Tier2Route::Llm => LABEL_LLM,
            };
            TierReading::Scored(Reading {
                label: label.to_string(),
                confidence: score.calibrated_score,
                scores: vec![
                    (LABEL_JEV.to_string(), score.p_jev),
                    (LABEL_LLM.to_string(), score.p_llm),
                ],
            })
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::tier3_proto::tier3_scorer_server::{
        Tier3Scorer, Tier3ScorerServer,
    };
    use crate::router::tier3_proto::{
        LabelScore, ScoreRequest, ScoreResponse, ScoreTasksRequest, ScoreTasksResponse, TaskScore,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tonic::{Request, Response, Status};

    #[test]
    fn tier_id_round_trips_through_hex() {
        let id = TierId::from_digest(&[0xab; 32]);
        assert_eq!(id.to_string(), "ab".repeat(16));
        assert_eq!(TierId::parse(&id.to_string()), Ok(id));
        assert!(TierId::parse("xyz").is_err());
        assert!(TierId::parse(&"g".repeat(32)).is_err());
    }

    #[test]
    fn tier_names_are_validated() {
        for ok in ["rules", "fasttext", "modernbert", "harrier", "oryn-2", "a", "prefilter_v2"] {
            assert!(TierName::new(ok).is_ok(), "{ok} should be valid");
        }
        for bad in ["", "Rules", "2fast", "has space", "x.y", "fallback", "tier3", &"a".repeat(33)] {
            assert!(TierName::new(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[derive(Clone, Default)]
    struct TestSidecar {
        joint: bool,
        score_calls: Arc<AtomicUsize>,
        tasks_calls: Arc<AtomicUsize>,
    }

    #[tonic::async_trait]
    impl Tier3Scorer for TestSidecar {
        async fn score(
            &self,
            _request: Request<ScoreRequest>,
        ) -> Result<Response<ScoreResponse>, Status> {
            self.score_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Response::new(ScoreResponse {
                raw_score: 0.97,
                label: LABEL_JEV.into(),
                confidence: 0.97,
                model_id: "test-sidecar".into(),
            }))
        }

        async fn score_tasks(
            &self,
            request: Request<ScoreTasksRequest>,
        ) -> Result<Response<ScoreTasksResponse>, Status> {
            self.tasks_calls.fetch_add(1, Ordering::SeqCst);
            if !self.joint {
                return Err(Status::unimplemented("no multi-task heads"));
            }
            let results = request
                .into_inner()
                .tasks
                .into_iter()
                .filter_map(|task| match task.as_str() {
                    "route" => Some(TaskScore {
                        task,
                        label: LABEL_JEV.into(),
                        confidence: 0.97,
                        scores: Vec::new(),
                        output_kind: 1,
                    }),
                    "tag" => Some(TaskScore {
                        task,
                        label: "engineering".into(),
                        confidence: 0.85,
                        scores: vec![
                            LabelScore { label: "engineering".into(), score: 0.85 },
                            LabelScore { label: "help_desk".into(), score: 0.10 },
                        ],
                        output_kind: 2,
                    }),
                    _ => None,
                })
                .collect();
            Ok(Response::new(ScoreTasksResponse {
                model_id: "joint-test-sidecar".into(),
                results,
                unsupported: Vec::new(),
            }))
        }
    }

    async fn endpoint(sidecar: TestSidecar) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(Tier3ScorerServer::new(sidecar))
                .serve(address)
                .await
                .unwrap();
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        format!("http://{address}")
    }

    fn instance_id(instance: &TierInstance) -> TierId {
        TierInstance::derive_id(instance.name(), instance.tier(), instance.params(), instance.artifact_hash())
    }

    fn encoder_config(name: &str, model_format: ModelFormat, model: Option<&str>, endpoint: &str) -> (TierName, Tier, Params) {
        (
            TierName::new(name).unwrap(),
            Tier { exex: Role::Encoder, transport: Transport::Grpc, request_shape: Some(RequestShape::Score) },
            Params {
                model_format: Some(model_format), model: model.map(str::to_string), config: None,
                endpoint: Some(endpoint.to_string()), timeout_ms: Some(250), enabled: None,
            },
        )
    }

    fn rules_config(name: &str, config: Option<&str>) -> (TierName, Tier, Params) {
        (
            TierName::new(name).unwrap(),
            Tier { exex: Role::Rules, transport: Transport::InProcess, request_shape: None },
            Params {
                model_format: None, model: None, config: config.map(str::to_string),
                endpoint: None, timeout_ms: None, enabled: None,
            },
        )
    }

    fn build_entry(entry: (TierName, Tier, Params)) -> Result<TierInstance, Box<dyn Error>> {
        TierInstance::build(entry.0, entry.1, entry.2)
    }

    #[tokio::test]
    async fn an_instance_keeps_everything_it_was_built_from() {
        let entry = encoder_config("enc-a", ModelFormat::Onnx, Some("model-a"), "http://127.0.0.1:9");
        let instance = build_entry(entry.clone()).unwrap();
        assert_eq!(instance.name(), &entry.0);
        assert_eq!(instance.tier(), &entry.1);
        assert_eq!(instance.params(), &entry.2);
    }

    #[tokio::test]
    async fn two_instances_of_one_tier_differ_by_their_parameters_and_so_by_id() {
        let onnx = build_entry(encoder_config("enc-a", ModelFormat::Onnx, Some("modernbert"), "http://127.0.0.1:9")).unwrap();
        let gguf = build_entry(encoder_config("enc-b", ModelFormat::Gguf, Some("harrier"), "http://127.0.0.1:9")).unwrap();
        assert_eq!(onnx.tier(), gguf.tier());
        assert_ne!(instance_id(&onnx), instance_id(&gguf));
    }

    #[tokio::test]
    async fn id_includes_the_tier_name_and_model_but_not_the_endpoint() {
        let base = build_entry(encoder_config("enc-a", ModelFormat::Onnx, Some("m"), "http://127.0.0.1:9")).unwrap();
        let renamed = build_entry(encoder_config("renamed", ModelFormat::Onnx, Some("m"), "http://127.0.0.1:9")).unwrap();
        let moved = build_entry(encoder_config("enc-a", ModelFormat::Onnx, Some("m"), "http://127.0.0.2:9")).unwrap();
        let swapped = build_entry(encoder_config("enc-a", ModelFormat::Onnx, Some("oryn"), "http://127.0.0.1:9")).unwrap();
        assert_ne!(instance_id(&base), instance_id(&renamed));
        assert_eq!(instance_id(&base), instance_id(&moved));
        assert_ne!(instance_id(&base), instance_id(&swapped));
    }

    #[test]
    fn a_model_format_that_does_not_fit_the_role_is_rejected() {
        let (name, tier, mut params) = rules_config("rules-a", Some("config/tier1.toml"));
        params.model_format = Some(ModelFormat::Onnx);
        let error = TierInstance::build(name, tier, params).err().expect("must fail");
        assert!(error.to_string().contains("model_format"));
    }

    #[test]
    fn a_rules_instance_loads_its_rule_file_and_hashes_it() {
        let instance = build_entry(rules_config("rules-a", Some("config/tier1.toml"))).unwrap();
        assert!(instance.artifact_hash().is_some());
        let missing = build_entry(rules_config("rules-b", None)).err().expect("must fail");
        assert!(missing.to_string().contains("no `config` path"));
    }

    #[test]
    fn rule_file_location_is_not_part_of_the_id_but_content_is() {
        let dir = std::env::temp_dir().join(format!("tier-id-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let original = std::fs::read_to_string("config/tier1.toml").unwrap();
        let (a, b) = (dir.join("a.toml"), dir.join("b.toml"));
        std::fs::write(&a, &original).unwrap();
        std::fs::write(&b, &original).unwrap();
        let same_a = build_entry(rules_config("rules-a", a.to_str())).unwrap();
        let same_b = build_entry(rules_config("rules-a", b.to_str())).unwrap();
        // Different path, identical content: location is not identity.
        assert_eq!(instance_id(&same_a), instance_id(&same_b));
        assert_eq!(same_a.artifact_hash(), same_b.artifact_hash());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_failed_sidecar_call_is_labelled_by_model_not_by_name() {
        let instance = build_entry(encoder_config("enc-a", ModelFormat::Onnx, Some("model-a"), "http://127.0.0.1:9")).unwrap();
        let tasks = [String::from("route")];
        let output = instance
            .evaluate(&TierContext { prompt: "test", tasks: &tasks, task_profiles: &[], budget: Duration::from_millis(500) })
            .await;
        assert_eq!(output.model_id, "model-a");
        assert!(matches!(output.readings[&String::from("route")], TierReading::Abstain(_)));
    }

    #[tokio::test]
    async fn joint_sidecar_scores_multiple_tasks_in_one_rpc() {
        let sidecar = TestSidecar { joint: true, ..Default::default() };
        let calls = sidecar.tasks_calls.clone();
        let scorer = Tier3Runner::new(endpoint(sidecar).await, Duration::from_secs(2)).unwrap();
        let tier = TierInstance::encoder("alternate", RequestShape::ScoreTasks, None, scorer);
        let tasks = [String::from("route"), String::from("tag")];
        let profiles = [
            Task {
                name: String::from("route"),
                output_kind: OutputKind::Binary,
                labels: None,
                label_metadata: Vec::new(),
                taxonomy_version: None,
                fallback: super::super::task::FallbackPolicy::Defer,
            },
            Task {
                name: String::from("tag"),
                output_kind: OutputKind::Multiclass,
                labels: None,
                label_metadata: Vec::new(),
                taxonomy_version: None,
                fallback: super::super::task::FallbackPolicy::Defer,
            },
        ];
        let output = tier
            .evaluate(&TierContext { prompt: "test", tasks: &tasks, task_profiles: &profiles, budget: Duration::from_secs(2) })
            .await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(matches!(output.readings[&String::from("route")], TierReading::Scored(_)));
        assert!(matches!(output.readings[&String::from("tag")], TierReading::Scored(_)));
    }
    #[tokio::test]
    async fn score_tasks_request_shape_is_used_even_for_a_single_task() {
        let sidecar = TestSidecar { joint: true, ..Default::default() };
        let task_calls = sidecar.tasks_calls.clone();
        let score_calls = sidecar.score_calls.clone();
        let scorer = Tier3Runner::new(endpoint(sidecar).await, Duration::from_secs(2)).unwrap();
        let tier = TierInstance::encoder("route-scorer", RequestShape::ScoreTasks, None, scorer);
        let tasks = [String::from("route")];
        let profiles = [Task {
            name: String::from("route"),
            output_kind: OutputKind::Binary,
            labels: None,
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: super::super::task::FallbackPolicy::Defer,
        }];
        let output = tier.evaluate(&TierContext {
            prompt: "test",
            tasks: &tasks,
            task_profiles: &profiles,
            budget: Duration::from_secs(2),
        }).await;
        assert_eq!(task_calls.load(Ordering::SeqCst), 1);
        assert_eq!(score_calls.load(Ordering::SeqCst), 0);
        assert!(matches!(output.readings[&String::from("route")], TierReading::Scored(_)));
    }

    #[tokio::test]
    async fn score_tasks_unimplemented_is_not_silently_downgraded_to_score() {
        let sidecar = TestSidecar { joint: false, ..Default::default() };
        let task_calls = sidecar.tasks_calls.clone();
        let score_calls = sidecar.score_calls.clone();
        let scorer = Tier3Runner::new(endpoint(sidecar).await, Duration::from_secs(2)).unwrap();
        let tier = TierInstance::encoder("no-joint", RequestShape::ScoreTasks, None, scorer);
        let tasks = [String::from("route")];
        let profiles = [Task {
            name: String::from("route"),
            output_kind: OutputKind::Binary,
            labels: None,
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: super::super::task::FallbackPolicy::Defer,
        }];
        let output = tier.evaluate(&TierContext {
            prompt: "test",
            tasks: &tasks,
            task_profiles: &profiles,
            budget: Duration::from_secs(2),
        }).await;
        assert_eq!(task_calls.load(Ordering::SeqCst), 1);
        assert_eq!(score_calls.load(Ordering::SeqCst), 0);
        assert!(matches!(output.readings[&String::from("route")], TierReading::Abstain("score_tasks_unimplemented")));
    }
}
