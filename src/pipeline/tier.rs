// src/pipeline/tier.rs
//
// Execution-tier adapters. A tier returns raw readings for the task IDs it can
// serve; all task-specific acceptance policy lives outside this module.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use super::task::{OutputKind, Reading, TaskId, TaskProfile, TierReading, LABEL_JEV, LABEL_LLM};
use crate::router::tier1_runner::Tier1Engine;
use crate::router::tier2_runner::{Tier2Outcome, Tier2Route, Tier2Runner};
use crate::router::tier3_runner::{TasksError, Tier3Runner};

pub struct TierContext<'a> {
    pub prompt: &'a str,
    /// Tasks still open that this tier is eligible to answer.
    pub tasks: &'a [TaskId],
    /// Task contracts are passed to tiers so heads can validate requested output shapes.
    pub task_profiles: &'a [TaskProfile],
    /// Time left on the run's global deadline.
    pub budget: Duration,
}

#[derive(Debug, Clone)]
pub struct TierOutput {
    pub model_id: String,
    pub readings: BTreeMap<TaskId, TierReading>,
}

impl TierOutput {
    pub fn abstain_all(
        model_id: impl Into<String>,
        tasks: &[TaskId],
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

#[async_trait]
pub trait Tier: Send + Sync {
    /// Position in the execution cascade; it is not a task identifier.
    fn id(&self) -> u8;
    fn name(&self) -> &str;
    /// A tier without a compatible head reports the task as unsupported.
    async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput;
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
/// TaskProfile and is therefore independent of this implementation.
pub trait TaskScorer: Send + Sync {
    fn score(&self, prompt: &str, profile: &TaskProfile) -> TierReading;
}

fn route_only(ctx: &TierContext<'_>, model_id: &str, route: TierReading) -> TierOutput {
    let route_id = TaskId::route();
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

pub struct Tier1Tier {
    engine: Arc<Tier1Engine>,
    task_scorers: BTreeMap<TaskId, Arc<dyn TaskScorer>>,
}

impl Tier1Tier {
    pub fn new(engine: Arc<Tier1Engine>) -> Self {
        Self { engine, task_scorers: BTreeMap::new() }
    }

    pub fn with_task_scorer(mut self, task: TaskId, scorer: Arc<dyn TaskScorer>) -> Self {
        self.task_scorers.insert(task, scorer);
        self
    }
}

#[async_trait]
impl Tier for Tier1Tier {
    fn id(&self) -> u8 {
        1
    }

    fn name(&self) -> &str {
        "heuristic"
    }

    async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
        let mut readings = BTreeMap::new();
        if ctx.tasks.contains(&TaskId::route()) {
            let result = self.engine.classify(ctx.prompt);
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
            readings.insert(TaskId::route(), TierReading::Scored(Reading {
                label: label.to_string(),
                confidence,
                scores: vec![
                    (LABEL_JEV.to_string(), p_jev),
                    (LABEL_LLM.to_string(), 1.0 - p_jev),
                ],
            }));
        }
        for task in ctx.tasks.iter().filter(|task| task.as_str() != TaskId::route().as_str()) {
            let reading = self.task_scorers.get(task)
                .and_then(|scorer| ctx.task_profiles.iter().find(|profile| profile.task.as_str() == task.as_str())
                    .map(|profile| scorer.score(ctx.prompt, profile)))
                .unwrap_or(TierReading::Unsupported);
            readings.insert(task.clone(), reading);
        }
        TierOutput { model_id: self.name().to_string(), readings }
    }
}

pub struct Tier2Tier {
    runner: Arc<Tier2Runner>,
    task_scorers: BTreeMap<TaskId, Arc<dyn TaskScorer>>,
}

impl Tier2Tier {
    pub fn new(runner: Arc<Tier2Runner>) -> Self {
        Self { runner, task_scorers: BTreeMap::new() }
    }

    pub fn with_task_scorer(mut self, task: TaskId, scorer: Arc<dyn TaskScorer>) -> Self {
        self.task_scorers.insert(task, scorer);
        self
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

#[async_trait]
impl Tier for Tier2Tier {
    fn id(&self) -> u8 {
        2
    }

    fn name(&self) -> &str {
        "fasttext"
    }

    async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
        let mut readings = BTreeMap::new();
        if ctx.tasks.contains(&TaskId::route()) {
            let outcome = self.runner.evaluate(ctx.prompt);
            readings.insert(TaskId::route(), tier2_reading(&outcome));
        }
        for task in ctx.tasks.iter().filter(|task| task.as_str() != TaskId::route().as_str()) {
            let reading = self.task_scorers.get(task)
                .and_then(|scorer| ctx.task_profiles.iter().find(|profile| profile.task.as_str() == task.as_str())
                    .map(|profile| scorer.score(ctx.prompt, profile)))
                .unwrap_or(TierReading::Unsupported);
            readings.insert(task.clone(), reading);
        }
        TierOutput { model_id: self.name().to_string(), readings }
    }
}

/// Generic adapter for any gRPC sidecar implementing the Tier 3 scorer
/// protocol. Tier 4 (Harrier/Oryn) is another instance with a different ID,
/// name, and endpoint; the Rust transport and output contract are identical.
pub struct GrpcScoringTier {
    id: u8,
    name: &'static str,
    runner: Tier3Runner,
    joint: bool,
    joint_unimplemented: AtomicBool,
}

impl GrpcScoringTier {
    pub fn new(id: u8, name: &'static str, runner: Tier3Runner, joint: bool) -> Self {
        Self {
            id,
            name,
            runner,
            joint,
            joint_unimplemented: AtomicBool::new(false),
        }
    }

    async fn route_via_score(&self, ctx: &TierContext<'_>) -> TierOutput {
        match self.runner.score_raw(ctx.prompt, ctx.budget).await {
            Err(reason) => TierOutput::abstain_all(self.name, ctx.tasks, reason),
            Ok(raw) => route_only(
                ctx,
                &raw.model_id,
                TierReading::Scored(Reading::new(raw.label, raw.confidence)),
            ),
        }
    }
}

#[async_trait]
impl Tier for GrpcScoringTier {
    fn id(&self) -> u8 {
        self.id
    }

    fn name(&self) -> &str {
        self.name
    }

    async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
        let route_id = TaskId::route();
        let only_route = ctx.tasks.len() == 1 && ctx.tasks[0] == route_id;
        let use_joint = self.joint
            && !only_route
            && !self.joint_unimplemented.load(Ordering::Relaxed);

        if use_joint {
            let requested_profiles: Vec<TaskProfile> = ctx.tasks.iter()
                .filter_map(|task| ctx.task_profiles.iter()
                    .find(|profile| profile.task.as_str() == task.as_str()).cloned())
                .collect();
            match self
                .runner
                .score_tasks(ctx.prompt, &requested_profiles, ctx.budget)
                .await
            {
                Ok(response) => {
                    let readings = ctx
                        .tasks
                        .iter()
                        .map(|task| {
                            let reading = response
                                .results
                                .iter()
                                .find(|result| result.task == task.as_str())
                                .map(|result| {
                                    let expected = ctx.task_profiles.iter()
                                        .find(|profile| profile.task.as_str() == task.as_str())
                                        .map(|profile| match profile.output_kind {
                                            OutputKind::Binary => 1,
                                            OutputKind::Multiclass => 2,
                                            OutputKind::Multilabel => 3,
                                            OutputKind::Ordinal => 4,
                                        });
                                    // Zero is the proto3 default and remains
                                    // accepted for legacy sidecars that omit kinds.
                                    let kind_matches = expected
                                        .map_or(true, |kind| result.output_kind == 0 || result.output_kind == kind);
                                    if !kind_matches {
                                        return TierReading::Unsupported;
                                    }
                                    TierReading::Scored(Reading {
                                        label: result.label.clone(),
                                        confidence: result.confidence,
                                        scores: result
                                            .scores
                                            .iter()
                                            .map(|score| (score.label.clone(), score.score))
                                            .collect(),
                                    })
                                })
                                .unwrap_or(TierReading::Unsupported);
                            (task.clone(), reading)
                        })
                        .collect();
                    return TierOutput {
                        model_id: response.model_id,
                        readings,
                    };
                }
                Err(TasksError::Fault(reason)) => {
                    return TierOutput::abstain_all(self.name, ctx.tasks, reason);
                }
                Err(TasksError::Unimplemented) => {
                    tracing::warn!(tier = self.id, "sidecar has no ScoreTasks RPC; falling back to single-task Score");
                    self.joint_unimplemented.store(true, Ordering::Relaxed);
                }
            }
        }

        if ctx.tasks.contains(&route_id) {
            self.route_via_score(ctx).await
        } else {
            unsupported_all(ctx, self.name)
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

    #[tokio::test]
    async fn joint_sidecar_scores_multiple_tasks_in_one_rpc() {
        let sidecar = TestSidecar { joint: true, ..Default::default() };
        let calls = sidecar.tasks_calls.clone();
        let scorer = Tier3Runner::new(endpoint(sidecar).await, Duration::from_secs(2)).unwrap();
        let tier = GrpcScoringTier::new(4, "alternate-sidecar", scorer, true);
        let tasks = [TaskId::route(), TaskId::tag()];
        let profiles = [
            TaskProfile {
                task: TaskId::route(),
                output_kind: OutputKind::Binary,
                labels: None,
                label_metadata: Vec::new(),
                taxonomy_version: None,
                fallback: super::super::task::FallbackPolicy::Defer,
            },
            TaskProfile {
                task: TaskId::tag(),
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
        assert!(matches!(output.readings[&TaskId::route()], TierReading::Scored(_)));
        assert!(matches!(output.readings[&TaskId::tag()], TierReading::Scored(_)));
    }
}
