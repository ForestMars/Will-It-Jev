// src/pipeline/tier.rs
//
// Scorer side of the split. A `Tier` returns raw per-task readings and never applies
// a threshold; thresholds live in `TaskProfile`. The adapters below wrap the existing
// tier runners without changing them (tier 1 and 2 are route-only, tiers 3 and 4 are
// the gRPC sidecars).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use super::task::{Reading, TaskId, TierReading, LABEL_JEV, LABEL_LLM};
use crate::router::tier1_runner::Tier1Engine;
use crate::router::tier2_runner::{Tier2Outcome, Tier2Route, Tier2Runner};
use crate::router::tier3_runner::{Tier3Runner, TasksError};

pub struct TierContext<'a> {
    pub prompt: &'a str,
    /// Tasks still open that this tier is allowed to answer, never empty.
    pub tasks: &'a [TaskId],
    /// Time left on the run's global deadline.
    pub budget: Duration,
}

#[derive(Debug, Clone)]
pub struct TierOutput {
    pub model_id: String,
    pub readings: BTreeMap<TaskId, TierReading>,
}

impl TierOutput {
    pub fn abstain_all(model_id: impl Into<String>, tasks: &[TaskId], reason: &'static str) -> Self {
        Self {
            model_id: model_id.into(),
            readings: tasks.iter().map(|t| (t.clone(), TierReading::Abstain(reason))).collect(),
        }
    }
}

#[async_trait]
pub trait Tier: Send + Sync {
    /// Position in the cascade, matching the numbering in telemetry (1..=4 for the
    /// local tiers).
    fn id(&self) -> u8;
    fn name(&self) -> &str;
    /// Answer as many of `ctx.tasks` as the tier can. Tasks the tier has no head for
    /// are `Abstain("unsupported_task")`, not an error.
    async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput;
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
                TierReading::Abstain("unsupported_task")
            };
            (task.clone(), reading)
        })
        .collect();
    TierOutput { model_id: model_id.to_string(), readings }
}

// ---- Tier 1: heuristic pattern engine ------------------------------------

pub struct Tier1Tier {
    engine: Arc<Tier1Engine>,
}

impl Tier1Tier {
    pub fn new(engine: Arc<Tier1Engine>) -> Self {
        Self { engine }
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
        let r = self.engine.classify(ctx.prompt);
        // Tier 1 reports a jev confidence, and only when it decided.
        let reading = if r.decided {
            TierReading::Scored(Reading::new(LABEL_JEV, r.confidence))
        } else {
            TierReading::Abstain("ambiguous")
        };
        route_only(ctx, "heuristic", reading)
    }
}

// ---- Tier 2: FastText -----------------------------------------------------

pub struct Tier2Tier {
    runner: Arc<Tier2Runner>,
}

impl Tier2Tier {
    pub fn new(runner: Arc<Tier2Runner>) -> Self {
        Self { runner }
    }
}

/// Raw reading from a Tier 2 outcome. The calibrated margin is the confidence the
/// cascade thresholds, whether or not Tier 2 itself resolved.
pub fn tier2_reading(outcome: &Tier2Outcome) -> TierReading {
    let label = |route: &Tier2Route| match route {
        Tier2Route::Jev => LABEL_JEV,
        Tier2Route::LLM => LABEL_LLM,
    };
    match outcome {
        Tier2Outcome::Resolved { route, calibrated_score, .. } => {
            TierReading::Scored(Reading::new(label(route), *calibrated_score))
        }
        Tier2Outcome::PassThrough { reason, .. } if matches!(*reason, "empty_input" | "no_predictions") => {
            TierReading::Abstain(reason)
        }
        Tier2Outcome::PassThrough { top_guess, score, .. } => {
            TierReading::Scored(Reading::new(label(top_guess), *score))
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
        let outcome = self.runner.evaluate(ctx.prompt);
        route_only(ctx, "fasttext", tier2_reading(&outcome))
    }
}

// ---- Tiers 3 and 4: gRPC sidecars ----------------------------------------

pub struct Tier3Tier {
    id: u8,
    name: &'static str,
    runner: Tier3Runner,
    /// The sidecar implements `ScoreTasks`. Off by default: until a sidecar has a head
    /// for a second task there is nothing for the joint call to add.
    joint: bool,
    /// Set the first time the sidecar answers UNIMPLEMENTED, so later requests skip
    /// the doomed round trip.
    joint_unimplemented: AtomicBool,
}

impl Tier3Tier {
    pub fn new(id: u8, name: &'static str, runner: Tier3Runner, joint: bool) -> Self {
        Self { id, name, runner, joint, joint_unimplemented: AtomicBool::new(false) }
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
impl Tier for Tier3Tier {
    fn id(&self) -> u8 {
        self.id
    }
    fn name(&self) -> &str {
        self.name
    }
    async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
        let route = TaskId::route();
        let only_route = ctx.tasks.len() == 1 && ctx.tasks[0] == route;
        let use_joint =
            self.joint && !only_route && !self.joint_unimplemented.load(Ordering::Relaxed);

        if use_joint {
            let ids: Vec<String> = ctx.tasks.iter().map(|t| t.as_str().to_string()).collect();
            match self.runner.score_tasks(ctx.prompt, &ids, ctx.budget).await {
                Ok(resp) => {
                    let readings = ctx
                        .tasks
                        .iter()
                        .map(|task| {
                            let reading = resp
                                .results
                                .iter()
                                .find(|r| r.task == task.as_str())
                                .map(|r| {
                                    TierReading::Scored(Reading {
                                        label: r.label.clone(),
                                        confidence: r.confidence,
                                        scores: r.scores.iter().map(|s| (s.label.clone(), s.score)).collect(),
                                    })
                                })
                                .unwrap_or(TierReading::Abstain("unsupported_task"));
                            (task.clone(), reading)
                        })
                        .collect();
                    return TierOutput { model_id: resp.model_id, readings };
                }
                Err(TasksError::Fault(reason)) => {
                    return TierOutput::abstain_all(self.name, ctx.tasks, reason);
                }
                Err(TasksError::Unimplemented) => {
                    tracing::warn!(tier = self.id, "sidecar has no ScoreTasks; serving route via Score only");
                    self.joint_unimplemented.store(true, Ordering::Relaxed);
                }
            }
        }

        if ctx.tasks.contains(&route) {
            self.route_via_score(ctx).await
        } else {
            TierOutput::abstain_all(self.name, ctx.tasks, "unsupported_task")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::tier3_proto::tier3_scorer_server::{Tier3Scorer, Tier3ScorerServer};
    use crate::router::tier3_proto::{
        LabelScore, ScoreRequest, ScoreResponse, ScoreTasksRequest, ScoreTasksResponse, TaskScore,
    };
    use std::sync::atomic::AtomicUsize;
    use tonic::{Request, Response, Status};

    #[derive(Clone, Default)]
    struct Sidecar {
        joint: bool,
        score_calls: Arc<AtomicUsize>,
        task_calls: Arc<AtomicUsize>,
    }

    #[tonic::async_trait]
    impl Tier3Scorer for Sidecar {
        async fn score(&self, _: Request<ScoreRequest>) -> Result<Response<ScoreResponse>, Status> {
            self.score_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Response::new(ScoreResponse {
                raw_score: 0.9,
                confidence: 0.9,
                label: LABEL_JEV.into(),
                model_id: "legacy".into(),
            }))
        }
        async fn score_tasks(
            &self,
            req: Request<ScoreTasksRequest>,
        ) -> Result<Response<ScoreTasksResponse>, Status> {
            self.task_calls.fetch_add(1, Ordering::SeqCst);
            if !self.joint {
                return Err(Status::unimplemented("no joint heads"));
            }
            let mut results = Vec::new();
            let mut unsupported = Vec::new();
            for task in req.into_inner().tasks {
                match task.as_str() {
                    "route" => results.push(TaskScore {
                        task, label: LABEL_JEV.into(), confidence: 0.97, scores: vec![],
                    }),
                    "tag" => results.push(TaskScore {
                        task,
                        label: "engineering".into(),
                        confidence: 0.85,
                        scores: vec![
                            LabelScore { label: "engineering".into(), score: 0.85 },
                            LabelScore { label: "help_desk".into(), score: 0.10 },
                        ],
                    }),
                    _ => unsupported.push(task),
                }
            }
            Ok(Response::new(ScoreTasksResponse { model_id: "joint".into(), results, unsupported }))
        }
    }

    async fn serve(sidecar: Sidecar) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(Tier3ScorerServer::new(sidecar))
                .serve(addr)
                .await
                .unwrap();
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        format!("http://{addr}")
    }

    fn tier(endpoint: String, joint: bool) -> Tier3Tier {
        let runner = Tier3Runner::new(endpoint, 0.95, 0.70, Duration::from_secs(2)).unwrap();
        Tier3Tier::new(4, "harrier", runner, joint)
    }

    fn ctx<'a>(tasks: &'a [TaskId]) -> TierContext<'a> {
        TierContext { prompt: "p", tasks, budget: Duration::from_secs(2) }
    }

    #[tokio::test]
    async fn joint_sidecar_serves_both_tasks_in_one_rpc() {
        let sidecar = Sidecar { joint: true, ..Default::default() };
        let (score_calls, task_calls) = (sidecar.score_calls.clone(), sidecar.task_calls.clone());
        let t = tier(serve(sidecar).await, true);
        let tasks = [TaskId::route(), TaskId::tag()];
        let out = t.evaluate(&ctx(&tasks)).await;

        assert_eq!(task_calls.load(Ordering::SeqCst), 1);
        assert_eq!(score_calls.load(Ordering::SeqCst), 0);
        assert_eq!(out.model_id, "joint");
        let TierReading::Scored(tag) = &out.readings[&TaskId::tag()] else { panic!() };
        assert_eq!(tag.label, "engineering");
        assert_eq!(tag.scores.len(), 2);
        assert!(matches!(out.readings[&TaskId::route()], TierReading::Scored(_)));
    }

    #[tokio::test]
    async fn route_alone_keeps_using_the_legacy_score_rpc() {
        let sidecar = Sidecar { joint: true, ..Default::default() };
        let (score_calls, task_calls) = (sidecar.score_calls.clone(), sidecar.task_calls.clone());
        let t = tier(serve(sidecar).await, true);
        let tasks = [TaskId::route()];
        t.evaluate(&ctx(&tasks)).await;
        assert_eq!((score_calls.load(Ordering::SeqCst), task_calls.load(Ordering::SeqCst)), (1, 0));
    }

    /// A sidecar that predates ScoreTasks: route still resolves, tag abstains, and the
    /// router stops probing the missing RPC after the first UNIMPLEMENTED.
    #[tokio::test]
    async fn old_sidecar_degrades_to_route_only() {
        let sidecar = Sidecar::default();
        let (score_calls, task_calls) = (sidecar.score_calls.clone(), sidecar.task_calls.clone());
        let t = tier(serve(sidecar).await, true);
        let tasks = [TaskId::route(), TaskId::tag()];
        for _ in 0..2 {
            let out = t.evaluate(&ctx(&tasks)).await;
            assert!(matches!(out.readings[&TaskId::route()], TierReading::Scored(_)));
            assert_eq!(out.readings[&TaskId::tag()], TierReading::Abstain("unsupported_task"));
        }
        assert_eq!(task_calls.load(Ordering::SeqCst), 1, "probed once, then remembered");
        assert_eq!(score_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn unreachable_sidecar_abstains_every_task() {
        let t = tier("http://127.0.0.1:1".into(), true);
        let tasks = [TaskId::route(), TaskId::tag()];
        let out = t.evaluate(&TierContext { prompt: "p", tasks: &tasks, budget: Duration::from_millis(200) }).await;
        assert!(out.readings.values().all(|r| matches!(r, TierReading::Abstain(_))));
    }

    #[test]
    fn tier2_reading_mirrors_the_legacy_outcome() {
        let resolved = Tier2Outcome::Resolved {
            route: Tier2Route::Jev, confidence: 0.99, calibrated_score: 0.97,
            p_jev: 0.985, p_llm: 0.015, raw_probabilities: vec![],
        };
        assert_eq!(tier2_reading(&resolved), TierReading::Scored(Reading::new(LABEL_JEV, 0.97)));
        let empty = Tier2Outcome::PassThrough {
            top_guess: Tier2Route::LLM, score: 0.0, reason: "empty_input",
            p_jev: 0.0, p_llm: 0.0, raw_probabilities: vec![],
        };
        assert_eq!(tier2_reading(&empty), TierReading::Abstain("empty_input"));
    }
}
