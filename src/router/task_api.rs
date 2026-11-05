// src/router/task_api.rs
//
// Task-agnostic HTTP entry point. Callers select configured task IDs and any
// non-empty required subset. Required outcomes return early; remaining tasks
// continue in the same cascade and can be fetched by run ID.

use std::{
    collections::{BTreeSet, HashMap},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::pipeline::{
    runner::{CascadeRunner, Resolution, RunOutcome, TaskOutcome},
    task::{Decision, TaskId, TaskProfile, TierPolicyMap, TierReading},
};

const RUN_TTL: Duration = Duration::from_secs(900);

#[derive(Clone)]
struct TaskApiState {
    cascade: Arc<CascadeRunner>,
    profiles: Vec<TaskProfile>,
    policies: Arc<TierPolicyMap>,
    budget: Duration,
    runs: TaskRunStore,
}

#[derive(Deserialize)]
struct TaskExecutionRequest {
    prompt: String,
    tasks: Vec<String>,
    #[serde(default)]
    required_tasks: Option<Vec<String>>,
}

#[derive(Clone, Default)]
struct TaskRunStore {
    entries: Arc<Mutex<HashMap<String, TaskRunEntry>>>,
}

struct TaskRunEntry {
    result: watch::Receiver<Option<Result<RunOutcome, String>>>,
    expires_at: Instant,
}

enum TaskRunLookup {
    Missing,
    Pending,
    Complete(RunOutcome),
    Failed(String),
}

impl TaskRunStore {
    fn register(
        &self,
        run_id: String,
        done: JoinHandle<RunOutcome>,
    ) -> Result<(), String> {
        let (sender, receiver) = watch::channel(None);
        {
            let mut entries = self.entries.lock()
                .map_err(|_| "task run store mutex poisoned".to_string())?;
            entries.retain(|_, entry| entry.expires_at > Instant::now());
            entries.insert(run_id, TaskRunEntry {
                result: receiver,
                expires_at: Instant::now() + RUN_TTL,
            });
        }
        tokio::spawn(async move {
            let result = done.await.map_err(|error| error.to_string());
            let _ = sender.send(Some(result));
        });
        Ok(())
    }

    fn poll(&self, run_id: &str) -> Result<TaskRunLookup, String> {
        let mut entries = self.entries.lock()
            .map_err(|_| "task run store mutex poisoned".to_string())?;
        entries.retain(|_, entry| entry.expires_at > Instant::now());
        let Some(entry) = entries.get(run_id) else {
            return Ok(TaskRunLookup::Missing);
        };
        Ok(match entry.result.borrow().clone() {
            None => TaskRunLookup::Pending,
            Some(Ok(outcome)) => TaskRunLookup::Complete(outcome),
            Some(Err(error)) => TaskRunLookup::Failed(error),
        })
    }
}

#[derive(Debug)]
enum ApiError {
    Invalid(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            Self::Invalid(detail) => (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "invalid_task_request", "detail": detail})),
            ).into_response(),
            Self::Internal(detail) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "task_execution_failed", "detail": detail})),
            ).into_response(),
        }
    }
}

pub fn router(
    cascade: Arc<CascadeRunner>,
    profiles: Vec<TaskProfile>,
    policies: Arc<TierPolicyMap>,
    budget: Duration,
) -> Router {
    let state = TaskApiState {
        cascade,
        profiles,
        policies,
        budget,
        runs: TaskRunStore::default(),
    };
    Router::new()
        .route("/", post(execute_tasks))
        .route("/:run_id", get(get_task_run))
        .with_state(state)
}

async fn execute_tasks(
    State(state): State<TaskApiState>,
    Json(request): Json<TaskExecutionRequest>,
) -> Response {
    match execute_task_request(&state, request).await {
        Ok(payload) => (StatusCode::OK, Json(payload)).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn get_task_run(
    State(state): State<TaskApiState>,
    Path(run_id): Path<String>,
) -> Response {
    match state.runs.poll(&run_id) {
        Ok(TaskRunLookup::Missing) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "task_run_not_found", "run_id": run_id})),
        ).into_response(),
        Ok(TaskRunLookup::Pending) => (
            StatusCode::ACCEPTED,
            Json(json!({"run_id": run_id, "status": "pending"})),
        ).into_response(),
        Ok(TaskRunLookup::Complete(outcome)) => (
            StatusCode::OK,
            Json(json!({
                "run_id": run_id,
                "status": "complete",
                "tasks": all_task_results(&outcome),
            })),
        ).into_response(),
        Ok(TaskRunLookup::Failed(error)) | Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "task_execution_failed",
                "run_id": run_id,
                "detail": error,
            })),
        ).into_response(),
    }
}

async fn execute_task_request(
    state: &TaskApiState,
    request: TaskExecutionRequest,
) -> Result<Value, ApiError> {
    if request.prompt.trim().is_empty() {
        return Err(ApiError::Invalid("prompt must not be empty".into()));
    }
    if request.tasks.is_empty() {
        return Err(ApiError::Invalid(
            "tasks must contain at least one configured task ID".into(),
        ));
    }

    let mut selected = BTreeSet::new();
    let mut profiles = Vec::with_capacity(request.tasks.len());
    for name in &request.tasks {
        if name.trim().is_empty() {
            return Err(ApiError::Invalid("task IDs must not be empty".into()));
        }
        if !selected.insert(name.clone()) {
            return Err(ApiError::Invalid(format!("duplicate task ID {name:?}")));
        }
        let profile = state.profiles.iter()
            .find(|profile| profile.task.as_str() == name)
            .ok_or_else(|| ApiError::Invalid(format!("task {name:?} is not configured")))?;
        profiles.push(profile.clone());
    }

    let required_names = request.required_tasks
        .unwrap_or_else(|| request.tasks.clone());
    if required_names.is_empty() {
        return Err(ApiError::Invalid(
            "required_tasks must contain at least one task ID".into(),
        ));
    }

    let mut required_ids = BTreeSet::new();
    for name in &required_names {
        let id = TaskId::new(name.clone());
        if !required_ids.insert(id) {
            return Err(ApiError::Invalid(format!(
                "duplicate required task ID {name:?}"
            )));
        }
        if !selected.contains(name) {
            return Err(ApiError::Invalid(format!(
                "required task {name:?} is not in tasks"
            )));
        }
    }

    let run_id = Uuid::new_v4().to_string();
    if required_ids.len() == profiles.len() {
        let outcome = state.cascade
            .run(&request.prompt, &profiles, &state.policies, state.budget)
            .await;
        return Ok(json!({
            "run_id": run_id,
            "status": "complete",
            "tasks": all_task_results(&outcome),
        }));
    }

    let pending_names: Vec<String> = request.tasks.iter()
        .filter(|name| !required_ids.contains(&TaskId::new((*name).clone())))
        .cloned()
        .collect();
    let split = Arc::clone(&state.cascade).run_split(
        request.prompt,
        profiles,
        required_ids.clone(),
        Arc::clone(&state.policies),
        state.budget,
    );
    state.runs.register(run_id.clone(), split.done)
        .map_err(ApiError::Internal)?;
    let required_outcome = split.required.await.map_err(|error| {
        ApiError::Internal(format!("required task result channel closed: {error}"))
    })?;

    Ok(json!({
        "run_id": run_id,
        "status": "partial",
        "required_tasks": required_names,
        "tasks": selected_task_results(&required_outcome, &required_ids),
        "pending_tasks": pending_names,
    }))
}

fn all_task_results(outcome: &RunOutcome) -> Value {
    let mut tasks = Map::new();
    for (task_id, task_outcome) in &outcome.tasks {
        tasks.insert(task_id.as_str().to_string(), task_outcome_json(task_outcome));
    }
    Value::Object(tasks)
}

fn selected_task_results(outcome: &RunOutcome, selected: &BTreeSet<TaskId>) -> Value {
    let mut tasks = Map::new();
    for task_id in selected {
        if let Some(task_outcome) = outcome.tasks.get(task_id) {
            tasks.insert(task_id.as_str().to_string(), task_outcome_json(task_outcome));
        }
    }
    Value::Object(tasks)
}

fn task_outcome_json(outcome: &TaskOutcome) -> Value {
    let resolution = match &outcome.resolution {
        Resolution::Tier { tier, model_id, reading } => json!({
            "kind": "tier",
            "tier": tier,
            "model_id": model_id,
            "label": reading.label,
            "confidence": reading.confidence,
            "scores": reading.scores.iter().map(|(label, score)| {
                json!({"label": label, "score": score})
            }).collect::<Vec<_>>(),
        }),
        Resolution::Fallback { label } => json!({
            "kind": "fallback",
            "label": label,
        }),
        Resolution::Deferred => json!({"kind": "deferred"}),
    };

    let trace = outcome.trace.iter().map(|entry| {
        let reading = match &entry.reading {
            TierReading::Scored(reading) => json!({
                "kind": "scored",
                "label": reading.label,
                "confidence": reading.confidence,
            }),
            TierReading::Abstain(reason) => json!({
                "kind": "abstain",
                "reason": reason,
            }),
            TierReading::Unsupported => json!({"kind": "unsupported"}),
        };
        let decision = match &entry.decision {
            Decision::Resolve(reading) => json!({
                "kind": "resolve",
                "label": reading.label,
                "confidence": reading.confidence,
            }),
            Decision::Pass { reason, top_guess, score } => json!({
                "kind": "pass",
                "reason": reason,
                "top_guess": top_guess,
                "score": score,
            }),
        };
        json!({
            "tier": entry.tier,
            "model_id": entry.model_id,
            "reading": reading,
            "decision": decision,
        })
    }).collect::<Vec<_>>();
    json!({"resolution": resolution, "trace": trace})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{
        task::{FallbackPolicy, OutputKind, Reading, TierPolicy},
        tier::{Tier, TierContext, TierOutput},
    };
    use async_trait::async_trait;
    use std::collections::BTreeMap;
    use tokio::time::{sleep, timeout};

    type CallLog = Arc<Mutex<Vec<(u8, Vec<String>)>>>;

    struct FixtureTier {
        id: u8,
        delay: Duration,
        readings: BTreeMap<TaskId, TierReading>,
        calls: CallLog,
    }

    #[async_trait]
    impl Tier for FixtureTier {
        fn id(&self) -> u8 { self.id }
        fn name(&self) -> &str { "fixture" }

        async fn evaluate(&self, context: &TierContext<'_>) -> TierOutput {
            self.calls.lock().unwrap().push((
                self.id,
                context.tasks.iter().map(|task| task.as_str().to_string()).collect(),
            ));
            sleep(self.delay).await;
            let readings = context.tasks.iter().map(|task| {
                (
                    task.clone(),
                    self.readings.get(task).cloned().unwrap_or(TierReading::Unsupported),
                )
            }).collect();
            TierOutput {
                model_id: format!("fixture-tier-{}", self.id),
                readings,
            }
        }
    }

    fn profile(task: &str, kind: OutputKind, labels: &[&str]) -> TaskProfile {
        TaskProfile {
            task: TaskId::new(task),
            output_kind: kind,
            labels: Some(labels.iter().map(|label| (*label).to_string()).collect()),
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        }
    }

    #[tokio::test]
    async fn generic_api_runs_three_arbitrary_tasks_and_returns_required_subset_first() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let profiles = vec![
            profile("customer_intent", OutputKind::Multiclass, &["billing", "account"]),
            profile("urgency", OutputKind::Ordinal, &["low", "medium", "high"]),
            profile("fraud_risk", OutputKind::Multilabel, &["account_takeover", "payment_abuse"]),
        ];
        let tier1 = FixtureTier {
            id: 1,
            delay: Duration::ZERO,
            readings: [(
                TaskId::new("customer_intent"),
                TierReading::Scored(Reading::new("billing", 0.97)),
            )].into(),
            calls: calls.clone(),
        };
        let tier2 = FixtureTier {
            id: 2,
            delay: Duration::from_millis(300),
            readings: [
                (TaskId::new("urgency"), TierReading::Scored(Reading::new("high", 0.91))),
                (TaskId::new("fraud_risk"), TierReading::Scored(Reading::new("payment_abuse", 0.88))),
            ].into(),
            calls: calls.clone(),
        };
        let cascade = Arc::new(CascadeRunner::new(vec![
            Arc::new(tier1),
            Arc::new(tier2),
        ]));
        let policies: TierPolicyMap = [
            (1, [(TaskId::new("customer_intent"), TierPolicy::new(0.90))].into()),
            (2, [
                (TaskId::new("urgency"), TierPolicy::new(0.80)),
                (TaskId::new("fraud_risk"), TierPolicy::new(0.80)),
            ].into()),
        ].into();
        let state = TaskApiState {
            cascade,
            profiles,
            policies: Arc::new(policies),
            budget: Duration::from_secs(2),
            runs: TaskRunStore::default(),
        };
        let request = TaskExecutionRequest {
            prompt: "Evaluate the support request.".into(),
            tasks: vec![
                "customer_intent".into(),
                "urgency".into(),
                "fraud_risk".into(),
            ],
            required_tasks: Some(vec!["customer_intent".into()]),
        };

        let response = execute_task_request(&state, request).await.unwrap();
        assert_eq!(response["status"], "partial");
        assert_eq!(
            response["tasks"]["customer_intent"]["resolution"]["label"],
            "billing"
        );
        assert_eq!(response["pending_tasks"], json!(["urgency", "fraud_risk"]));
        let run_id = response["run_id"].as_str().unwrap().to_string();
        assert!(matches!(state.runs.poll(&run_id).unwrap(), TaskRunLookup::Pending));

        let complete = timeout(Duration::from_secs(2), async {
            loop {
                match state.runs.poll(&run_id).unwrap() {
                    TaskRunLookup::Complete(outcome) => break outcome,
                    TaskRunLookup::Failed(error) => panic!("task run failed: {error}"),
                    TaskRunLookup::Missing => panic!("task run expired before completion"),
                    TaskRunLookup::Pending => sleep(Duration::from_millis(5)).await,
                }
            }
        }).await.expect("best-effort tasks should finish within the test budget");

        assert_eq!(complete.tasks.len(), 3);
        assert_eq!(
            complete.tasks[&TaskId::new("urgency")].resolution,
            Resolution::Tier {
                tier: 2,
                model_id: "fixture-tier-2".into(),
                reading: Reading::new("high", 0.91),
            }
        );
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], (1, vec!["customer_intent".into()]));
        assert_eq!(
            calls[1],
            (2, vec!["urgency".into(), "fraud_risk".into()])
        );
    }
}
