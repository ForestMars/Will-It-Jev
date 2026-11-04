// src/router/tier3_runner.rs
//
// Generic gRPC client for task-scoring sidecars. Tier 3 (ModernBERT), Tier 4
// (Harrier), and any compatible future sidecar use this same wire-protocol
// implementation. It returns raw predictions; task policies resolve them.

use std::time::Duration;

use tokio::time::timeout;
use tonic::transport::Channel;

use super::tier3_proto::{
    tier3_scorer_client::Tier3ScorerClient, ScoreRequest, ScoreTasksRequest,
    ScoreTasksResponse,
};

#[derive(Clone)]
pub struct Tier3Runner {
    client: Tier3ScorerClient<Channel>,
    max_call: Duration,
}

impl Tier3Runner {
    /// `connect_lazy` lets the application start when a sidecar is unavailable;
    /// connectivity faults are surfaced as per-request pass-through results.
    pub fn new(
        endpoint: String,
        max_call: Duration,
    ) -> Result<Self, tonic::codegen::http::uri::InvalidUri> {
        let channel = Channel::from_shared(endpoint)?.connect_lazy();
        Ok(Self {
            client: Tier3ScorerClient::new(channel),
            max_call,
        })
    }

    /// Legacy single-head scoring request. The task policy is applied by the
    /// cascade, not by this transport client.
    pub async fn score_raw(
        &self,
        prompt: &str,
        budget: Duration,
    ) -> Result<RawScore, &'static str> {
        if budget.is_zero() {
            return Err("deadline_exhausted");
        }
        let deadline = budget.min(self.max_call);
        let mut client = self.client.clone();
        let request = ScoreRequest {
            prompt: prompt.to_string(),
        };
        match timeout(deadline, client.score(request)).await {
            Err(_) => Err("timeout"),
            Ok(Err(_)) => Err("rpc_error"),
            Ok(Ok(response)) => {
                let response = response.into_inner();
                Ok(RawScore {
                    label: response.label,
                    confidence: response.confidence as f32,
                    model_id: response.model_id,
                })
            }
        }
    }

    /// Joint call: one shared encoder execution with one output head per task.
    pub async fn score_tasks(
        &self,
        prompt: &str,
        tasks: &[String],
        budget: Duration,
    ) -> Result<ScoreTasksResponse, TasksError> {
        if budget.is_zero() {
            return Err(TasksError::Fault("deadline_exhausted"));
        }
        let deadline = budget.min(self.max_call);
        let mut client = self.client.clone();
        let request = ScoreTasksRequest {
            prompt: prompt.to_string(),
            tasks: tasks.to_vec(),
        };
        match timeout(deadline, client.score_tasks(request)).await {
            Err(_) => Err(TasksError::Fault("timeout")),
            Ok(Err(status)) if status.code() == tonic::Code::Unimplemented => {
                Err(TasksError::Unimplemented)
            }
            Ok(Err(_)) => Err(TasksError::Fault("rpc_error")),
            Ok(Ok(response)) => Ok(response.into_inner()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RawScore {
    pub label: String,
    pub confidence: f32,
    pub model_id: String,
}

#[derive(Debug)]
pub enum TasksError {
    /// The sidecar predates the multi-task ScoreTasks RPC.
    Unimplemented,
    Fault(&'static str),
}
