// src/pipeline/runner.rs
//
// Generic cascade execution. The runner knows tiers and task IDs, but task
// definitions and tier/task resolution policies are supplied independently.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use tracing::debug;

use super::task::{
    Decision, FallbackPolicy, Reading, TaskId, TaskProfile, TierPolicyMap, TierReading,
};
use super::tier::{Tier, TierContext};

#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    Tier {
        tier: u8,
        model_id: String,
        reading: Reading,
    },
    Fallback {
        label: String,
    },
    Deferred,
}

#[derive(Debug, Clone)]
pub struct TierTrace {
    pub tier: u8,
    pub model_id: String,
    pub reading: TierReading,
    pub decision: Decision,
}

#[derive(Debug, Clone)]
pub struct TaskOutcome {
    pub resolution: Resolution,
    pub trace: Vec<TierTrace>,
}

#[derive(Debug, Clone)]
pub struct TierCall {
    pub tier: u8,
    pub tasks: Vec<TaskId>,
    pub took: Duration,
}

#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub tasks: BTreeMap<TaskId, TaskOutcome>,
    pub tier_calls: Vec<TierCall>,
}

pub struct CascadeRunner {
    tiers: Vec<Arc<dyn Tier>>,
}

/// Required-task completion and full cascade completion are separate signals from one run.
pub struct SplitRun {
    pub required: oneshot::Receiver<RunOutcome>,
    pub done: JoinHandle<RunOutcome>,
}

impl CascadeRunner {
    pub fn new(mut tiers: Vec<Arc<dyn Tier>>) -> Self {
        tiers.sort_by_key(|tier| tier.id());
        Self { tiers }
    }

    /// `profiles` describe tasks. `policies` independently determine which
    /// tier may resolve each task and the task's calibrated bar at that tier.
    pub async fn run(
        &self,
        prompt: &str,
        profiles: &[TaskProfile],
        policies: &TierPolicyMap,
        budget: Duration,
    ) -> RunOutcome {
        self.run_inner(prompt, profiles, policies, budget, None).await
    }

    /// Starts exactly one cascade. The required receiver is published as soon as
    /// its task IDs resolve; best-effort tasks continue in the same execution.
    pub fn run_split(
        self: Arc<Self>,
        prompt: String,
        profiles: Vec<TaskProfile>,
        required_tasks: BTreeSet<TaskId>,
        policies: Arc<TierPolicyMap>,
        budget: Duration,
    ) -> SplitRun {
        let (sender, required) = oneshot::channel();
        let done = tokio::spawn(async move {
            self.run_inner(&prompt, &profiles, &policies, budget, Some((required_tasks, sender))).await
        });
        SplitRun { required, done }
    }

    async fn run_inner(
        &self,
        prompt: &str,
        profiles: &[TaskProfile],
        policies: &TierPolicyMap,
        budget: Duration,
        mut required_tx: Option<(BTreeSet<TaskId>, oneshot::Sender<RunOutcome>)>,
    ) -> RunOutcome {
        let start = Instant::now();
        let mut active: BTreeSet<TaskId> = profiles.iter().map(|p| p.task.clone()).collect();
        debug_assert_eq!(active.len(), profiles.len(), "duplicate task ids in profiles");

        let mut resolved: BTreeMap<TaskId, Resolution> = BTreeMap::new();
        let mut traces: BTreeMap<TaskId, Vec<TierTrace>> = BTreeMap::new();
        let mut tier_calls = Vec::new();

        for tier in &self.tiers {
            if active.is_empty() || start.elapsed() >= budget {
                break;
            }

            let by_task = policies.get(&tier.id());
            let wanted: Vec<&TaskProfile> = profiles
                .iter()
                .filter(|profile| {
                    active.contains(&profile.task)
                        && by_task.is_some_and(|map| map.contains_key(&profile.task))
                })
                .collect();
            if wanted.is_empty() {
                continue;
            }

            let ids: Vec<TaskId> = wanted.iter().map(|profile| profile.task.clone()).collect();
            let call_started = Instant::now();
            debug!(
                tier = tier.id(),
                tier_name = tier.name(),
                task_count = ids.len(),
                "executing inference tier"
            );
            let output = tier
                .evaluate(&TierContext {
                    prompt,
                    tasks: &ids,
                    task_profiles: profiles,
                    budget: budget.saturating_sub(start.elapsed()),
                })
                .await;
            tier_calls.push(TierCall {
                tier: tier.id(),
                tasks: ids,
                took: call_started.elapsed(),
            });

            for profile in wanted {
                let reading = output
                    .readings
                    .get(&profile.task)
                    .cloned()
                    .unwrap_or(TierReading::Unsupported);
                let policy = by_task
                    .and_then(|map| map.get(&profile.task))
                    .expect("wanted task must have a tier policy");
                let decision = profile.decide(policy, &reading);
                debug!(task = %profile.task, tier = tier.id(), ?decision, "task decision");

                if let Decision::Resolve(resolved_reading) = &decision {
                    resolved.insert(
                        profile.task.clone(),
                        Resolution::Tier {
                            tier: tier.id(),
                            model_id: output.model_id.clone(),
                            reading: resolved_reading.clone(),
                        },
                    );
                    active.remove(&profile.task);
                }

                traces.entry(profile.task.clone()).or_default().push(TierTrace {
                    tier: tier.id(),
                    model_id: output.model_id.clone(),
                    reading,
                    decision,
                });
            }

            if let Some((required, _)) = required_tx.as_ref() {
                if required.iter().all(|task| resolved.contains_key(task)) {
                    if let Some((_, sender)) = required_tx.take() {
                        let _ = sender.send(build_outcome(profiles, &resolved, &traces, &tier_calls));
                    }
                }
            }
        }

        let outcome = build_outcome(profiles, &resolved, &traces, &tier_calls);
        if let Some((_, sender)) = required_tx.take() {
            let _ = sender.send(outcome.clone());
        }
        outcome
    }
}

fn build_outcome(
    profiles: &[TaskProfile],
    resolved: &BTreeMap<TaskId, Resolution>,
    traces: &BTreeMap<TaskId, Vec<TierTrace>>,
    tier_calls: &[TierCall],
) -> RunOutcome {
    let tasks = profiles.iter().map(|profile| {
        let resolution = resolved.get(&profile.task).cloned().unwrap_or_else(|| {
            match &profile.fallback {
                FallbackPolicy::Label(label) => Resolution::Fallback { label: label.clone() },
                FallbackPolicy::Defer => Resolution::Deferred,
            }
        });
        let trace = traces.get(&profile.task).cloned().unwrap_or_default();
        (profile.task.clone(), TaskOutcome { resolution, trace })
    }).collect();
    RunOutcome { tasks, tier_calls: tier_calls.to_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::task::{TierPolicy, LABEL_JEV, LABEL_LLM};
    use crate::pipeline::tier::TierOutput;
    use async_trait::async_trait;
    use std::sync::Mutex;

    type TierCallLog = Arc<Mutex<Vec<(u8, Vec<TaskId>)>>>;

    struct FakeTier {
        id: u8,
        answers: Vec<(TaskId, TierReading)>,
        calls: TierCallLog,
    }

    #[async_trait]
    impl Tier for FakeTier {
        fn id(&self) -> u8 { self.id }
        fn name(&self) -> &str { "fake" }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            self.calls.lock().unwrap().push((self.id, ctx.tasks.to_vec()));
            let readings = ctx.tasks.iter().map(|task| {
                let reading = self.answers.iter()
                    .find(|(id, _)| id == task)
                    .map(|(_, reading)| reading.clone())
                    .unwrap_or(TierReading::Abstain("unsupported_task"));
                (task.clone(), reading)
            }).collect();
            TierOutput { model_id: format!("fake-{}", self.id), readings }
        }
    }

    fn scored(label: &str, confidence: f32) -> TierReading {
        TierReading::Scored(Reading::new(label, confidence))
    }

    #[tokio::test]
    async fn separate_tasks_share_a_tier_call_and_resolve_independently() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tier = FakeTier {
            id: 3,
            answers: vec![
                (TaskId::route(), scored(LABEL_JEV, 0.97)),
                (TaskId::tag(), scored("engineering", 0.85)),
            ],
            calls: calls.clone(),
        };
        let runner = CascadeRunner::new(vec![Arc::new(tier)]);
        let profiles = vec![
            TaskProfile {
                task: TaskId::route(),
                output_kind: super::super::task::OutputKind::Binary,
                labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
                label_metadata: Vec::new(),
                taxonomy_version: None,
                fallback: FallbackPolicy::Label(LABEL_LLM.into()),
            },
            TaskProfile {
                task: TaskId::tag(),
                output_kind: super::super::task::OutputKind::Multiclass,
                labels: Some(vec!["engineering".into(), "help_desk".into()]),
                label_metadata: Vec::new(),
                taxonomy_version: None,
                fallback: FallbackPolicy::Defer,
            },
        ];
        let mut policies = TierPolicyMap::new();
        let mut tier3_policies = BTreeMap::new();
        tier3_policies.insert(TaskId::route(), TierPolicy::new(0.95).with_label(LABEL_LLM, 0.70));
        tier3_policies.insert(TaskId::tag(), TierPolicy::new(0.80));
        policies.insert(3, tier3_policies);

        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.len(), 2);
        assert!(matches!(
            &result.tasks[&TaskId::route()].resolution,
            Resolution::Tier { tier: 3, .. }
        ));
        assert!(matches!(
            &result.tasks[&TaskId::tag()].resolution,
            Resolution::Tier { tier: 3, .. }
        ));
    }

    struct DelayedTier {
        id: u8,
        delay: Duration,
    }

    #[async_trait]
    impl Tier for DelayedTier {
        fn id(&self) -> u8 { self.id }
        fn name(&self) -> &str { "delayed" }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            tokio::time::sleep(self.delay).await;
            let readings = ctx.tasks.iter().map(|task| {
                let reading = if *task == TaskId::route() {
                    scored(LABEL_JEV, 0.99)
                } else {
                    scored("help_desk", 0.92)
                };
                (task.clone(), reading)
            }).collect();
            TierOutput { model_id: format!("delayed-{}", self.id), readings }
        }
    }

    #[tokio::test]
    async fn required_route_is_published_before_best_effort_tier_finishes() {
        let runner = Arc::new(CascadeRunner::new(vec![
            Arc::new(DelayedTier { id: 1, delay: Duration::from_millis(5) }),
            Arc::new(DelayedTier { id: 3, delay: Duration::from_millis(250) }),
        ]));
        let profiles = vec![
            TaskProfile {
                task: TaskId::route(),
                output_kind: super::super::task::OutputKind::Binary,
                labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
                label_metadata: Vec::new(),
                taxonomy_version: None,
                fallback: FallbackPolicy::Label(LABEL_LLM.into()),
            },
            TaskProfile {
                task: TaskId::tag(),
                output_kind: super::super::task::OutputKind::Multiclass,
                labels: Some(vec!["help_desk".into()]),
                label_metadata: Vec::new(),
                taxonomy_version: None,
                fallback: FallbackPolicy::Defer,
            },
        ];
        let policies: TierPolicyMap = [
            (1, [(TaskId::route(), TierPolicy::new(0.9))].into()),
            (3, [(TaskId::tag(), TierPolicy::new(0.8))].into()),
        ].into();
        let split = runner.run_split(
            "prompt".into(),
            profiles,
            [TaskId::route()].into(),
            Arc::new(policies),
            Duration::from_secs(2),
        );
        let required = tokio::time::timeout(Duration::from_millis(100), split.required)
            .await.expect("route must resolve before tier 3 finishes")
            .expect("required result sender must remain alive");
        assert!(matches!(
            required.tasks[&TaskId::route()].resolution,
            Resolution::Tier { tier: 1, .. }
        ));
        assert_eq!(required.tasks[&TaskId::tag()].resolution, Resolution::Deferred);
        let complete = split.done.await.unwrap();
        assert!(matches!(
            complete.tasks[&TaskId::tag()].resolution,
            Resolution::Tier { tier: 3, .. }
        ));
        assert!(matches!(
            complete.tasks[&TaskId::route()].resolution,
            Resolution::Tier { tier: 1, .. }
        ));
    }

    #[tokio::test]
    async fn task_without_a_policy_at_a_tier_is_not_sent_to_that_tier() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tier = FakeTier {
            id: 3,
            answers: vec![(TaskId::tag(), scored("engineering", 0.99))],
            calls: calls.clone(),
        };
        let runner = CascadeRunner::new(vec![Arc::new(tier)]);
        let tag = TaskProfile {
            task: TaskId::tag(),
            output_kind: super::super::task::OutputKind::Multiclass,
            labels: None,
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        };
        let profiles = vec![tag];
        let policies = TierPolicyMap::new();

        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(result.tasks[&TaskId::tag()].resolution, Resolution::Deferred);
    }
}
