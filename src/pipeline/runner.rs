// src/pipeline/runner.rs
//
// Generic cascade execution over configured tier instances. A tier definition
// supplies an execution contract; each TierInstance contributes its own
// identity, model selection and executor configuration. The registry vector is
// the only source of execution order.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use tracing::debug;

use super::registry::{
    TierDirectory, TierId, TierInstance, TierInstanceId, TierRegistry,
};
use super::task::{
    Decision, FallbackPolicy, Reading, TaskId, TaskProfile, TierPolicyMap, TierReading,
};
use super::tier::TierContext;

#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    Tier {
        tier_instance_id: TierInstanceId,
        tier_id: TierId,
        model_id: String,
        reading: Reading,
    },
    Fallback { label: String },
    Deferred,
}

#[derive(Debug, Clone)]
pub struct TierTrace {
    pub tier_instance_id: TierInstanceId,
    pub tier_id: TierId,
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
    pub tier_instance_id: TierInstanceId,
    pub tier_id: TierId,
    pub model_id: String,
    pub tasks: Vec<TaskId>,
    pub took: Duration,
}

#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub tasks: BTreeMap<TaskId, TaskOutcome>,
    pub tier_calls: Vec<TierCall>,
}

pub struct CascadeRunner {
    /// Ordered IDs only. Position is the index in this vector.
    sequence: Vec<TierInstanceId>,
    /// Stable-ID lookup for configured instances.
    instances: BTreeMap<TierInstanceId, TierInstance>,
    directory: Arc<TierDirectory>,
}

/// Required-task completion and full cascade completion are separate signals from one run.
pub struct SplitRun {
    pub required: oneshot::Receiver<RunOutcome>,
    pub done: JoinHandle<RunOutcome>,
}

impl CascadeRunner {
    /// Instances execute in the order in which they appear in the registry.
    pub fn new(registry: TierRegistry) -> Self {
        let directory = Arc::new(registry.directory());
        let (sequence, instances) = registry.into_ordered();
        Self { sequence, instances, directory }
    }

    pub fn directory(&self) -> Arc<TierDirectory> {
        Arc::clone(&self.directory)
    }

    pub async fn run(
        &self,
        prompt: &str,
        profiles: &[TaskProfile],
        policies: &TierPolicyMap,
        budget: Duration,
    ) -> RunOutcome {
        self.run_inner(prompt, profiles, policies, budget, None).await
    }

    /// Starts exactly one cascade. Required task results are published as soon
    /// as they resolve; best-effort tasks continue through the same run.
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
        let mut active: BTreeSet<TaskId> = profiles.iter().map(|profile| profile.task.clone()).collect();
        debug_assert_eq!(active.len(), profiles.len(), "duplicate task IDs in profiles");

        let mut resolved: BTreeMap<TaskId, Resolution> = BTreeMap::new();
        let mut traces: BTreeMap<TaskId, Vec<TierTrace>> = BTreeMap::new();
        let mut tier_calls = Vec::new();

        for instance_id in &self.sequence {
            if active.is_empty() || start.elapsed() >= budget {
                break;
            }
            let instance = self.instances.get(instance_id)
                .expect("every ID in the execution sequence has a registered instance");

            // Policies are bound to a configured instance, not its reusable
            // execution definition. Two grpc instances can therefore differ.
            let by_task = policies.get(&instance.id);
            let wanted: Vec<&TaskProfile> = profiles.iter().filter(|profile| {
                active.contains(&profile.task)
                    && by_task.is_some_and(|map| map.contains_key(&profile.task))
            }).collect();
            if wanted.is_empty() {
                continue;
            }

            let task_ids: Vec<TaskId> = wanted.iter().map(|profile| profile.task.clone()).collect();
            let call_started = Instant::now();
            let executor = instance.executor();
            debug!(
                tier_instance_id = %instance.id,
                tier_id = %instance.tier_id,
                model_id = instance.model_id.as_deref().unwrap_or("reported-by-executor"),
                task_count = task_ids.len(),
                "executing tier instance"
            );
            let output = executor.evaluate(&TierContext {
                prompt,
                tier_instance_id: &instance.id,
                tier_id: &instance.tier_id,
                model_id: instance.model_id.as_deref(),
                tasks: &task_ids,
                task_profiles: profiles,
                budget: budget.saturating_sub(start.elapsed()),
            }).await;

            // Prefer the configured model reference. An executor-reported ID
            // remains the fallback for older sidecars without model config.
            let model_id = instance.model_id.clone()
                .unwrap_or_else(|| output.model_id.clone());
            tier_calls.push(TierCall {
                tier_instance_id: instance.id.clone(),
                tier_id: instance.tier_id.clone(),
                model_id: model_id.clone(),
                tasks: task_ids,
                took: call_started.elapsed(),
            });

            for profile in wanted {
                let reading = output.readings.get(&profile.task)
                    .cloned().unwrap_or(TierReading::Unsupported);
                let policy = by_task.and_then(|map| map.get(&profile.task))
                    .expect("wanted task must have an instance policy");
                let decision = profile.decide(policy, &reading);
                debug!(
                    task = %profile.task,
                    tier_instance_id = %instance.id,
                    tier_id = %instance.tier_id,
                    ?decision,
                    "task decision"
                );

                if let Decision::Resolve(resolved_reading) = &decision {
                    resolved.insert(profile.task.clone(), Resolution::Tier {
                        tier_instance_id: instance.id.clone(),
                        tier_id: instance.tier_id.clone(),
                        model_id: model_id.clone(),
                        reading: resolved_reading.clone(),
                    });
                    active.remove(&profile.task);
                }

                traces.entry(profile.task.clone()).or_default().push(TierTrace {
                    tier_instance_id: instance.id.clone(),
                    tier_id: instance.tier_id.clone(),
                    model_id: model_id.clone(),
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
    use crate::pipeline::registry::{iid, tid, TierInstance};
    use crate::pipeline::task::{OutputKind, TierPolicy, LABEL_JEV, LABEL_LLM};
    use crate::pipeline::tier::{Tier, TierOutput};
    use async_trait::async_trait;
    use std::sync::Mutex;

    type CallLog = Arc<Mutex<Vec<(String, String, Option<String>, Vec<TaskId>)>>>;

    struct FakeTier {
        id: TierId,
        output_model: String,
        answers: Vec<(TaskId, TierReading)>,
        calls: CallLog,
    }

    #[async_trait]
    impl Tier for FakeTier {
        fn id(&self) -> &TierId { &self.id }
        fn name(&self) -> &str { self.id.as_str() }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            self.calls.lock().unwrap().push((
                ctx.tier_instance_id.to_string(),
                ctx.tier_id.to_string(),
                ctx.model_id.map(str::to_string),
                ctx.tasks.to_vec(),
            ));
            let readings = ctx.tasks.iter().map(|task| {
                let reading = self.answers.iter().find(|(id, _)| id == task)
                    .map(|(_, reading)| reading.clone())
                    .unwrap_or(TierReading::Unsupported);
                (task.clone(), reading)
            }).collect();
            TierOutput { model_id: self.output_model.clone(), readings }
        }
    }

    fn fake_instance(
        instance: &str,
        model_id: &str,
        answers: Vec<(TaskId, TierReading)>,
        calls: &CallLog,
    ) -> TierInstance {
        let tier_id = tid("grpc");
        let executor: Arc<dyn Tier> = Arc::new(FakeTier {
            id: tier_id.clone(),
            output_model: format!("sidecar-{model_id}"),
            answers,
            calls: calls.clone(),
        });
        TierInstance::new(iid(instance), tier_id, Some(model_id.into()), executor).unwrap()
    }

    fn scored(label: &str, confidence: f32) -> TierReading {
        TierReading::Scored(Reading::new(label, confidence))
    }

    fn profile(task: &str, kind: OutputKind, labels: Vec<&str>) -> TaskProfile {
        TaskProfile {
            task: TaskId::new(task),
            output_kind: kind,
            labels: if labels.is_empty() { None } else { Some(labels.into_iter().map(str::to_string).collect()) },
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        }
    }

    fn policy_map(instance: &str, task: &str, threshold: f32) -> TierPolicyMap {
        [(
            iid(instance),
            [(TaskId::new(task), TierPolicy::new(threshold))].into(),
        )].into()
    }

    #[tokio::test]
    async fn two_instances_share_a_definition_but_keep_models_and_results_distinct() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let instances = vec![
            fake_instance("encoder_blue", "modernbert-base", vec![(TaskId::new("risk"), scored("low", 0.70))], &calls),
            fake_instance("encoder_green", "harrier-v1", vec![(TaskId::new("risk"), scored("high", 0.93))], &calls),
        ];
        let runner = CascadeRunner::new(TierRegistry::of(instances));
        let profile = profile("risk", OutputKind::Multiclass, vec!["low", "high"]);
        let policies: TierPolicyMap = [
            (iid("encoder_blue"), [(profile.task.clone(), TierPolicy::new(0.99))].into()),
            (iid("encoder_green"), [(profile.task.clone(), TierPolicy::new(0.90))].into()),
        ].into();

        let outcome = runner.run("prompt", &[profile], &policies, Duration::from_secs(1)).await;
        assert_eq!(calls.lock().unwrap().iter().map(|call| call.0.as_str()).collect::<Vec<_>>(),
            ["encoder_blue", "encoder_green"]);
        assert_eq!(outcome.tasks[&TaskId::new("risk")].resolution, Resolution::Tier {
            tier_instance_id: iid("encoder_green"),
            tier_id: tid("grpc"),
            model_id: "harrier-v1".into(),
            reading: Reading::new("high", 0.93),
        });
        assert_eq!(outcome.tier_calls[0].model_id, "modernbert-base");
        assert_eq!(outcome.tier_calls[1].model_id, "harrier-v1");
        let calls = calls.lock().unwrap();
        assert_eq!(calls[0].2.as_deref(), Some("modernbert-base"));
        assert_eq!(calls[1].2.as_deref(), Some("harrier-v1"));
        assert_eq!(outcome.tier_calls[0].tier_id, outcome.tier_calls[1].tier_id);
    }

    #[tokio::test]
    async fn a_policy_targets_one_instance_without_leaking_to_its_sibling() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = CascadeRunner::new(TierRegistry::of(vec![
            fake_instance("encoder_blue", "model-a", vec![(TaskId::new("risk"), scored("high", 0.95))], &calls),
            fake_instance("encoder_green", "model-b", vec![(TaskId::new("risk"), scored("high", 0.95))], &calls),
        ]));
        let profile = profile("risk", OutputKind::Multiclass, vec!["high"]);
        let policies = policy_map("encoder_green", "risk", 0.90);

        let outcome = runner.run("prompt", &[profile], &policies, Duration::from_secs(1)).await;
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "encoder_green");
        assert!(matches!(
            outcome.tasks[&TaskId::new("risk")].resolution,
            Resolution::Tier { ref tier_instance_id, .. } if tier_instance_id == &iid("encoder_green")
        ));
    }

    #[tokio::test]
    async fn instance_order_is_vector_order_not_name_or_definition_order() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = CascadeRunner::new(TierRegistry::of(vec![
            fake_instance("zeta", "model-z", vec![], &calls),
            fake_instance("alpha", "model-a", vec![], &calls),
            fake_instance("middle", "model-m", vec![(TaskId::route(), scored(LABEL_JEV, 0.99))], &calls),
        ]));
        let policies: TierPolicyMap = ["zeta", "alpha", "middle"].into_iter()
            .map(|id| (iid(id), [(TaskId::route(), TierPolicy::new(0.90))].into()))
            .collect();
        let outcome = runner.run("prompt", &[profile("route", OutputKind::Binary, vec![LABEL_JEV, LABEL_LLM])],
            &policies, Duration::from_secs(1)).await;

        assert_eq!(calls.lock().unwrap().iter().map(|call| call.0.as_str()).collect::<Vec<_>>(),
            ["zeta", "alpha", "middle"]);
        assert!(matches!(
            outcome.tasks[&TaskId::route()].resolution,
            Resolution::Tier { ref tier_instance_id, .. } if tier_instance_id == &iid("middle")
        ));
    }

    #[tokio::test]
    async fn one_instance_handles_multiple_arbitrary_tasks_in_one_call() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = CascadeRunner::new(TierRegistry::of(vec![
            fake_instance("shared_encoder", "model-shared", vec![
                (TaskId::new("customer_intent"), scored("support", 0.94)),
                (TaskId::new("urgency"), scored("high", 0.91)),
            ], &calls),
        ]));
        let profiles = vec![
            profile("customer_intent", OutputKind::Multiclass, vec!["support"]),
            profile("urgency", OutputKind::Ordinal, vec!["high"]),
        ];
        let policies: TierPolicyMap = [(
            iid("shared_encoder"),
            profiles.iter().map(|p| (p.task.clone(), TierPolicy::new(0.80))).collect(),
        )].into();

        let outcome = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        assert_eq!(calls.lock().unwrap().len(), 1, "the shared encoder should be invoked once");
        assert_eq!(outcome.tasks.len(), 2);
        assert!(matches!(outcome.tasks[&TaskId::new("customer_intent")].resolution, Resolution::Tier { .. }));
        assert!(matches!(outcome.tasks[&TaskId::new("urgency")].resolution, Resolution::Tier { .. }));
        assert_eq!(outcome.tier_calls[0].tasks.len(), 2);
    }

    #[tokio::test]
    async fn required_tasks_publish_before_best_effort_instances_finish() {
        struct Delayed {
            id: TierId,
            delay: Duration,
            label: &'static str,
        }
        #[async_trait]
        impl Tier for Delayed {
            fn id(&self) -> &TierId { &self.id }
            fn name(&self) -> &str { self.id.as_str() }
            async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
                tokio::time::sleep(self.delay).await;
                TierOutput {
                    model_id: "delayed".into(),
                    readings: ctx.tasks.iter().map(|task| (
                        task.clone(),
                        TierReading::Scored(Reading::new(self.label, 0.99)),
                    )).collect(),
                }
            }
        }
        let calls = Arc::new(Mutex::new(Vec::new()));
        let make = |name: &str, delay, label| {
            let tier_id = tid("grpc");
            let executor: Arc<dyn Tier> = Arc::new(Delayed { id: tier_id.clone(), delay, label });
            TierInstance::new(iid(name), tier_id, Some(name.into()), executor).unwrap()
        };
        let runner = Arc::new(CascadeRunner::new(TierRegistry::of(vec![
            make("fast", Duration::from_millis(5), LABEL_JEV),
            make("slow", Duration::from_millis(250), "help_desk"),
        ])));
        let profiles = vec![
            profile("route", OutputKind::Binary, vec![LABEL_JEV, LABEL_LLM]),
            profile("tag", OutputKind::Multiclass, vec!["help_desk"]),
        ];
        let policies: TierPolicyMap = [
            (iid("fast"), [(TaskId::route(), TierPolicy::new(0.90))].into()),
            (iid("slow"), [(TaskId::new("tag"), TierPolicy::new(0.80))].into()),
        ].into();
        let split = runner.run_split("prompt".into(), profiles, [TaskId::route()].into(),
            Arc::new(policies), Duration::from_secs(2));
        let required = tokio::time::timeout(Duration::from_millis(100), split.required)
            .await.expect("required result should be published after fast instance")
            .expect("required sender should remain alive");
        assert!(matches!(required.tasks[&TaskId::route()].resolution, Resolution::Tier { .. }));
        let complete = split.done.await.unwrap();
        assert!(matches!(complete.tasks[&TaskId::new("tag")].resolution, Resolution::Tier { .. }));
    }
}
