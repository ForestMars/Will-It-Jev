// src/pipeline/runner.rs
//
// Generic cascade execution. The runner knows tiers and task IDs, but task
// definitions and tier/task resolution policies are supplied independently.
// Tier instances run in the registry's configured vector order; names never decide
// order.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use tracing::debug;

use super::registry::{TierDirectory, TierId, TierInstance, TierInstanceId, TierRegistry};
use super::task::{
    Decision, FallbackPolicy, Reading, TaskId, TaskProfile, TierPolicyMap, TierReading,
};
use super::tier::{Tier, TierContext};

#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    Tier {
        tier_instance_id: TierInstanceId,
        tier_id: TierId,
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
    order: Vec<TierInstanceId>,
    instances: BTreeMap<TierInstanceId, TierInstance>,
    directory: Arc<TierDirectory>,
}

/// Required-task completion and full cascade completion are separate signals from one run.
pub struct SplitRun {
    pub required: oneshot::Receiver<RunOutcome>,
    pub done: JoinHandle<RunOutcome>,
}

impl CascadeRunner {
    /// Tier instances execute in registry order.
    pub fn new(registry: TierRegistry) -> Self {
        let (order, instances, directory) = registry.into_parts();
        Self { order, instances, directory: Arc::new(directory) }
    }

    /// Instance names, architecture IDs, and vector positions.
    pub fn directory(&self) -> Arc<TierDirectory> {
        Arc::clone(&self.directory)
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

        for instance_id in &self.order {
            if active.is_empty() || start.elapsed() >= budget {
                break;
            }
            let instance = self.instances.get(instance_id).expect("ordered instance must be registered");
            let tier = &instance.executor;
            let by_task = policies.get(instance_id);
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
                tier_instance_id = %instance.id,
                tier_id = %instance.tier_id,
                model_id = %instance.model_id,
                tier_name = tier.name(),
                task_count = ids.len(),
                "executing inference tier instance"
            );
            let output = tier
                .evaluate(&TierContext {
                    prompt,
                    tasks: &ids,
                    task_profiles: profiles,
                    model_id: &instance.model_id,
                    budget: budget.saturating_sub(start.elapsed()),
                })
                .await;
            let model_id = if output.model_id.trim().is_empty() {
                instance.model_id.clone()
            } else {
                output.model_id.clone()
            };
            tier_calls.push(TierCall {
                tier_instance_id: instance.id.clone(),
                tier_id: instance.tier_id.clone(),
                model_id: model_id.clone(),
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
                debug!(task = %profile.task, tier_instance_id = %instance.id, tier_id = %instance.tier_id, ?decision, "task decision");

                if let Decision::Resolve(resolved_reading) = &decision {
                    resolved.insert(
                        profile.task.clone(),
                        Resolution::Tier {
                            tier_instance_id: instance.id.clone(),
                            tier_id: instance.tier_id.clone(),
                            model_id: model_id.clone(),
                            reading: resolved_reading.clone(),
                        },
                    );
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
    use crate::pipeline::registry::{iid, tid, TierDefinition, TierDefinitionRegistry, TierInstance, TierKind};
    use crate::pipeline::task::{OutputKind, TierPolicy, LABEL_JEV, LABEL_LLM};
    use crate::pipeline::tier::TierOutput;
    use async_trait::async_trait;
    use std::sync::Mutex;

    type TierCallLog = Arc<Mutex<Vec<(String, Vec<TaskId>)>>>;

    struct FakeTier {
        id: TierId,
        answers: Vec<(TaskId, TierReading)>,
        calls: TierCallLog,
    }

    impl FakeTier {
        fn new(name: &str, answers: Vec<(TaskId, TierReading)>, calls: &TierCallLog) -> Self {
            Self { id: tid(name), answers, calls: calls.clone() }
        }
    }

    #[async_trait]
    impl Tier for FakeTier {
        fn id(&self) -> &TierId { &self.id }
        fn name(&self) -> &str { "fake" }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            self.calls.lock().unwrap().push((self.id.to_string(), ctx.tasks.to_vec()));
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

    fn route_profile() -> TaskProfile {
        TaskProfile {
            task: TaskId::route(),
            output_kind: OutputKind::Binary,
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Label(LABEL_LLM.into()),
        }
    }

    fn tag_profile(labels: Option<Vec<String>>) -> TaskProfile {
        TaskProfile {
            task: TaskId::tag(),
            output_kind: OutputKind::Multiclass,
            labels,
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        }
    }

    fn resolved_at(resolution: &Resolution, name: &str) -> bool {
        matches!(resolution, Resolution::Tier { tier_instance_id, .. } if tier_instance_id.as_str() == name)
    }

    #[tokio::test]
    async fn separate_tasks_share_a_tier_call_and_resolve_independently() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tier = FakeTier::new(
            "modernbert",
            vec![
                (TaskId::route(), scored(LABEL_JEV, 0.97)),
                (TaskId::tag(), scored("engineering", 0.85)),
            ],
            &calls,
        );
        let runner = CascadeRunner::new(TierRegistry::of(vec![Arc::new(tier)]));
        let profiles = vec![
            route_profile(),
            tag_profile(Some(vec!["engineering".into(), "help_desk".into()])),
        ];
        let mut policies = TierPolicyMap::new();
        let mut by_task = BTreeMap::new();
        by_task.insert(TaskId::route(), TierPolicy::new(0.95).with_label(LABEL_LLM, 0.70));
        by_task.insert(TaskId::tag(), TierPolicy::new(0.80));
        policies.insert(iid("modernbert"), by_task);

        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.len(), 2);
        assert!(resolved_at(&result.tasks[&TaskId::route()].resolution, "modernbert"));
        assert!(resolved_at(&result.tasks[&TaskId::tag()].resolution, "modernbert"));
    }

    struct DelayedTier {
        id: TierId,
        delay: Duration,
    }

    #[async_trait]
    impl Tier for DelayedTier {
        fn id(&self) -> &TierId { &self.id }
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
        let runner = Arc::new(CascadeRunner::new(TierRegistry::of(vec![
            Arc::new(DelayedTier { id: tid("rules"), delay: Duration::from_millis(5) }),
            Arc::new(DelayedTier { id: tid("modernbert"), delay: Duration::from_millis(250) }),
        ])));
        let profiles = vec![route_profile(), tag_profile(Some(vec!["help_desk".into()]))];
        let policies: TierPolicyMap = [
            (iid("rules"), [(TaskId::route(), TierPolicy::new(0.9))].into()),
            (iid("modernbert"), [(TaskId::tag(), TierPolicy::new(0.8))].into()),
        ].into();
        let split = runner.run_split(
            "prompt".into(),
            profiles,
            [TaskId::route()].into(),
            Arc::new(policies),
            Duration::from_secs(2),
        );
        let required = tokio::time::timeout(Duration::from_millis(100), split.required)
            .await.expect("route must resolve before the modernbert tier finishes")
            .expect("required result sender must remain alive");
        assert!(resolved_at(&required.tasks[&TaskId::route()].resolution, "rules"));
        assert_eq!(required.tasks[&TaskId::tag()].resolution, Resolution::Deferred);
        let complete = split.done.await.unwrap();
        assert!(resolved_at(&complete.tasks[&TaskId::tag()].resolution, "modernbert"));
        assert!(resolved_at(&complete.tasks[&TaskId::route()].resolution, "rules"));
    }

    #[tokio::test]
    async fn task_without_a_policy_at_a_tier_is_not_sent_to_that_tier() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tier = FakeTier::new(
            "modernbert",
            vec![(TaskId::tag(), scored("engineering", 0.99))],
            &calls,
        );
        let runner = CascadeRunner::new(TierRegistry::of(vec![Arc::new(tier)]));
        let profiles = vec![tag_profile(None)];
        let policies = TierPolicyMap::new();

        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(result.tasks[&TaskId::tag()].resolution, Resolution::Deferred);
    }

    /// Order comes from the registry. Names that sort differently from their
    /// execution position must not change it.
    #[tokio::test]
    async fn tiers_run_in_registry_order_regardless_of_their_names() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = CascadeRunner::new(TierRegistry::of(vec![
            Arc::new(FakeTier::new("zulu", vec![], &calls)),
            Arc::new(FakeTier::new("alpha", vec![], &calls)),
            Arc::new(FakeTier::new("mike", vec![(TaskId::route(), scored(LABEL_JEV, 0.99))], &calls)),
        ]));
        let policies: TierPolicyMap = ["zulu", "alpha", "mike"]
            .into_iter()
            .map(|name| (iid(name), [(TaskId::route(), TierPolicy::new(0.9))].into()))
            .collect();

        let result = runner
            .run("prompt", &[route_profile()], &policies, Duration::from_secs(1))
            .await;
        let order: Vec<String> = calls.lock().unwrap().iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(order, ["zulu", "alpha", "mike"]);
        assert!(resolved_at(&result.tasks[&TaskId::route()].resolution, "mike"));
        let called: Vec<String> = result.tier_calls.iter().map(|call| call.tier_instance_id.to_string()).collect();
        assert_eq!(called, ["zulu", "alpha", "mike"]);
    }

    /// The point of the named registry: a new tier is added by registering it
    /// and giving it policies. Neither the runner nor any tier adapter changes.
    #[tokio::test]
    async fn a_new_named_tier_can_be_inserted_between_existing_tiers() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let policies: TierPolicyMap = [
            (iid("fasttext"), [(TaskId::route(), TierPolicy::new(0.95))].into()),
            (iid("triage"), [(TaskId::route(), TierPolicy::new(0.80))].into()),
            (iid("modernbert"), [(TaskId::route(), TierPolicy::new(0.95))].into()),
        ].into();

        let before = CascadeRunner::new(TierRegistry::of(vec![
            Arc::new(FakeTier::new("fasttext", vec![(TaskId::route(), scored(LABEL_JEV, 0.60))], &calls)),
            Arc::new(FakeTier::new("modernbert", vec![(TaskId::route(), scored(LABEL_JEV, 0.99))], &calls)),
        ]));
        let outcome = before.run("p", &[route_profile()], &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&outcome.tasks[&TaskId::route()].resolution, "modernbert"));

        calls.lock().unwrap().clear();
        let after = CascadeRunner::new(TierRegistry::of(vec![
            Arc::new(FakeTier::new("fasttext", vec![(TaskId::route(), scored(LABEL_JEV, 0.60))], &calls)),
            Arc::new(FakeTier::new("triage", vec![(TaskId::route(), scored(LABEL_JEV, 0.85))], &calls)),
            Arc::new(FakeTier::new("modernbert", vec![(TaskId::route(), scored(LABEL_JEV, 0.99))], &calls)),
        ]));
        let outcome = after.run("p", &[route_profile()], &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&outcome.tasks[&TaskId::route()].resolution, "triage"));
        let order: Vec<String> = calls.lock().unwrap().iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(order, ["fasttext", "triage"], "modernbert must not run once triage resolves");
        assert_eq!(after.directory().position(&iid("triage")), Some(1));
    }

    struct ModelInstanceTier {
        architecture: TierId,
        calls: Arc<Mutex<Vec<(String, Vec<TaskId>)>>>,
        readings: BTreeMap<TaskId, TierReading>,
    }

    #[async_trait]
    impl Tier for ModelInstanceTier {
        fn id(&self) -> &TierId { &self.architecture }
        fn name(&self) -> &str { "grpc" }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            self.calls.lock().unwrap().push((ctx.model_id.to_string(), ctx.tasks.to_vec()));
            let readings = ctx.tasks.iter().map(|task| (
                task.clone(),
                self.readings.get(task).cloned().unwrap_or(TierReading::Unsupported),
            )).collect();
            TierOutput { model_id: ctx.model_id.to_string(), readings }
        }
    }

    #[tokio::test]
    async fn two_instances_share_architecture_but_keep_models_order_and_policies_separate() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut definitions = TierDefinitionRegistry::new();
        definitions.register(TierDefinition::new(tid("encoder"), TierKind::Grpc)).unwrap();
        let mut registry = TierRegistry::new(definitions);
        let first = Arc::new(ModelInstanceTier {
            architecture: tid("encoder"),
            calls: calls.clone(),
            readings: [(TaskId::route(), scored(LABEL_JEV, 0.60))].into(),
        });
        let second = Arc::new(ModelInstanceTier {
            architecture: tid("encoder"),
            calls: calls.clone(),
            readings: [
                (TaskId::route(), scored(LABEL_JEV, 0.97)),
                (TaskId::tag(), scored("engineering", 0.93)),
            ].into(),
        });
        registry.register(TierInstance::new(iid("zulu_encoder"), tid("encoder"), "modernbert-base", first).unwrap()).unwrap();
        registry.register(TierInstance::new(iid("alpha_encoder"), tid("encoder"), "harrier-270m", second).unwrap()).unwrap();

        let runner = CascadeRunner::new(registry);
        let policies: TierPolicyMap = [
            (iid("zulu_encoder"), [(TaskId::route(), TierPolicy::new(0.90))].into()),
            (iid("alpha_encoder"), [
                (TaskId::route(), TierPolicy::new(0.90)),
                (TaskId::tag(), TierPolicy::new(0.80)),
            ].into()),
        ].into();
        let profiles = [route_profile(), tag_profile(Some(vec!["engineering".into()]))];
        let outcome = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;

        assert_eq!(calls.lock().unwrap().as_slice(), &[
            ("modernbert-base".into(), vec![TaskId::route()]),
            ("harrier-270m".into(), vec![TaskId::route(), TaskId::tag()]),
        ]);
        assert_eq!(outcome.tier_calls.iter().map(|call| call.tier_instance_id.as_str()).collect::<Vec<_>>(),
            ["zulu_encoder", "alpha_encoder"]);
        assert_eq!(outcome.tier_calls.iter().map(|call| call.tier_id.as_str()).collect::<Vec<_>>(),
            ["encoder", "encoder"]);
        assert_eq!(outcome.tier_calls.iter().map(|call| call.model_id.as_str()).collect::<Vec<_>>(),
            ["modernbert-base", "harrier-270m"]);
        assert!(matches!(
            &outcome.tasks[&TaskId::route()].resolution,
            Resolution::Tier { tier_instance_id, tier_id, model_id, .. }
                if tier_instance_id.as_str() == "alpha_encoder"
                    && tier_id.as_str() == "encoder"
                    && model_id == "harrier-270m"
        ));
        assert!(matches!(
            &outcome.tasks[&TaskId::tag()].resolution,
            Resolution::Tier { tier_instance_id, .. } if tier_instance_id.as_str() == "alpha_encoder"
        ));
    }

    #[tokio::test]
    async fn a_policy_for_an_unregistered_tier_is_never_executed() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = CascadeRunner::new(TierRegistry::of(vec![Arc::new(FakeTier::new(
            "rules",
            vec![(TaskId::route(), scored(LABEL_JEV, 0.99))],
            &calls,
        ))]));
        let policies: TierPolicyMap = [
            (iid("rules"), [(TaskId::route(), TierPolicy::new(0.95))].into()),
            (iid("ghost"), [(TaskId::route(), TierPolicy::new(0.10))].into()),
        ].into();
        let outcome = runner.run("p", &[route_profile()], &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&outcome.tasks[&TaskId::route()].resolution, "rules"));
        assert_eq!(calls.lock().unwrap().len(), 1);
    }
}
