// src/pipeline/runner.rs
//
// Generic shared-cascade execution. TierRegistry catalogs available instances;
// Cascade supplies the single execution order; tier/task policies determine
// which tasks participate at each position.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::debug;

use super::cascade::Cascade;
use super::registry::TierRegistry;
use super::task::{Decision, FallbackPolicy, Reading, Task, TierPolicyMap, TierReading};
use super::tier::{TierContext, TierId, TierInstance, TierName};

#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    Tier {
        tier_id: TierId,
        tier: TierName,
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
    pub tier_id: TierId,
    pub tier: TierName,
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
    pub tier_id: TierId,
    pub tier: TierName,
    pub tasks: Vec<String>,
    pub took: Duration,
}

#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub tasks: BTreeMap<String, TaskOutcome>,
    pub tier_calls: Vec<TierCall>,
}

#[derive(Debug)]
pub enum CascadeRunnerBuildError {
    MissingTier(TierId),
}

impl std::fmt::Display for CascadeRunnerBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTier(id) => write!(f, "cascade references unavailable tier {id}"),
        }
    }
}

impl std::error::Error for CascadeRunnerBuildError {}

pub struct CascadeRunner {
    /// Complete catalog of available tier instances, including unused entries.
    registry: TierRegistry,
    /// Implementations used by the cascade, keyed by Cascade-assigned TierId.
    tiers: BTreeMap<TierId, Arc<TierInstance>>,
    cascade: Arc<Cascade>,
}

/// Required-task completion and full cascade completion are separate signals.
pub struct SplitRun {
    pub required: oneshot::Receiver<RunOutcome>,
    pub done: JoinHandle<RunOutcome>,
}

impl CascadeRunner {
    /// Bind the available registry to one validated ordered cascade.
    pub fn new(
        registry: TierRegistry,
        cascade: Cascade,
    ) -> Result<Self, CascadeRunnerBuildError> {
        let mut tiers = BTreeMap::new();
        for id in cascade.order() {
            let name = cascade.tier_name(id)
                .ok_or_else(|| CascadeRunnerBuildError::MissingTier(*id))?;
            let tier = registry.get(name)
                .ok_or_else(|| CascadeRunnerBuildError::MissingTier(*id))?;
            let actual = TierInstance::derive_id(
                tier.name(), tier.tier(), tier.params(), tier.artifact_hash(),
            );
            if actual != *id {
                return Err(CascadeRunnerBuildError::MissingTier(*id));
            }
            tiers.insert(*id, tier);
        }
        Ok(Self { registry, tiers, cascade: Arc::new(cascade) })
    }

    /// Complete inventory of available instances, including unused entries.
    pub fn registry(&self) -> &TierRegistry {
        &self.registry
    }

    /// The immutable shared cascade.
    pub fn cascade(&self) -> Arc<Cascade> {
        Arc::clone(&self.cascade)
    }

    pub async fn run(
        &self,
        prompt: &str,
        profiles: &[Task],
        policies: &TierPolicyMap,
        budget: Duration,
    ) -> RunOutcome {
        self.run_inner(prompt, profiles, policies, budget, None).await
    }

    /// Starts one shared-cascade run. The required result is published as soon
    /// as every required task resolves or exhausts its policy-bearing tiers.
    pub fn run_split(
        self: Arc<Self>,
        prompt: String,
        profiles: Vec<Task>,
        required_tasks: BTreeSet<String>,
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
        profiles: &[Task],
        policies: &TierPolicyMap,
        budget: Duration,
        mut required_tx: Option<(BTreeSet<String>, oneshot::Sender<RunOutcome>)>,
    ) -> RunOutcome {
        let start = Instant::now();
        let profile_order: Vec<String> = profiles.iter()
            .map(|profile| profile.name.clone())
            .collect();
        let profile_by_name: BTreeMap<&str, &Task> = profiles.iter()
            .map(|profile| (profile.name.as_str(), profile))
            .collect();
        debug_assert_eq!(
            profile_by_name.len(),
            profiles.len(),
            "duplicate task names in profiles",
        );

        // A task participates only at tiers where a corresponding policy exists.
        // Tasks with no policy anywhere immediately take their configured fallback.
        let last_policy_index: BTreeMap<String, usize> = profiles.iter()
            .filter_map(|profile| {
                self.cascade.order().iter().enumerate()
                    .filter(|(_, tier_id)| {
                        policies.get(tier_id)
                            .is_some_and(|by_task| by_task.contains_key(&profile.name))
                    })
                    .map(|(index, _)| (profile.name.clone(), index))
                    .last()
            })
            .collect();
        let mut active: BTreeSet<String> = last_policy_index.keys().cloned().collect();

        let mut resolved: BTreeMap<String, Resolution> = BTreeMap::new();
        let mut traces: BTreeMap<String, Vec<TierTrace>> = BTreeMap::new();
        let mut tier_calls = Vec::new();

        if required_tx.as_ref().is_some_and(|(required, _)| {
            required.iter().all(|task| !active.contains(task))
        }) {
            if let Some((_, sender)) = required_tx.take() {
                let _ = sender.send(build_outcome(profiles, &resolved, &traces, &tier_calls));
            }
        }

        for (tier_index, tier_id) in self.cascade.order().iter().copied().enumerate() {
            if active.is_empty() || start.elapsed() >= budget {
                break;
            }

            let task_ids: Vec<String> = profile_order.iter()
                .filter(|task| {
                    active.contains(*task)
                        && policies.get(&tier_id)
                            .is_some_and(|by_task| by_task.contains_key(*task))
                })
                .cloned()
                .collect();
            if task_ids.is_empty() {
                continue;
            }

            let Some(tier) = self.tiers.get(&tier_id).cloned() else {
                for task in task_ids {
                    active.remove(&task);
                }
                continue;
            };
            let tier_name = tier.name().clone();
            let remaining = budget.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                break;
            }

            let call_started = Instant::now();
            let evaluation = tokio::time::timeout(
                remaining,
                tier.evaluate(&TierContext {
                    prompt,
                    tasks: &task_ids,
                    task_profiles: profiles,
                    budget: remaining,
                }),
            ).await;
            let output = match evaluation {
                Ok(output) => output,
                Err(_) => break,
            };
            tier_calls.push(TierCall {
                tier_id,
                tier: tier_name.clone(),
                tasks: task_ids.clone(),
                took: call_started.elapsed(),
            });

            let by_task = policies.get(&tier_id)
                .expect("tasks were selected only when this tier has a policy");
            for task_name in task_ids {
                if !active.contains(&task_name) {
                    continue;
                }
                let Some(profile) = profile_by_name.get(task_name.as_str()).copied() else {
                    active.remove(&task_name);
                    continue;
                };
                let policy = by_task.get(&task_name)
                    .expect("task batch contains only tasks with policies at this tier");
                let reading = output.readings.get(&task_name)
                    .cloned()
                    .unwrap_or(TierReading::Unsupported);
                let decision = profile.decide(policy, &reading);
                debug!(task = %task_name, tier = %tier_name, ?decision, "task decision");

                if let Decision::Resolve(resolved_reading) = &decision {
                    resolved.insert(task_name.clone(), Resolution::Tier {
                        tier_id,
                        tier: tier_name.clone(),
                        model_id: output.model_id.clone(),
                        reading: resolved_reading.clone(),
                    });
                    active.remove(&task_name);
                } else if last_policy_index.get(&task_name) == Some(&tier_index) {
                    active.remove(&task_name);
                }

                traces.entry(task_name.clone()).or_default().push(TierTrace {
                    tier_id,
                    tier: tier_name.clone(),
                    model_id: output.model_id.clone(),
                    reading,
                    decision,
                });
            }

            if required_tx.as_ref().is_some_and(|(required, _)| {
                required.iter().all(|task| !active.contains(task))
            }) {
                if let Some((_, sender)) = required_tx.take() {
                    let _ = sender.send(build_outcome(profiles, &resolved, &traces, &tier_calls));
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
    profiles: &[Task],
    resolved: &BTreeMap<String, Resolution>,
    traces: &BTreeMap<String, Vec<TierTrace>>,
    tier_calls: &[TierCall],
) -> RunOutcome {
    let tasks = profiles.iter().map(|profile| {
        let resolution = resolved.get(&profile.name).cloned().unwrap_or_else(|| {
            match &profile.fallback {
                FallbackPolicy::Label(label) => Resolution::Fallback { label: label.clone() },
                FallbackPolicy::Defer => Resolution::Deferred,
            }
        });
        let trace = traces.get(&profile.name).cloned().unwrap_or_default();
        (profile.name.clone(), TaskOutcome { resolution, trace })
    }).collect();
    RunOutcome { tasks, tier_calls: tier_calls.to_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::cascade::Slot;
    use crate::pipeline::registry::TierRegistry;
    use crate::pipeline::task::{OutputKind, TierPolicy, LABEL_JEV, LABEL_LLM};
    use crate::pipeline::tier::{tname, FakeEvaluate, TierOutput};
    use async_trait::async_trait;
    use std::sync::Mutex;

    type TierCallLog = Arc<Mutex<Vec<(String, Vec<String>)>>>;

    struct IdentityOnly;

    #[async_trait]
    impl FakeEvaluate for IdentityOnly {
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            TierOutput::abstain_all("identity-only", ctx.tasks, "test")
        }
    }

    fn tier_id(name: &str) -> TierId {
        let instance = TierInstance::fake(name, Arc::new(IdentityOnly));
        TierInstance::derive_id(
            instance.name(), instance.tier(), instance.params(), instance.artifact_hash(),
        )
    }

    fn test_runner(tiers: Vec<Arc<TierInstance>>, order: &[&str]) -> CascadeRunner {
        let registry = TierRegistry::of(tiers);
        let names = order.iter().map(|name| tname(name)).collect::<Vec<_>>();
        let cascade = Cascade::from_names(&names, &registry).unwrap();
        CascadeRunner::new(registry, cascade).unwrap()
    }

    struct FakeTier {
        name: String,
        answers: Vec<(String, TierReading)>,
        calls: TierCallLog,
    }

    impl FakeTier {
        fn new(
            name: &str,
            answers: Vec<(String, TierReading)>,
            calls: &TierCallLog,
        ) -> TierInstance {
            TierInstance::fake(name, Arc::new(Self {
                name: name.to_string(),
                answers,
                calls: calls.clone(),
            }))
        }
    }

    #[async_trait]
    impl FakeEvaluate for FakeTier {
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            self.calls.lock().unwrap().push((self.name.clone(), ctx.tasks.to_vec()));
            let readings = ctx.tasks.iter().map(|task| {
                let reading = self.answers.iter()
                    .find(|(id, _)| id == task)
                    .map(|(_, reading)| reading.clone())
                    .unwrap_or(TierReading::Abstain("unsupported_task"));
                (task.clone(), reading)
            }).collect();
            TierOutput { model_id: format!("fake-{}", self.name), readings }
        }
    }

    struct DelayedTier {
        name: String,
        delay: Duration,
    }

    impl DelayedTier {
        fn new(name: &str, delay: Duration) -> TierInstance {
            TierInstance::fake(name, Arc::new(Self { name: name.to_string(), delay }))
        }
    }

    #[async_trait]
    impl FakeEvaluate for DelayedTier {
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            tokio::time::sleep(self.delay).await;
            let readings = ctx.tasks.iter().map(|task| {
                let reading = if task.as_str() == "route" {
                    scored(LABEL_JEV, 0.99)
                } else {
                    scored("help_desk", 0.92)
                };
                (task.clone(), reading)
            }).collect();
            TierOutput { model_id: format!("delayed-{}", self.name), readings }
        }
    }

    fn scored(label: &str, confidence: f32) -> TierReading {
        TierReading::Scored(Reading::new(label, confidence))
    }

    fn route_profile() -> Task {
        Task {
            name: String::from("route"),
            output_kind: OutputKind::Binary,
            labels: Some(vec![LABEL_JEV.into(), LABEL_LLM.into()]),
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Label(LABEL_LLM.into()),
        }
    }

    fn tag_profile(labels: Option<Vec<String>>) -> Task {
        Task {
            name: String::from("tag"),
            output_kind: OutputKind::Multiclass,
            labels,
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        }
    }

    fn resolved_at(resolution: &Resolution, name: &str) -> bool {
        matches!(resolution, Resolution::Tier { tier, .. } if tier.as_str() == name)
    }

    #[tokio::test]
    async fn separate_tasks_share_a_tier_call_and_resolve_independently() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = test_runner(
            vec![Arc::new(FakeTier::new(
                "modernbert",
                vec![
                    (String::from("route"), scored(LABEL_JEV, 0.97)),
                    (String::from("tag"), scored("engineering", 0.85)),
                ],
                &calls,
            ))],
            &["modernbert"],
        );
        let profiles = vec![
            route_profile(),
            tag_profile(Some(vec!["engineering".into(), "help_desk".into()])),
        ];
        let policies: TierPolicyMap = [(
            tier_id("modernbert"),
            [
                (String::from("route"), TierPolicy::new(0.95).with_label(LABEL_LLM, 0.70)),
                (String::from("tag"), TierPolicy::new(0.80)),
            ].into(),
        )].into();

        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.len(), 2);
        assert!(resolved_at(&result.tasks["route"].resolution, "modernbert"));
        assert!(resolved_at(&result.tasks["tag"].resolution, "modernbert"));
    }

    #[tokio::test]
    async fn required_route_is_published_before_best_effort_tier_finishes() {
        let runner = Arc::new(test_runner(vec![
            Arc::new(DelayedTier::new("rules", Duration::from_millis(5))),
            Arc::new(DelayedTier::new("modernbert", Duration::from_millis(250))),
        ], &["rules", "modernbert"]));
        let profiles = vec![route_profile(), tag_profile(Some(vec!["help_desk".into()]))];
        let policies: TierPolicyMap = [
            (tier_id("rules"), [(String::from("route"), TierPolicy::new(0.9))].into()),
            (tier_id("modernbert"), [(String::from("tag"), TierPolicy::new(0.8))].into()),
        ].into();
        let split = runner.run_split(
            "prompt".into(),
            profiles,
            [String::from("route")].into(),
            Arc::new(policies),
            Duration::from_secs(2),
        );
        let required = tokio::time::timeout(Duration::from_millis(100), split.required)
            .await.expect("route must resolve before the modernbert tier finishes")
            .expect("required result sender must remain alive");
        assert!(resolved_at(&required.tasks["route"].resolution, "rules"));
        assert_eq!(required.tasks["tag"].resolution, Resolution::Deferred);
        let complete = split.done.await.unwrap();
        assert!(resolved_at(&complete.tasks["tag"].resolution, "modernbert"));
        assert!(resolved_at(&complete.tasks["route"].resolution, "rules"));
    }

    #[tokio::test]
    async fn tasks_without_a_policy_at_a_tier_are_not_sent_to_that_tier() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = test_runner(vec![Arc::new(FakeTier::new(
            "modernbert",
            vec![(String::from("tag"), scored("engineering", 0.99))],
            &calls,
        ))], &["modernbert"]);
        let result = runner.run(
            "prompt",
            &[tag_profile(None)],
            &TierPolicyMap::new(),
            Duration::from_secs(1),
        ).await;
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(result.tasks["tag"].resolution, Resolution::Deferred);
    }

    #[tokio::test]
    async fn tiers_run_in_the_authored_cascade_order_regardless_of_registry_names() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = test_runner(vec![
            Arc::new(FakeTier::new("zulu", vec![], &calls)),
            Arc::new(FakeTier::new("alpha", vec![], &calls)),
            Arc::new(FakeTier::new("mike", vec![
                (String::from("route"), scored(LABEL_JEV, 0.99)),
            ], &calls)),
        ], &["zulu", "alpha", "mike"]);
        let policies: TierPolicyMap = ["zulu", "alpha", "mike"].into_iter()
            .map(|name| (tier_id(name), [(
                String::from("route"), TierPolicy::new(0.9),
            )].into()))
            .collect();

        let result = runner.run("prompt", &[route_profile()], &policies, Duration::from_secs(1)).await;
        let order: Vec<String> = calls.lock().unwrap().iter()
            .map(|(name, _)| name.clone())
            .collect();
        assert_eq!(order, ["zulu", "alpha", "mike"]);
        assert!(resolved_at(&result.tasks["route"].resolution, "mike"));
        let called: Vec<String> = result.tier_calls.iter()
            .map(|call| call.tier.to_string())
            .collect();
        assert_eq!(called, ["zulu", "alpha", "mike"]);
    }

    #[tokio::test]
    async fn inserting_a_tier_changes_the_shared_order_without_changing_task_contracts() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let policies: TierPolicyMap = [
            (tier_id("fasttext"), [(
                String::from("route"), TierPolicy::new(0.95),
            )].into()),
            (tier_id("prefilter"), [(
                String::from("route"), TierPolicy::new(0.80),
            )].into()),
            (tier_id("modernbert"), [(
                String::from("route"), TierPolicy::new(0.95),
            )].into()),
        ].into();

        let before = test_runner(vec![
            Arc::new(FakeTier::new("fasttext", vec![
                (String::from("route"), scored(LABEL_JEV, 0.60)),
            ], &calls)),
            Arc::new(FakeTier::new("modernbert", vec![
                (String::from("route"), scored(LABEL_JEV, 0.99)),
            ], &calls)),
        ], &["fasttext", "modernbert"]);
        let outcome = before.run("p", &[route_profile()], &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&outcome.tasks["route"].resolution, "modernbert"));

        calls.lock().unwrap().clear();
        let after = test_runner(vec![
            Arc::new(FakeTier::new("fasttext", vec![
                (String::from("route"), scored(LABEL_JEV, 0.60)),
            ], &calls)),
            Arc::new(FakeTier::new("prefilter", vec![
                (String::from("route"), scored(LABEL_JEV, 0.85)),
            ], &calls)),
            Arc::new(FakeTier::new("modernbert", vec![
                (String::from("route"), scored(LABEL_JEV, 0.99)),
            ], &calls)),
        ], &["fasttext", "prefilter", "modernbert"]);
        let outcome = after.run("p", &[route_profile()], &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&outcome.tasks["route"].resolution, "prefilter"));
        let order: Vec<String> = calls.lock().unwrap().iter()
            .map(|(name, _)| name.clone())
            .collect();
        assert_eq!(order, ["fasttext", "prefilter"]);
        assert_eq!(after.cascade().slot(&tier_id("prefilter")).map(Slot::index), Some(1));
    }

    #[tokio::test]
    async fn registry_retains_available_tiers_absent_from_the_cascade() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registry = TierRegistry::of(vec![
            Arc::new(FakeTier::new("rules", vec![], &calls)),
            Arc::new(FakeTier::new("modernbert", vec![], &calls)),
            Arc::new(FakeTier::new("harrier", vec![], &calls)),
        ]);
        let names = [tname("rules"), tname("modernbert")];
        let cascade = Cascade::from_names(&names, &registry).unwrap();
        let runner = CascadeRunner::new(registry, cascade).unwrap();
        assert!(runner.registry().get(&tname("harrier")).is_some());
        assert_eq!(runner.cascade().slot(&tier_id("harrier")).map(Slot::index), None);
    }

    #[tokio::test]
    async fn a_policy_for_a_tier_outside_the_cascade_is_never_executed() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = test_runner(vec![Arc::new(FakeTier::new(
            "rules",
            vec![(String::from("route"), scored(LABEL_JEV, 0.99))],
            &calls,
        ))], &["rules"]);
        let policies: TierPolicyMap = [
            (tier_id("rules"), [(
                String::from("route"), TierPolicy::new(0.95),
            )].into()),
            (tier_id("ghost"), [(
                String::from("route"), TierPolicy::new(0.10),
            )].into()),
        ].into();
        let outcome = runner.run("p", &[route_profile()], &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&outcome.tasks["route"].resolution, "rules"));
        assert_eq!(calls.lock().unwrap().len(), 1);
    }
}
