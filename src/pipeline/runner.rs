// src/pipeline/runner.rs
//
// Generic cascade execution. TierRegistry catalogs available tier instances by
// name; CascadeGraph supplies task-specific dependencies and traversal order,
// and the runner turns that order into slots. Task definitions and tier/task
// resolution policies remain independent.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use tracing::debug;

use super::cascade::CascadeGraph;
use super::tier_types::{Slot, TierId, TierName};
use super::registry::TierRegistry;
use super::task::{
    Decision, FallbackPolicy, Reading, Task, TierPolicyMap, TierReading,
};
use super::tier::{TierContext, TierInstance};

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
pub enum CascadeRunnerBuildError { MissingTier(TierId) }

impl std::fmt::Display for CascadeRunnerBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::MissingTier(id) => write!(f, "cascade graph references unavailable tier {id}") }
    }
}
impl std::error::Error for CascadeRunnerBuildError {}

pub struct CascadeRunner {
    /// Complete catalog of available tiers, including tiers absent from this graph.
    registry: TierRegistry,
    /// Implementations for graph nodes, keyed by graph-assigned TierId.
    /// Execution order comes exclusively from CascadeGraph::order().
    tiers: BTreeMap<TierId, Arc<TierInstance>>,
    graph: Arc<CascadeGraph>,
}

/// Required-task completion and full cascade completion are separate signals from one run.
pub struct SplitRun {
    pub required: oneshot::Receiver<RunOutcome>,
    pub done: JoinHandle<RunOutcome>,
}

impl CascadeRunner {
    /// Bind the available TierRegistry to a validated task-cascade DAG.
    pub fn new(registry: TierRegistry, graph: CascadeGraph) -> Result<Self, CascadeRunnerBuildError> {
        let mut tiers = BTreeMap::new();
        for id in graph.order() {
            let name = graph.tier_name(id).ok_or_else(|| CascadeRunnerBuildError::MissingTier(*id))?;
            let tier = registry.get(name).ok_or_else(|| CascadeRunnerBuildError::MissingTier(*id))?;
            let actual = TierInstance::derive_id(tier.name(), tier.tier(), tier.params(), tier.artifact_hash());
            if actual != *id {
                return Err(CascadeRunnerBuildError::MissingTier(*id));
            }
            tiers.insert(*id, tier);
        }
        Ok(Self { registry, tiers, graph: Arc::new(graph) })
    }

    /// Complete registry of available tier instances, including unused entries.
    pub fn registry(&self) -> &TierRegistry {
        &self.registry
    }

    /// The immutable execution graph, including its derived slot projection.
    pub fn graph(&self) -> Arc<CascadeGraph> {
        Arc::clone(&self.graph)
    }

    #[allow(dead_code)]
    pub fn slot_of(&self, id: &TierId) -> Option<Slot> {
        self.graph.slot(id)
    }

    /// `profiles` describe tasks. `policies` independently determine which
    /// tier may resolve each task and the task's calibrated bar at that tier.
    pub async fn run(
        &self,
        prompt: &str,
        profiles: &[Task],
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
        use tokio::task::JoinSet;

        let start = Instant::now();
        let mut active: BTreeSet<String> = profiles.iter().map(|profile| profile.name.clone()).collect();
        debug_assert_eq!(active.len(), profiles.len(), "duplicate task names in profiles");

        let mut resolved: BTreeMap<String, Resolution> = BTreeMap::new();
        let mut traces: BTreeMap<String, Vec<TierTrace>> = BTreeMap::new();
        let mut tier_calls = Vec::new();
        let requested: BTreeSet<String> = active.clone();
        let plan = self.graph.plan(&requested);
        let profile_by_name: BTreeMap<&str, &Task> =
            profiles.iter().map(|profile| (profile.name.as_str(), profile)).collect();
        let profiles = Arc::new(profiles.to_vec());
        let prompt = Arc::<str>::from(prompt.to_owned());

        // Each task has its own next fallback transition. Independent task routes
        // may be in flight concurrently, even when their graph nodes share a later
        // tier. Tasks ready at the same tier are batched into one invocation.
        let mut ready: BTreeMap<TierId, BTreeSet<String>> = BTreeMap::new();
        for (task, entry) in &plan.task_entries {
            if active.contains(task) {
                ready.entry(entry.clone()).or_default().insert(task.clone());
            }
        }
        let mut inflight: JoinSet<(TierId, TierName, Vec<String>, Duration, super::tier::TierOutput)> = JoinSet::new();

        if required_tx.as_ref().is_some_and(|(required, _)| required.is_empty()) {
            if let Some((_, sender)) = required_tx.take() {
                let _ = sender.send(build_outcome(&profiles, &resolved, &traces, &tier_calls));
            }
        }

        loop {
            // Dispatch every tier with currently ready tasks. A task that reaches
            // the same tier later may form a subsequent batch; it never waits for
            // an unrelated predecessor from another task's route.
            let batches = std::mem::take(&mut ready);
            for (tier_id, task_set) in batches {
                if start.elapsed() >= budget { break; }
                let Some(tier) = self.tiers.get(&tier_id).cloned() else {
                    for task in task_set {
                        active.remove(&task);
                    }
                    continue;
                };
                let task_ids: Vec<String> = task_set.into_iter()
                    .filter(|task| active.contains(task))
                    .collect();
                if task_ids.is_empty() { continue; }
                let remaining = budget.saturating_sub(start.elapsed());
                let prompt = Arc::clone(&prompt);
                let all_profiles = Arc::clone(&profiles);
                let call_tier_id = tier_id;
                let call_tier_name = tier.name().clone();
                inflight.spawn(async move {
                    let call_started = Instant::now();
                    let output = tier.evaluate(&TierContext {
                        prompt: &prompt,
                        tasks: &task_ids,
                        task_profiles: &all_profiles,
                        budget: remaining,
                    }).await;
                    (call_tier_id, call_tier_name, task_ids, call_started.elapsed(), output)
                });
            }

            if inflight.is_empty() { break; }
            let remaining = budget.saturating_sub(start.elapsed());
            if remaining.is_zero() { inflight.abort_all(); break; }
            let Some(joined) = tokio::time::timeout(remaining, inflight.join_next()).await.ok().flatten() else {
                inflight.abort_all();
                break;
            };
            let mut completed = vec![joined];
            while let Some(next) = inflight.try_join_next() {
                completed.push(next);
            }

            for joined in completed {
                let (tier_id, tier_name, task_ids, took, output) = match joined {
                    Ok(result) => result,
                    Err(error) => {
                        tracing::error!(%error, "tier execution task failed");
                        continue;
                    }
                };
                tier_calls.push(TierCall { tier_id, tier: tier_name.clone(), tasks: task_ids.clone(), took });
                let by_task = policies.get(&tier_id);

                for task_name in task_ids {
                    if !active.contains(&task_name) { continue; }
                    let Some(profile) = profile_by_name.get(task_name.as_str()).copied() else {
                        active.remove(&task_name);
                        continue;
                    };
                    let reading = output.readings.get(&task_name).cloned().unwrap_or(TierReading::Unsupported);
                    let decision = match by_task.and_then(|map| map.get(&task_name)) {
                        Some(policy) => profile.decide(policy, &reading),
                        None => Decision::Pass { reason: "no_policy", top_guess: None, score: 0.0 },
                    };
                    debug!(task = %task_name, tier = %tier_name, ?decision, "task decision");

                    if let Decision::Resolve(resolved_reading) = &decision {
                        resolved.insert(task_name.clone(), Resolution::Tier {
                            tier_id,
                            tier: tier_name.clone(),
                            model_id: output.model_id.clone(),
                            reading: resolved_reading.clone(),
                        });
                        active.remove(&task_name);
                    } else if let Some(next) = self.graph.next(&task_name, &tier_id).copied() {
                        if plan.required_nodes.contains(&next) {
                            ready.entry(next).or_default().insert(task_name.clone());
                        }
                    } else {
                        // The task exhausted its declared cascade without resolving.
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
            }

            if let Some((required, _)) = required_tx.as_ref() {
                if required.iter().all(|task| resolved.contains_key(task) || !active.contains(task)) {
                    if let Some((_, sender)) = required_tx.take() {
                        let _ = sender.send(build_outcome(&profiles, &resolved, &traces, &tier_calls));
                    }
                }
            }

            if active.is_empty() || start.elapsed() >= budget {
                inflight.abort_all();
                break;
            }
        }

        let outcome = build_outcome(&profiles, &resolved, &traces, &tier_calls);
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
    use crate::pipeline::cascade::CascadeGraph;
    use crate::pipeline::tier_types::tname;
    use crate::pipeline::registry::TierRegistry;
    use crate::pipeline::task::{OutputKind, TierPolicy, LABEL_JEV, LABEL_LLM};
    use crate::pipeline::tier::{FakeEvaluate, TierOutput};
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
        TierInstance::derive_id(instance.name(), instance.tier(), instance.params(), instance.artifact_hash())
    }

    fn test_graph(registry: &TierRegistry, paths: &[(&str, &[&str])]) -> CascadeGraph {
        let paths: BTreeMap<String, Vec<TierName>> = paths.iter().map(|(task, names)| (
            (*task).to_string(),
            names.iter().map(|name| tname(name)).collect()
        )).collect();
        CascadeGraph::from_task_paths(paths, registry).unwrap()
    }

    fn test_runner(tiers: Vec<Arc<TierInstance>>, paths: &[(&str, &[&str])]) -> CascadeRunner {
        let registry = TierRegistry::of(tiers);
        let graph = test_graph(&registry, paths);
        CascadeRunner::new(registry, graph).unwrap()
    }


    struct FakeTier {
        name: String,
        answers: Vec<(String, TierReading)>,
        calls: TierCallLog,
    }

    impl FakeTier {
        fn new(name: &str, answers: Vec<(String, TierReading)>, calls: &TierCallLog) -> TierInstance {
            TierInstance::fake(name, Arc::new(Self { name: name.to_string(), answers, calls: calls.clone() }))
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

    fn scored(label: &str, confidence: f32) -> TierReading {
        TierReading::Scored(Reading::new(label, confidence))
    }

    fn route_profile() -> Task {
        Task {
            cascade: crate::pipeline::task::Cascade::default(),
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
            cascade: crate::pipeline::task::Cascade::default(),
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
        let tier = FakeTier::new(
            "modernbert",
            vec![
                (String::from("route"), scored(LABEL_JEV, 0.97)),
                (String::from("tag"), scored("engineering", 0.85)),
            ],
            &calls,
        );
        let runner = test_runner(vec![Arc::new(tier)], &[("route", &["modernbert"]), ("tag", &["modernbert"])]);
        let profiles = vec![
            route_profile(),
            tag_profile(Some(vec!["engineering".into(), "help_desk".into()])),
        ];
        let mut policies = TierPolicyMap::new();
        let mut by_task = BTreeMap::new();
        by_task.insert(String::from("route"), TierPolicy::new(0.95).with_label(LABEL_LLM, 0.70));
        by_task.insert(String::from("tag"), TierPolicy::new(0.80));
        policies.insert(tier_id("modernbert"), by_task);

        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.len(), 2);
        assert!(resolved_at(&result.tasks[&String::from("route")].resolution, "modernbert"));
        assert!(resolved_at(&result.tasks[&String::from("tag")].resolution, "modernbert"));
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
                let reading = if *task == String::from("route") {
                    scored(LABEL_JEV, 0.99)
                } else {
                    scored("help_desk", 0.92)
                };
                (task.clone(), reading)
            }).collect();
            TierOutput { model_id: format!("delayed-{}", self.name), readings }
        }
    }

    #[tokio::test]
    async fn required_route_is_published_before_best_effort_tier_finishes() {
        let runner = Arc::new(test_runner(vec![
            Arc::new(DelayedTier::new("rules", Duration::from_millis(5))),
            Arc::new(DelayedTier::new("modernbert", Duration::from_millis(250))),
        ], &[("route", &["rules"]), ("tag", &["modernbert"])]));
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
        assert!(resolved_at(&required.tasks[&String::from("route")].resolution, "rules"));
        assert_eq!(required.tasks[&String::from("tag")].resolution, Resolution::Deferred);
        let complete = split.done.await.unwrap();
        assert!(resolved_at(&complete.tasks[&String::from("tag")].resolution, "modernbert"));
        assert!(resolved_at(&complete.tasks[&String::from("route")].resolution, "rules"));
    }

    #[tokio::test]
    async fn task_without_a_policy_at_a_tier_is_not_sent_to_that_tier() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tier = FakeTier::new(
            "modernbert",
            vec![(String::from("tag"), scored("engineering", 0.99))],
            &calls,
        );
        let runner = test_runner(vec![Arc::new(tier)], &[("tag", &["modernbert"])]);
        let profiles = vec![tag_profile(None)];
        let policies = TierPolicyMap::new();

        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(result.tasks[&String::from("tag")].resolution, Resolution::Deferred);
    }

    /// The task path, not registry insertion order or TierName sorting, constrains
    /// the execution order.
    #[tokio::test]
    async fn tiers_run_in_registry_order_regardless_of_their_names() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = test_runner(vec![
            Arc::new(FakeTier::new("zulu", vec![], &calls)),
            Arc::new(FakeTier::new("alpha", vec![], &calls)),
            Arc::new(FakeTier::new("mike", vec![(String::from("route"), scored(LABEL_JEV, 0.99))], &calls)),
        ], &[("route", &["zulu", "alpha", "mike"])]);
        let policies: TierPolicyMap = ["zulu", "alpha", "mike"]
            .into_iter()
            .map(|name| (tier_id(name), [(String::from("route"), TierPolicy::new(0.9))].into()))
            .collect();

        let result = runner
            .run("prompt", &[route_profile()], &policies, Duration::from_secs(1))
            .await;
        let order: Vec<String> = calls.lock().unwrap().iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(order, ["zulu", "alpha", "mike"]);
        assert!(resolved_at(&result.tasks[&String::from("route")].resolution, "mike"));
        let called: Vec<String> = result.tier_calls.iter().map(|call| call.tier.to_string()).collect();
        assert_eq!(called, ["zulu", "alpha", "mike"]);
    }

    /// A new TierName can be registered and inserted into a task path without
    /// changing CascadeRunner or another tier adapter.
    #[tokio::test]
    async fn a_new_named_tier_can_be_inserted_between_existing_tiers() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let policies: TierPolicyMap = [
            (tier_id("fasttext"), [(String::from("route"), TierPolicy::new(0.95))].into()),
            (tier_id("prefilter"), [(String::from("route"), TierPolicy::new(0.80))].into()),
            (tier_id("modernbert"), [(String::from("route"), TierPolicy::new(0.95))].into()),
        ].into();

        let before = test_runner(vec![
            Arc::new(FakeTier::new("fasttext", vec![(String::from("route"), scored(LABEL_JEV, 0.60))], &calls)),
            Arc::new(FakeTier::new("modernbert", vec![(String::from("route"), scored(LABEL_JEV, 0.99))], &calls)),
        ], &[("route", &["fasttext", "modernbert"])]);
        let outcome = before.run("p", &[route_profile()], &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&outcome.tasks[&String::from("route")].resolution, "modernbert"));

        calls.lock().unwrap().clear();
        let after = test_runner(vec![
            Arc::new(FakeTier::new("fasttext", vec![(String::from("route"), scored(LABEL_JEV, 0.60))], &calls)),
            Arc::new(FakeTier::new("prefilter", vec![(String::from("route"), scored(LABEL_JEV, 0.85))], &calls)),
            Arc::new(FakeTier::new("modernbert", vec![(String::from("route"), scored(LABEL_JEV, 0.99))], &calls)),
        ], &[("route", &["fasttext", "prefilter", "modernbert"])]);
        let outcome = after.run("p", &[route_profile()], &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&outcome.tasks[&String::from("route")].resolution, "prefilter"));
        let order: Vec<String> = calls.lock().unwrap().iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(order, ["fasttext", "prefilter"], "modernbert must not run once prefilter resolves");
        assert_eq!(after.graph().slot(&tier_id("prefilter")).map(Slot::index), Some(1));
    }

    #[tokio::test]
    async fn registry_retains_available_tiers_absent_from_the_execution_graph() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registered = vec![
            Arc::new(FakeTier::new("rules", vec![], &calls)),
            Arc::new(FakeTier::new("modernbert", vec![], &calls)),
            Arc::new(FakeTier::new("harrier", vec![], &calls)),
        ];
        let registry = TierRegistry::of(registered);
        let graph = test_graph(&registry, &[("route", &["rules", "modernbert"])]);
        let runner = CascadeRunner::new(registry, graph).unwrap();
        assert!(runner.registry().get(&tname("harrier")).is_some());
        let harrier_id = tier_id("harrier");
        assert_eq!(runner.graph().slot(&harrier_id).map(Slot::index), None);
    }

    #[tokio::test]
    async fn a_policy_for_an_unregistered_tier_is_never_executed() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = test_runner(vec![Arc::new(FakeTier::new(
            "rules",
            vec![(String::from("route"), scored(LABEL_JEV, 0.99))],
            &calls,
        ))], &[("route", &["rules"])]);
        let policies: TierPolicyMap = [
            (tier_id("rules"), [(String::from("route"), TierPolicy::new(0.95))].into()),
            (tier_id("ghost"), [(String::from("route"), TierPolicy::new(0.10))].into()),
        ].into();
        let outcome = runner.run("p", &[route_profile()], &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&outcome.tasks[&String::from("route")].resolution, "rules"));
        assert_eq!(calls.lock().unwrap().len(), 1);
    }
}
