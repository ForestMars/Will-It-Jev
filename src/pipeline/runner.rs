// src/pipeline/runner.rs
//
// Task cascades compile into a global union of shared tier instances.
// Runtime scheduling advances each task along its own path and batches tasks
// that have reached the same node together, without cross-task path barriers.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::{JoinHandle, JoinSet};
use tracing::{debug, warn};

use super::cascade::CascadeGraph;
use super::task::{Decision, FallbackPolicy, Reading, Task, TierPolicyMap, TierReading};
use super::tier::{TierContext, TierId, TierName};

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
    pub tier: TierName,
    pub tasks: Vec<String>,
    pub took: Duration,
}

#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub tasks: BTreeMap<String, TaskOutcome>,
    pub tier_calls: Vec<TierCall>,
}

pub struct CascadeRunner {
    /// The graph owns every materialized TierInstance used for this run.
    graph: Arc<CascadeGraph>,
}

/// Required-task completion and full cascade completion are separate signals.
pub struct SplitRun {
    pub required: oneshot::Receiver<RunOutcome>,
    pub done: JoinHandle<RunOutcome>,
}

impl CascadeRunner {
    pub fn new(graph: CascadeGraph) -> Self {
        Self { graph: Arc::new(graph) }
    }

    /// The computed execution graph, including its materialized instances.
    pub fn graph(&self) -> Arc<CascadeGraph> {
        Arc::clone(&self.graph)
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

    /// Starts one graph execution. Required tasks can complete while other tasks
    /// continue through their own remaining Cascade paths.
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

        // Compile each task's own ordered path to the shared global node IDs.
        // Policies filter participation, but never alter path ordering.
        let task_paths: BTreeMap<String, Vec<TierId>> = profiles.iter()
            .map(|profile| {
                let path = self.graph.task_path(&profile.name).unwrap_or_default()
                    .iter()
                    .filter(|tier_id| policies.get(*tier_id)
                        .is_some_and(|by_task| by_task.contains_key(&profile.name)))
                    .copied()
                    .collect();
                (profile.name.clone(), path)
            })
            .collect();
        let mut next_index: BTreeMap<String, usize> = profile_order.iter()
            .map(|task| (task.clone(), 0))
            .collect();
        let mut active: BTreeSet<String> = task_paths.iter()
            .filter_map(|(task, path)| (!path.is_empty()).then_some(task.clone()))
            .collect();

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

        let shared_profiles = Arc::new(profiles.to_vec());
        let mut jobs = JoinSet::new();
        let mut jobs_by_id: std::collections::HashMap<tokio::task::Id, (TierId, Vec<String>)>
            = std::collections::HashMap::new();
        let mut pending = BTreeSet::new();

        loop {
            // Find each task's current frontier. Tasks at the same shared node
            // are batched when they become ready together; a task never waits
            // for another task's unrelated predecessor to reach that node.
            let mut ready: BTreeMap<TierId, Vec<String>> = BTreeMap::new();
            for task in &profile_order {
                if !active.contains(task) || pending.contains(task) {
                    continue;
                }
                let index = *next_index.get(task).expect("every profile has a cursor");
                if let Some(tier_id) = task_paths.get(task).and_then(|path| path.get(index)) {
                    ready.entry(*tier_id).or_default().push(task.clone());
                }
            }

            for (tier_id, task_ids) in ready {
                let Some(tier) = self.graph.instance(&tier_id) else {
                    for task in task_ids {
                        active.remove(&task);
                    }
                    continue;
                };

                let remaining = budget.saturating_sub(start.elapsed());
                if remaining.is_zero() {
                    break;
                }

                let tier_name = tier.name().clone();
                let prompt = prompt.to_string();
                let profiles_for_job = Arc::clone(&shared_profiles);
                let tasks_for_job = task_ids.clone();
                pending.extend(task_ids.iter().cloned());

                let abort = jobs.spawn(async move {
                    let call_started = Instant::now();
                    let evaluation = tokio::time::timeout(
                        remaining,
                        tier.evaluate(&TierContext {
                            prompt: &prompt,
                            tasks: &tasks_for_job,
                            task_profiles: profiles_for_job.as_slice(),
                            budget: remaining,
                        }),
                    ).await;
                    (
                        tier_id,
                        tier_name,
                        tasks_for_job,
                        call_started.elapsed(),
                        evaluation,
                    )
                });
                jobs_by_id.insert(abort.id(), (tier_id, task_ids));
            }

            if jobs.is_empty() {
                break;
            }

            match jobs.join_next_with_id().await {
                Some(Ok((job_id, (tier_id, tier_name, task_ids, took, evaluation)))) => {
                    jobs_by_id.remove(&job_id);
                    for task in &task_ids {
                        pending.remove(task);
                    }

                    match evaluation {
                        Ok(output) => {
                            tier_calls.push(TierCall {
                                tier: tier_name.clone(),
                                tasks: task_ids.clone(),
                                took,
                            });

                            let by_task = policies.get(&tier_id)
                                .expect("task paths include only tiers with policies");
                            for task_name in task_ids {
                                if !active.contains(&task_name) {
                                    continue;
                                }
                                let Some(profile) = profile_by_name.get(task_name.as_str()).copied() else {
                                    active.remove(&task_name);
                                    continue;
                                };
                                let policy = by_task.get(&task_name)
                                    .expect("task batch contains only tasks with policies");
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
                                } else {
                                    let cursor = next_index.get_mut(&task_name)
                                        .expect("active profile has a path cursor");
                                    *cursor += 1;
                                    if *cursor >= task_paths[&task_name].len() {
                                        active.remove(&task_name);
                                    }
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
                        Err(_) => {
                            // A timed-out tier only exhausts the tasks in this
                            // invocation. Other independent task paths continue.
                            for task in task_ids {
                                active.remove(&task);
                            }
                        }
                    }
                }
                Some(Err(error)) => {
                    if let Some((tier_id, task_ids)) = jobs_by_id.remove(&error.id()) {
                        for task in task_ids {
                            pending.remove(&task);
                            active.remove(&task);
                        }
                        warn!(tier_id = %tier_id, error = %error, "tier evaluation task failed");
                    } else {
                        warn!(error = %error, "tier evaluation task failed without a registered job");
                    }
                }
                None => break,
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
    use crate::pipeline::cascade::{Cascade, CascadeGraph, Slot};
    use crate::pipeline::registry::TierRegistry;
    use crate::pipeline::task::{FallbackPolicy, OutputKind, TierPolicy, LABEL_JEV, LABEL_LLM};
    use crate::pipeline::tier::{tname, FakeEvaluate, ModelFormat, Params, RequestShape, Role, Tier, TierContext, TierOutput, Transport};
    use async_trait::async_trait;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    type CallLog = Arc<Mutex<Vec<(String, Vec<String>)>>>;

    struct FakeTier {
        name: String,
        delay: Duration,
        answers: BTreeMap<String, TierReading>,
        calls: CallLog,
    }
    #[async_trait]
    impl FakeEvaluate for FakeTier {
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            self.calls.lock().unwrap().push((self.name.clone(), ctx.tasks.to_vec()));
            if !self.delay.is_zero() { tokio::time::sleep(self.delay).await; }
            let readings = ctx.tasks.iter().map(|task| (
                task.clone(),
                self.answers.get(task).cloned().unwrap_or(TierReading::Abstain("unsupported_task")),
            )).collect();
            TierOutput { model_id: format!("fake-{}", self.name), readings }
        }
    }

    fn fake_tier(name: &str, calls: &CallLog, delay: Duration,
        answers: &[(&str, TierReading)]) -> (super::super::tier::TierName, Tier, Params, Arc<dyn FakeEvaluate>) {
        (
            tname(name),
            Tier { exex: Role::Encoder, transport: Transport::Grpc, request_shape: Some(RequestShape::Score) },
            Params {
                model_format: Some(ModelFormat::Onnx), model: Some(name.into()), config: None,
                endpoint: Some("http://127.0.0.1:9".into()), timeout_ms: Some(250), enabled: None,
            },
            Arc::new(FakeTier {
                name: name.into(), delay,
                answers: answers.iter().map(|(task, reading)| ((*task).into(), reading.clone())).collect(),
                calls: Arc::clone(calls),
            }),
        )
    }

    fn task(name: &str, path: &[&str], kind: OutputKind,
        labels: &[&str], fallback: FallbackPolicy) -> Task {
        Task {
            name: name.into(),
            cascade: Cascade::from_names(&path.iter().map(|name| tname(name)).collect::<Vec<_>>()).unwrap(),
            output_kind: kind,
            labels: Some(labels.iter().map(|label| (*label).to_string()).collect()),
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback,
        }
    }
    fn route(path: &[&str]) -> Task {
        task("route", path, OutputKind::Binary, &[LABEL_JEV, LABEL_LLM],
            FallbackPolicy::Label(LABEL_LLM.into()))
    }
    fn tag(path: &[&str]) -> Task {
        task("tag", path, OutputKind::Multiclass, &["engineering", "help_desk"],
            FallbackPolicy::Defer)
    }
    fn scored(label: &str, score: f32) -> TierReading { TierReading::Scored(Reading::new(label, score)) }
    fn tier_id(graph: &CascadeGraph, name: &str) -> TierId {
        graph.id_for_name(&tname(name)).expect("tier id exists")
    }
    fn resolved_at(resolution: &Resolution, name: &str) -> bool {
        matches!(resolution, Resolution::Tier { tier, .. } if tier.as_str() == name)
    }

    #[tokio::test]
    async fn tasks_at_the_same_frontier_share_one_union_node_invocation() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registry = TierRegistry::of(vec![
            fake_tier("modernbert", &calls, Duration::ZERO, &[
                ("route", scored(LABEL_JEV, 0.97)), ("tag", scored("engineering", 0.88)),
            ]),
        ]);
        let profiles = vec![route(&["modernbert"]), tag(&["modernbert"])];
        let graph = CascadeGraph::from_tasks(&profiles, &registry).unwrap();
        let modernbert = tier_id(&graph, "modernbert");
        let runner = CascadeRunner::new(graph);
        let policies: TierPolicyMap = [(
            modernbert,
            [
                ("route".into(), TierPolicy::new(0.95)),
                ("tag".into(), TierPolicy::new(0.80)),
            ].into(),
        )].into();

        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&result.tasks["route"].resolution, "modernbert"));
        assert!(resolved_at(&result.tasks["tag"].resolution, "modernbert"));
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            [("modernbert".into(), vec!["route".into(), "tag".into()])],
        );
    }

    #[tokio::test]
    async fn required_task_can_finish_before_another_tasks_path_continues() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registry = TierRegistry::of(vec![
            fake_tier("rules", &calls, Duration::from_millis(5), &[("route", scored(LABEL_JEV, 0.99))]),
            fake_tier("modernbert", &calls, Duration::from_millis(250), &[("tag", scored("help_desk", 0.92))]),
        ]);
        let profiles = vec![route(&["rules"]), tag(&["modernbert"])];
        let graph = CascadeGraph::from_tasks(&profiles, &registry).unwrap();
        let rules = tier_id(&graph, "rules");
        let modernbert = tier_id(&graph, "modernbert");
        let runner = Arc::new(CascadeRunner::new(graph));
        let policies: TierPolicyMap = [
            (rules, [("route".into(), TierPolicy::new(0.9))].into()),
            (modernbert, [("tag".into(), TierPolicy::new(0.8))].into()),
        ].into();
        let split = runner.run_split("prompt".into(), profiles, ["route".into()].into(),
            Arc::new(policies), Duration::from_secs(2));
        let required = tokio::time::timeout(Duration::from_millis(100), split.required)
            .await.expect("route should complete before tag's tier").unwrap();
        assert!(resolved_at(&required.tasks["route"].resolution, "rules"));
        assert_eq!(required.tasks["tag"].resolution, Resolution::Deferred);
        let complete = split.done.await.unwrap();
        assert!(resolved_at(&complete.tasks["tag"].resolution, "modernbert"));
    }

    #[tokio::test]
    async fn a_slow_sidecar_predecessor_does_not_block_a_required_task_at_a_shared_node() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registry = TierRegistry::of(vec![
            fake_tier("rules", &calls, Duration::from_millis(5), &[
                ("route", TierReading::Abstain("pass")),
            ]),
            fake_tier("prefilter", &calls, Duration::from_millis(250), &[
                ("tag", TierReading::Abstain("continue")),
            ]),
            fake_tier("modernbert", &calls, Duration::from_millis(5), &[
                ("route", scored(LABEL_JEV, 0.99)),
                ("tag", scored("engineering", 0.91)),
            ]),
        ]);
        // Put the optional task first deliberately. Its predecessor is slower,
        // but it shares the eventual encoder TierInstance with the required route.
        let profiles = vec![
            tag(&["prefilter", "modernbert"]),
            route(&["rules", "modernbert"]),
        ];
        let graph = CascadeGraph::from_tasks(&profiles, &registry).unwrap();
        let rules = tier_id(&graph, "rules");
        let prefilter = tier_id(&graph, "prefilter");
        let modernbert = tier_id(&graph, "modernbert");
        let runner = Arc::new(CascadeRunner::new(graph));
        let policies: TierPolicyMap = [
            (rules, [("route".into(), TierPolicy::new(0.9))].into()),
            (prefilter, [("tag".into(), TierPolicy::new(0.8))].into()),
            (modernbert, [
                ("route".into(), TierPolicy::new(0.9)),
                ("tag".into(), TierPolicy::new(0.8)),
            ].into()),
        ].into();

        let split = runner.run_split(
            "prompt".into(),
            profiles,
            ["route".into()].into(),
            Arc::new(policies),
            Duration::from_secs(2),
        );
        let required = tokio::time::timeout(Duration::from_millis(100), split.required)
            .await
            .expect("the route must reach the shared node without waiting for tag")
            .unwrap();
        assert!(resolved_at(&required.tasks["route"].resolution, "modernbert"));
        assert_eq!(required.tasks["tag"].resolution, Resolution::Deferred);

        let complete = split.done.await.unwrap();
        assert!(resolved_at(&complete.tasks["tag"].resolution, "modernbert"));

        let mut encoder_batches: Vec<Vec<String>> = calls.lock().unwrap().iter()
            .filter(|(tier, _)| tier == "modernbert")
            .map(|(_, tasks)| tasks.clone())
            .collect();
        encoder_batches.sort();
        assert_eq!(encoder_batches, [vec!["route".to_string()], vec!["tag".to_string()]]);
    }

    #[tokio::test]
    async fn a_policy_cannot_make_a_tier_outside_the_task_cascade_eligible() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registry = TierRegistry::of(vec![
            fake_tier("rules", &calls, Duration::ZERO, &[("route", scored(LABEL_JEV, 0.99))]),
            fake_tier("harrier", &calls, Duration::ZERO, &[("route", scored(LABEL_LLM, 0.99))]),
        ]);
        let profiles = vec![route(&["rules"])];
        let graph = CascadeGraph::from_tasks(&profiles, &registry).unwrap();
        let rules = tier_id(&graph, "rules");
        let unrelated = TierId::from_digest(&[0xab; 32]);
        let runner = CascadeRunner::new(graph);
        let policies: TierPolicyMap = [
            (rules, [("route".into(), TierPolicy::new(0.9))].into()),
            (unrelated, [("route".into(), TierPolicy::new(0.1))].into()),
        ].into();
        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&result.tasks["route"].resolution, "rules"));
        assert_eq!(calls.lock().unwrap().as_slice(), [("rules".into(), vec!["route".into()])]);
    }

    #[tokio::test]
    async fn registry_keeps_unused_definitions_without_graph_instances() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registry = TierRegistry::of(vec![
            fake_tier("rules", &calls, Duration::ZERO, &[]),
            fake_tier("harrier", &calls, Duration::ZERO, &[]),
        ]);
        let profiles = vec![route(&["rules"])];
        let graph = CascadeGraph::from_tasks(&profiles, &registry).unwrap();
        let runner = CascadeRunner::new(graph);
        assert!(registry.get(&tname("harrier")).is_some());
        assert!(runner.graph().id_for_name(&tname("harrier")).is_none());
        assert!(runner.graph().instance_for_name(&tname("harrier")).is_none());
    }

    #[tokio::test]
    async fn task_with_no_policy_uses_fallback_without_a_tier_call() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registry = TierRegistry::of(vec![fake_tier("rules", &calls, Duration::ZERO, &[])]);
        let profiles = vec![tag(&["rules"])];
        let graph = CascadeGraph::from_tasks(&profiles, &registry).unwrap();
        let runner = CascadeRunner::new(graph);
        let result = runner.run("prompt", &profiles, &TierPolicyMap::new(), Duration::from_secs(1)).await;
        assert_eq!(result.tasks["tag"].resolution, Resolution::Deferred);
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(Slot::from_index(3).map(Slot::index), Some(3));
    }
}
