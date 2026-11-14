// src/pipeline/runner.rs
//
// Task cascades compile into a shared execution graph. At each graph node,
// only tasks whose own path includes that tier and whose policy exists are batched.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::debug;

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

        // A task participates only at tiers where a corresponding policy exists.
        // Tasks with no policy anywhere immediately take their configured fallback.
        let last_policy_index: BTreeMap<String, usize> = profiles.iter()
            .filter_map(|profile| {
                self.graph.order().iter().enumerate()
                    .filter(|(_, tier_id)| {
                        self.graph.task_uses(&profile.name, tier_id)
                            && policies.get(tier_id)
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

        for (tier_index, tier_id) in self.graph.order().iter().copied().enumerate() {
            if active.is_empty() || start.elapsed() >= budget {
                break;
            }

            let task_ids: Vec<String> = profile_order.iter()
                .filter(|task| {
                    active.contains(*task)
                        && self.graph.task_uses(task, &tier_id)
                        && policies.get(&tier_id)
                            .is_some_and(|by_task| by_task.contains_key(*task))
                })
                .cloned()
                .collect();
            if task_ids.is_empty() {
                continue;
            }

            let Some(tier) = self.graph.instance(&tier_id) else {
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
    async fn task_paths_share_graph_nodes_and_batch_only_eligible_tasks() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registry = TierRegistry::of(vec![
            fake_tier("rules", &calls, Duration::ZERO, &[("route", TierReading::Abstain("pass"))]),
            fake_tier("modernbert", &calls, Duration::ZERO, &[
                ("route", scored(LABEL_JEV, 0.97)), ("tag", scored("engineering", 0.88)),
            ]),
        ]);
        let profiles = vec![route(&["rules", "modernbert"]), tag(&["modernbert"])];
        let graph = CascadeGraph::from_tasks(&profiles, &registry).unwrap();
        let rules = tier_id(&graph, "rules");
        let modernbert = tier_id(&graph, "modernbert");
        let runner = CascadeRunner::new(graph);
        let policies: TierPolicyMap = [
            (rules, [("route".into(), TierPolicy::new(0.78))].into()),
            (modernbert, [
                ("route".into(), TierPolicy::new(0.95)),
                ("tag".into(), TierPolicy::new(0.80)),
            ].into()),
        ].into();

        let result = runner.run("prompt", &profiles, &policies, Duration::from_secs(1)).await;
        assert!(resolved_at(&result.tasks["route"].resolution, "modernbert"));
        assert!(resolved_at(&result.tasks["tag"].resolution, "modernbert"));
        assert_eq!(calls.lock().unwrap().as_slice(), [
            ("rules".into(), vec!["route".into()]),
            ("modernbert".into(), vec!["route".into(), "tag".into()]),
        ]);
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
    async fn registry_keeps_instances_unused_by_every_task_path() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let registry = TierRegistry::of(vec![
            fake_tier("rules", &calls, Duration::ZERO, &[]),
            fake_tier("harrier", &calls, Duration::ZERO, &[]),
        ]);
        let profiles = vec![route(&registry, &["rules"])];
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
