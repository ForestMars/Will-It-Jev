// src/pipeline/runner.rs
//
// The generic engine: walk the tiers once, in order, over the union of tasks still
// open. Each tier is called at most once per run, with every open task whose profile
// allows that tier, so tasks share the execution. A task leaves the active set when
// its own profile resolves it; the run ends when the set is empty or the tiers run out.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::debug;

use super::task::{Decision, FallbackPolicy, Reading, TaskId, TaskProfile, TierReading};
use super::tier::{Tier, TierContext};

#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    /// A tier cleared the task's bar.
    Tier { tier: u8, model_id: String, reading: Reading },
    /// No tier resolved it and the profile names a fallback label.
    Fallback { label: String },
    /// No tier resolved it and the profile leaves that to the caller.
    Deferred,
}

#[derive(Debug, Clone)]
pub struct TierTrace {
    pub tier: u8,
    pub model_id: String,
    pub decision: Decision,
}

#[derive(Debug, Clone)]
pub struct TaskOutcome {
    pub resolution: Resolution,
    /// One entry per tier that was consulted for this task, in order.
    pub trace: Vec<TierTrace>,
}

/// One backend execution. `tasks` is how many use cases it served.
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

impl CascadeRunner {
    /// Tiers are ordered by `Tier::id`.
    pub fn new(mut tiers: Vec<Arc<dyn Tier>>) -> Self {
        tiers.sort_by_key(|tier| tier.id());
        Self { tiers }
    }

    /// `budget` is the global deadline for the whole run. Profiles must have distinct
    /// task ids.
    pub async fn run(&self, prompt: &str, profiles: &[TaskProfile], budget: Duration) -> RunOutcome {
        let start = Instant::now();
        let mut active: BTreeSet<TaskId> = profiles.iter().map(|p| p.task.clone()).collect();
        debug_assert_eq!(active.len(), profiles.len(), "duplicate task ids in profiles");

        let mut resolved: BTreeMap<TaskId, Resolution> = BTreeMap::new();
        let mut traces: BTreeMap<TaskId, Vec<TierTrace>> = BTreeMap::new();
        let mut tier_calls = Vec::new();

        for tier in &self.tiers {
            if active.is_empty() {
                break;
            }
            let wanted: Vec<&TaskProfile> = profiles
                .iter()
                .filter(|p| active.contains(&p.task) && p.policy(tier.id()).is_some())
                .collect();
            if wanted.is_empty() {
                continue;
            }
            let ids: Vec<TaskId> = wanted.iter().map(|p| p.task.clone()).collect();

            let t = Instant::now();
            let output = tier
                .evaluate(&TierContext {
                    prompt,
                    tasks: &ids,
                    budget: budget.saturating_sub(start.elapsed()),
                })
                .await;
            tier_calls.push(TierCall { tier: tier.id(), tasks: ids, took: t.elapsed() });

            for profile in wanted {
                let reading = output
                    .readings
                    .get(&profile.task)
                    .cloned()
                    .unwrap_or(TierReading::Abstain("no_reading"));
                let decision = profile.decide(tier.id(), &reading);
                debug!(task = %profile.task, tier = tier.id(), ?decision, "task decision");
                if let Decision::Resolve(reading) = &decision {
                    resolved.insert(
                        profile.task.clone(),
                        Resolution::Tier {
                            tier: tier.id(),
                            model_id: output.model_id.clone(),
                            reading: reading.clone(),
                        },
                    );
                    active.remove(&profile.task);
                }
                traces.entry(profile.task.clone()).or_default().push(TierTrace {
                    tier: tier.id(),
                    model_id: output.model_id.clone(),
                    decision,
                });
            }
        }

        let tasks = profiles
            .iter()
            .map(|profile| {
                let resolution = resolved.remove(&profile.task).unwrap_or_else(|| match &profile.fallback {
                    FallbackPolicy::Label(label) => Resolution::Fallback { label: label.clone() },
                    FallbackPolicy::Defer => Resolution::Deferred,
                });
                let trace = traces.remove(&profile.task).unwrap_or_default();
                (profile.task.clone(), TaskOutcome { resolution, trace })
            })
            .collect();

        RunOutcome { tasks, tier_calls }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::task::{TierPolicy, LABEL_JEV, LABEL_LLM};
    use crate::pipeline::tier::{TierContext, TierOutput};
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// A tier that answers from a fixed table and records every call it receives.
    struct Fake {
        id: u8,
        answers: Vec<(TaskId, TierReading)>,
        calls: Arc<Mutex<Vec<(u8, Vec<TaskId>)>>>,
    }

    #[async_trait]
    impl Tier for Fake {
        fn id(&self) -> u8 {
            self.id
        }
        fn name(&self) -> &str {
            "fake"
        }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            self.calls.lock().unwrap().push((self.id, ctx.tasks.to_vec()));
            let readings = ctx
                .tasks
                .iter()
                .map(|t| {
                    let r = self
                        .answers
                        .iter()
                        .find(|(task, _)| task == t)
                        .map(|(_, r)| r.clone())
                        .unwrap_or(TierReading::Abstain("unsupported_task"));
                    (t.clone(), r)
                })
                .collect();
            TierOutput { model_id: format!("fake{}", self.id), readings }
        }
    }

    fn tag_profile(tiers: &[u8], bar: f32, fallback: FallbackPolicy) -> TaskProfile {
        TaskProfile {
            task: TaskId::tag(),
            labels: Some(vec!["engineering".into(), "help_desk".into()]),
            tiers: tiers.iter().map(|t| (*t, TierPolicy::new(bar))).collect(),
            fallback,
        }
    }

    fn scored(label: &str, c: f32) -> TierReading {
        TierReading::Scored(Reading::new(label, c))
    }

    type Calls = Arc<Mutex<Vec<(u8, Vec<TaskId>)>>>;

    fn runner(tiers: Vec<(u8, Vec<(TaskId, TierReading)>)>) -> (CascadeRunner, Calls) {
        let calls: Calls = Arc::default();
        let tiers = tiers
            .into_iter()
            .map(|(id, answers)| Arc::new(Fake { id, answers, calls: calls.clone() }) as Arc<dyn Tier>)
            .collect();
        (CascadeRunner::new(tiers), calls)
    }

    /// Both tasks resolve out of one tier execution.
    #[tokio::test]
    async fn two_tasks_share_one_tier_call() {
        let (runner, calls) = runner(vec![(
            3,
            vec![
                (TaskId::route(), scored(LABEL_JEV, 0.97)),
                (TaskId::tag(), scored("engineering", 0.9)),
            ],
        )]);
        let profiles = [TaskProfile::route_default(), tag_profile(&[3], 0.8, FallbackPolicy::Defer)];
        let out = runner.run("p", &profiles, Duration::from_secs(1)).await;

        assert_eq!(calls.lock().unwrap().len(), 1, "one execution for two tasks");
        assert_eq!(out.tier_calls[0].tasks.len(), 2);
        assert!(matches!(out.tasks[&TaskId::route()].resolution, Resolution::Tier { tier: 3, .. }));
        assert!(matches!(out.tasks[&TaskId::tag()].resolution, Resolution::Tier { tier: 3, .. }));
    }

    /// One task exits early; the other continues, and the later tier sees only it.
    #[tokio::test]
    async fn resolved_task_leaves_and_the_other_continues() {
        let (runner, calls) = runner(vec![
            (3, vec![(TaskId::route(), scored(LABEL_JEV, 0.99)), (TaskId::tag(), scored("help_desk", 0.4))]),
            (4, vec![(TaskId::tag(), scored("help_desk", 0.92))]),
        ]);
        let profiles = [TaskProfile::route_default(), tag_profile(&[3, 4], 0.8, FallbackPolicy::Defer)];
        let out = runner.run("p", &profiles, Duration::from_secs(1)).await;

        let calls = calls.lock().unwrap();
        assert_eq!(calls[0], (3, vec![TaskId::route(), TaskId::tag()]));
        assert_eq!(calls[1], (4, vec![TaskId::tag()]));
        assert!(matches!(out.tasks[&TaskId::route()].resolution, Resolution::Tier { tier: 3, .. }));
        assert!(matches!(out.tasks[&TaskId::tag()].resolution, Resolution::Tier { tier: 4, .. }));
        assert_eq!(out.tasks[&TaskId::tag()].trace.len(), 2);
    }

    /// Profiles are tuned independently: the same reading resolves one task and not another.
    #[tokio::test]
    async fn each_task_applies_its_own_bar() {
        let (runner, _) = runner(vec![(3, vec![(TaskId::route(), scored(LABEL_LLM, 0.75)), (TaskId::tag(), scored("engineering", 0.75))])]);
        let profiles = [TaskProfile::route_default(), tag_profile(&[3], 0.9, FallbackPolicy::Defer)];
        let out = runner.run("p", &profiles, Duration::from_secs(1)).await;
        // route: llm bar is 0.70 so 0.75 resolves; tag: bar 0.9 so it does not.
        assert!(matches!(out.tasks[&TaskId::route()].resolution, Resolution::Tier { .. }));
        assert_eq!(out.tasks[&TaskId::tag()].resolution, Resolution::Deferred);
    }

    #[tokio::test]
    async fn fallbacks_apply_when_no_tier_resolves() {
        let (runner, _) = runner(vec![(3, vec![])]);
        let profiles = [TaskProfile::route_default(), tag_profile(&[3], 0.8, FallbackPolicy::Defer)];
        let out = runner.run("p", &profiles, Duration::from_secs(1)).await;
        assert_eq!(out.tasks[&TaskId::route()].resolution, Resolution::Fallback { label: LABEL_LLM.into() });
        assert_eq!(out.tasks[&TaskId::tag()].resolution, Resolution::Deferred);
    }

    #[tokio::test]
    async fn tiers_outside_a_profile_are_skipped_and_empty_runs_stop() {
        let (runner, calls) = runner(vec![(1, vec![]), (3, vec![(TaskId::tag(), scored("engineering", 0.95))]), (4, vec![])]);
        let profiles = [tag_profile(&[3, 4], 0.8, FallbackPolicy::Defer)];
        runner.run("p", &profiles, Duration::from_secs(1)).await;
        // tier 1 has no tag policy; tier 4 is never reached because tier 3 resolved it.
        assert_eq!(calls.lock().unwrap().iter().map(|c| c.0).collect::<Vec<_>>(), vec![3]);
    }
}
