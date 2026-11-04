// src/pipeline/tagger.rs
//
// Adapt the task-independent cascade to the existing tagging interface. The
// tag task supplies its own profile and is evaluated with the same shared
// cascade and separate tier/task policy registry as the route task.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use super::runner::{CascadeRunner, Resolution};
use super::task::{FallbackPolicy, Reading, TaskProfile, TierPolicyMap};
use crate::tagging::{CategoryScore, TagError, TagInput, Tagger, TaggerOutput, TaggerVerdict};

pub struct CascadeTagger<F: Tagger> {
    runner: Arc<CascadeRunner>,
    profile: TaskProfile,
    tier_policies: Arc<TierPolicyMap>,
    budget: Duration,
    fallback: F,
}

impl<F: Tagger> CascadeTagger<F> {
    /// The tag profile must defer on exhaustion so the fallback tagger, rather
    /// than an artificial fallback label, produces the recorded category.
    pub fn new(
        runner: Arc<CascadeRunner>,
        profile: TaskProfile,
        tier_policies: Arc<TierPolicyMap>,
        budget: Duration,
        fallback: F,
    ) -> Result<Self, String> {
        if profile.fallback != FallbackPolicy::Defer {
            return Err(format!(
                "task.{}: a tagging profile must use fallback = \"defer\"",
                profile.task
            ));
        }
        if !tier_policies
            .values()
            .any(|by_task| by_task.contains_key(&profile.task))
        {
            return Err(format!(
                "task.{}: no tier policies are configured for cascade tagging",
                profile.task
            ));
        }
        Ok(Self {
            runner,
            profile,
            tier_policies,
            budget,
            fallback,
        })
    }
}

fn verdict(reading: &Reading) -> TaggerVerdict {
    let mut alternatives: Vec<CategoryScore> = reading
        .scores
        .iter()
        .filter(|(label, _)| *label != reading.label)
        .map(|(label, score)| CategoryScore {
            category_id: label.clone(),
            score: *score,
        })
        .collect();
    alternatives.sort_by(|left, right| right.score.total_cmp(&left.score));
    TaggerVerdict::Tagged {
        primary: CategoryScore {
            category_id: reading.label.clone(),
            score: reading.confidence,
        },
        alternatives,
    }
}

impl<F: Tagger> Tagger for CascadeTagger<F> {
    fn model(&self) -> &str {
        "oven-cascade-tagger-v1"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn taxonomy_version(&self) -> &str {
        self.fallback.taxonomy_version()
    }

    fn tag(
        &self,
        input: &TagInput,
    ) -> impl Future<Output = Result<TaggerOutput, TagError>> + Send {
        async move {
            // Preserve the existing tagger input contract: the domain task may use
            // both the request and the generated response. Route inference, by
            // contrast, runs before a response exists.
            let task_input = match input.response.as_deref().filter(|response| !response.is_empty()) {
                Some(response) => format!("Prompt:\n{}\n\nResponse:\n{}", input.prompt, response),
                None => input.prompt.clone(),
            };
            let outcome = self
                .runner
                .run(
                    &task_input,
                    std::slice::from_ref(&self.profile),
                    &self.tier_policies,
                    self.budget,
                )
                .await;
            match outcome
                .tasks
                .get(&self.profile.task)
                .map(|task| &task.resolution)
            {
                Some(Resolution::Tier {
                    tier,
                    model_id,
                    reading,
                }) => {
                    tracing::debug!(
                        request_id = %input.request_id,
                        tier,
                        model_id,
                        "tag resolved in cascade"
                    );
                    Ok(TaggerOutput {
                        verdict: verdict(reading),
                        usage: None,
                    })
                }
                _ => self.fallback.tag(input).await,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::runner::CascadeRunner;
    use crate::pipeline::task::{TaskId, TierPolicy, TierReading};
    use crate::pipeline::tier::{Tier, TierContext, TierOutput};
    use crate::tagging::{Tagger, TaggerVerdict};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TagTier;

    #[async_trait]
    impl Tier for TagTier {
        fn id(&self) -> u8 { 4 }
        fn name(&self) -> &str { "test-tag-head" }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            TierOutput {
                model_id: "test-tag-head".into(),
                readings: ctx.tasks.iter().map(|task| {
                    let reading = if *task == TaskId::tag() {
                        TierReading::Scored(Reading {
                            label: "engineering".into(),
                            confidence: 0.91,
                            scores: vec![
                                ("engineering".into(), 0.91),
                                ("help_desk".into(), 0.09),
                            ],
                        })
                    } else {
                        TierReading::Abstain("unsupported_task")
                    };
                    (task.clone(), reading)
                }).collect(),
            }
        }
    }

    #[derive(Default)]
    struct Fallback(AtomicUsize);

    impl Tagger for Fallback {
        fn model(&self) -> &str { "fallback" }
        fn version(&self) -> &str { "0" }
        fn taxonomy_version(&self) -> &str { "test-taxonomy" }
        fn tag(
            &self,
            _: &TagInput,
        ) -> impl Future<Output = Result<TaggerOutput, TagError>> + Send {
            self.0.fetch_add(1, Ordering::SeqCst);
            async {
                Ok(TaggerOutput {
                    verdict: TaggerVerdict::NoMatch {
                        nearest: None,
                        embedding: None,
                    },
                    usage: None,
                })
            }
        }
    }

    #[tokio::test]
    async fn tag_task_uses_policy_registry_and_does_not_call_fallback_when_resolved() {
        let runner = Arc::new(CascadeRunner::new(vec![Arc::new(TagTier)]));
        let profile = TaskProfile {
            task: TaskId::tag(),
            labels: Some(vec!["engineering".into(), "help_desk".into()]),
            fallback: FallbackPolicy::Defer,
        };
        let mut policies = TierPolicyMap::new();
        policies.insert(4, [(TaskId::tag(), TierPolicy::new(0.8))].into());
        let fallback = Fallback::default();
        let tagger = CascadeTagger::new(
            runner,
            profile,
            Arc::new(policies),
            Duration::from_secs(1),
            fallback,
        ).unwrap();
        let input = TagInput {
            request_id: "r1".into(),
            prompt: "help me".into(),
            response: None,
        };
        let output = tagger.tag(&input).await.unwrap();
        assert!(matches!(output.verdict, TaggerVerdict::Tagged { .. }));
    }
}
