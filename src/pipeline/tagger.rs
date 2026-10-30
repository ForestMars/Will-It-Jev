// src/pipeline/tagger.rs
//
// Category tagging as the second use case on the shared cascade. `CascadeTagger`
// implements the existing `tagging::Tagger` contract, so the queue, sampling, JSONL
// and ClickHouse sinks are untouched. It runs the `tag` profile through the tiers;
// when no tier resolves it, it hands the request to the existing LLM tagger, which
// plays the role tier 6 plays for routing.
//
// The cascade tiers see the prompt only. A tagger model fed prompt + response (as
// OvenV1Tagger is) and one fed prompt alone are different classifiers, which is
// why this is behind `tag_via_cascade` and off by default.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use super::runner::{CascadeRunner, Resolution};
use super::task::{FallbackPolicy, Reading, TaskProfile};
use crate::tagging::{
    CategoryScore, TagError, TagInput, Tagger, TaggerOutput, TaggerVerdict,
};

pub struct CascadeTagger<F: Tagger> {
    runner: Arc<CascadeRunner>,
    profile: TaskProfile,
    budget: Duration,
    fallback: F,
}

impl<F: Tagger> CascadeTagger<F> {
    /// The tag profile must `Defer`: a fixed fallback label would be recorded as a
    /// real tag. A run that resolves nothing goes to `fallback` instead.
    pub fn new(
        runner: Arc<CascadeRunner>,
        profile: TaskProfile,
        budget: Duration,
        fallback: F,
    ) -> Result<Self, String> {
        if profile.fallback != FallbackPolicy::Defer {
            return Err(format!("task.{}: a tagging profile must use fallback = \"defer\"", profile.task));
        }
        Ok(Self { runner, profile, budget, fallback })
    }
}

fn verdict(reading: &Reading) -> TaggerVerdict {
    let mut alternatives: Vec<CategoryScore> = reading
        .scores
        .iter()
        .filter(|(label, _)| *label != reading.label)
        .map(|(label, score)| CategoryScore { category_id: label.clone(), score: *score })
        .collect();
    alternatives.sort_by(|a, b| b.score.total_cmp(&a.score));
    TaggerVerdict::Tagged {
        primary: CategoryScore { category_id: reading.label.clone(), score: reading.confidence },
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

    // Same taxonomy as the fallback tagger: both emit ids from it.
    fn taxonomy_version(&self) -> &str {
        self.fallback.taxonomy_version()
    }

    fn tag(
        &self,
        input: &TagInput,
    ) -> impl Future<Output = Result<TaggerOutput, TagError>> + Send {
        async move {
            let outcome = self
                .runner
                .run(&input.prompt, std::slice::from_ref(&self.profile), self.budget)
                .await;
            match outcome.tasks.get(&self.profile.task).map(|t| &t.resolution) {
                Some(Resolution::Tier { tier, model_id, reading }) => {
                    tracing::debug!(request_id = %input.request_id, tier, model_id, "tag resolved in cascade");
                    Ok(TaggerOutput { verdict: verdict(reading), usage: None })
                }
                _ => self.fallback.tag(input).await,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::task::{TaskId, TierPolicy, TierReading};
    use crate::pipeline::tier::{Tier, TierContext, TierOutput};
    use crate::tagging::TaggerVerdict;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scripted(TierReading);

    #[async_trait]
    impl Tier for Scripted {
        fn id(&self) -> u8 {
            4
        }
        fn name(&self) -> &str {
            "scripted"
        }
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            TierOutput {
                model_id: "scripted".into(),
                readings: ctx.tasks.iter().map(|t| (t.clone(), self.0.clone())).collect(),
            }
        }
    }

    #[derive(Default)]
    struct Fallback {
        calls: AtomicUsize,
    }

    impl Tagger for Fallback {
        fn model(&self) -> &str {
            "fallback"
        }
        fn version(&self) -> &str {
            "0"
        }
        fn taxonomy_version(&self) -> &str {
            "v1.0"
        }
        fn tag(&self, _: &TagInput) -> impl Future<Output = Result<TaggerOutput, TagError>> + Send {
            self.calls.fetch_add(1, Ordering::SeqCst);
            async {
                Ok(TaggerOutput {
                    verdict: TaggerVerdict::Tagged {
                        primary: CategoryScore { category_id: "other_general_request".into(), score: 1.0 },
                        alternatives: vec![],
                    },
                    usage: None,
                })
            }
        }
    }

    fn profile() -> TaskProfile {
        TaskProfile {
            task: TaskId::tag(),
            labels: Some(vec!["engineering".into(), "help_desk".into(), "web_search".into()]),
            tiers: [(4u8, TierPolicy::new(0.8))].into(),
            fallback: FallbackPolicy::Defer,
        }
    }

    fn input() -> TagInput {
        TagInput { request_id: "r".into(), prompt: "fix my laptop".into(), response: None }
    }

    fn tagger(reading: TierReading, fb: Fallback) -> CascadeTagger<Fallback> {
        let runner = Arc::new(CascadeRunner::new(vec![Arc::new(Scripted(reading))]));
        CascadeTagger::new(runner, profile(), Duration::from_secs(1), fb).unwrap()
    }

    #[tokio::test]
    async fn resolved_tag_skips_the_fallback_and_orders_alternatives() {
        let reading = Reading {
            label: "help_desk".into(),
            confidence: 0.9,
            scores: vec![("engineering".into(), 0.02), ("help_desk".into(), 0.9), ("web_search".into(), 0.05)],
        };
        let t = tagger(TierReading::Scored(reading), Fallback::default());
        let out = t.tag(&input()).await.unwrap();
        let TaggerVerdict::Tagged { primary, alternatives } = out.verdict else { panic!("not tagged") };
        assert_eq!(primary.category_id, "help_desk");
        assert_eq!(alternatives.iter().map(|a| a.category_id.as_str()).collect::<Vec<_>>(), ["web_search", "engineering"]);
        assert_eq!(t.fallback.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn unresolved_tag_goes_to_the_fallback_tagger() {
        let t = tagger(TierReading::Abstain("unsupported_task"), Fallback::default());
        let out = t.tag(&input()).await.unwrap();
        let TaggerVerdict::Tagged { primary, .. } = out.verdict else { panic!("not tagged") };
        assert_eq!(primary.category_id, "other_general_request");
        assert_eq!(t.fallback.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_tag_profile_with_a_fallback_label_is_rejected() {
        let mut p = profile();
        p.fallback = FallbackPolicy::Label("other_general_request".into());
        let runner = Arc::new(CascadeRunner::new(vec![]));
        assert!(CascadeTagger::new(runner, p, Duration::from_secs(1), Fallback::default()).is_err());
    }
}
