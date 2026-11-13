// src/pipeline/tagger.rs
//
// Adapt the task-independent cascade to the existing tagging interface. The
// tag task supplies its own profile and is evaluated with the same shared
// cascade and separate tier/task policy registry as the route task.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::legacy_tiers::LegacyTierNumbers;
use super::runner::{Resolution, RunOutcome};
use super::task::Reading;
use crate::tagging::{CategoryScore, TaskDecisionMetadata, TagError, TagInput, Tagger, TaggerOutput, TaggerVerdict};

struct JointEntry {
    receiver: oneshot::Receiver<Result<RunOutcome, ()>>,
    expires_at: Instant,
}

/// Request-ID keyed handoff from the route handler to asynchronous tag persistence.
/// The store carries an outcome from the existing cascade; it never starts inference.
#[derive(Clone, Default)]
pub struct JointTagStore {
    entries: Arc<Mutex<HashMap<String, JointEntry>>>,
}

impl JointTagStore {
    pub fn register(&self, request_id: String, done: JoinHandle<RunOutcome>, ttl: Duration) {
        let (sender, receiver) = oneshot::channel();
        let entry = JointEntry { receiver, expires_at: Instant::now() + ttl };
        {
            let mut entries = self.entries.lock().expect("joint tag store mutex poisoned");
            entries.retain(|_, value| value.expires_at > Instant::now());
            entries.insert(request_id, entry);
        }
        tokio::spawn(async move {
            let result = done.await.map_err(|_| ());
            let _ = sender.send(result);
        });
    }

    /// Ok(None) means this request was not included in a joint cascade and may
    /// use the legacy tagger. Timeout means it was included but has not finished;
    /// in that case a second inference must not be launched.
    pub async fn take(&self, request_id: &str, timeout: Duration) -> Result<Option<RunOutcome>, TagError> {
        let entry = {
            let mut entries = self.entries.lock()
                .map_err(|_| TagError::Failed("joint tag store mutex poisoned".into()))?;
            entries.retain(|_, value| value.expires_at > Instant::now());
            entries.remove(request_id)
        };
        let Some(entry) = entry else { return Ok(None) };
        match tokio::time::timeout(timeout, entry.receiver).await {
            Err(_) => Err(TagError::Timeout),
            Ok(Err(_)) | Ok(Ok(Err(()))) => {
                Err(TagError::Failed("joint cascade task terminated before tag persistence".into()))
            }
            Ok(Ok(Ok(outcome))) => Ok(Some(outcome)),
        }
    }
}

pub struct CascadeTagger<F: Tagger> {
    store: JointTagStore,
    budget: Duration,
    taxonomy_version: String,
    output_kind: String,
    fallback: F,
    legacy: Arc<LegacyTierNumbers>,
}

impl<F: Tagger> CascadeTagger<F> {
    pub fn new(
        store: JointTagStore,
        budget: Duration,
        taxonomy_version: impl Into<String>,
        output_kind: impl Into<String>,
        fallback: F,
    ) -> Self {
        Self {
            store,
            budget,
            taxonomy_version: taxonomy_version.into(),
            output_kind: output_kind.into(),
            fallback,
            legacy: Arc::new(LegacyTierNumbers::default()),
        }
    }

    /// Lets the tagger record a resolving tier's legacy number next to its name.
    pub fn with_legacy_numbers(mut self, legacy: Arc<LegacyTierNumbers>) -> Self {
        self.legacy = legacy;
        self
    }
}

fn verdict(reading: &Reading) -> TaggerVerdict {
    let mut alternatives: Vec<CategoryScore> = reading.scores.iter()
        .filter(|(label, _)| label.as_str() != reading.label.as_str())
        .map(|(label, score)| CategoryScore { category_id: label.clone(), score: *score })
        .collect();
    alternatives.sort_by(|left, right| right.score.total_cmp(&left.score));
    TaggerVerdict::Tagged {
        primary: CategoryScore { category_id: reading.label.clone(), score: reading.confidence },
        alternatives,
    }
}

impl<F: Tagger> Tagger for CascadeTagger<F> {
    fn model(&self) -> &str { "oven-joint-cascade-tagger-v1" }
    fn version(&self) -> &str { "1.0.0" }
    fn taxonomy_version(&self) -> &str { &self.taxonomy_version }

    async fn tag(&self, input: &TagInput) -> Result<TaggerOutput, TagError> {
        let Some(outcome) = self.store.take(&input.request_id, self.budget).await? else {
            return self.fallback.tag(input).await;
        };
        match outcome.tasks.get(&String::from("tag")).map(|task| &task.resolution) {
            Some(Resolution::Tier { tier_id, tier, model_id, reading }) => {
                tracing::debug!(request_id = %input.request_id, tier = %tier, model_id,
                    "tag decision reused from request cascade");
                Ok(TaggerOutput {
                    verdict: verdict(reading),
                    usage: None,
                    decision: Some(TaskDecisionMetadata {
                        task_id: String::from("tag").as_str().to_string(),
                        output_kind: self.output_kind.clone(),
                        resolving_tier: self.legacy.ordinal_of(tier),
                        resolving_tier_name: Some(tier.to_string()),
                        resolving_tier_id: Some(tier_id.to_string()),
                        model_id: Some(model_id.clone()),
                        scores: reading.scores.iter().map(|(category_id, score)| {
                            CategoryScore { category_id: category_id.clone(), score: *score }
                        }).collect(),
                    }),
                })
            }
            _ => Err(TagError::Failed(
                "tag task was included in the joint cascade but did not resolve".into()
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::tier::{tname, TierId};
    use std::future::Future;
    use crate::pipeline::runner::{RunOutcome, TaskOutcome};
    use crate::pipeline::runner::Resolution;
    use crate::tagging::{Tagger, TaggerVerdict};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct Fallback(AtomicUsize);
    impl Tagger for Fallback {
        fn model(&self) -> &str { "fallback" }
        fn version(&self) -> &str { "0" }
        fn taxonomy_version(&self) -> &str { "test-taxonomy" }
        fn tag(&self, _: &TagInput) -> impl Future<Output = Result<TaggerOutput, TagError>> + Send {
            self.0.fetch_add(1, Ordering::SeqCst);
            async { Ok(TaggerOutput {
                verdict: TaggerVerdict::NoMatch { nearest: None, embedding: None },
                usage: None,
                decision: None,
            }) }
        }
    }

    #[tokio::test]
    async fn missing_joint_entry_uses_legacy_fallback() {
        let tagger = CascadeTagger::new(JointTagStore::default(), Duration::from_secs(1), "1.1.0", "multiclass", Fallback::default());
        let input = TagInput { request_id: "not-joint".into(), prompt: "help".into(), response: None };
        let result = tagger.tag(&input).await.unwrap();
        assert!(matches!(result.verdict, TaggerVerdict::NoMatch { .. }));
    }

    #[tokio::test]
    async fn requested_joint_tag_is_reused_by_request_id() {
        let store = JointTagStore::default();
        let outcome = RunOutcome {
            tasks: [(
                String::from("tag"),
                TaskOutcome {
                    resolution: Resolution::Tier {
                        tier_id: TierId::from_digest(&[9; 32]),
                        tier: tname("modernbert"),
                        model_id: "shared-encoder".into(),
                        reading: Reading {
                            label: "help_desk".into(),
                            confidence: 0.91,
                            scores: vec![("help_desk".into(), 0.91), ("engineering".into(), 0.09)],
                        },
                    },
                    trace: Vec::new(),
                },
            )].into(),
            tier_calls: Vec::new(),
        };
        let done = tokio::spawn(async move { outcome });
        store.register("joint-1".into(), done, Duration::from_secs(1));
        let tagger = CascadeTagger::new(store, Duration::from_secs(1), "1.1.0", "multiclass", Fallback::default());
        let input = TagInput { request_id: "joint-1".into(), prompt: "help".into(), response: None };
        let result = tagger.tag(&input).await.unwrap();
        assert!(matches!(result.verdict,
            TaggerVerdict::Tagged { primary, .. } if primary.category_id == "help_desk"));
    }
}
