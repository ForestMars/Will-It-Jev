# Taskonomy training and held-out data

The taskonomy JSON defines the canonical label IDs, descriptions, and
classifier criteria. It is not itself a labeled training corpus.

For help_desk, the head-training inputs are expected at:

- taskonomies/data/help_desk_train.jsonl
- taskonomies/data/help_desk_heldout.jsonl

Each line is one labeled example:

~~~json
{"prompt":"I need access to Salesforce.","label":"app_access_req"}
~~~

Use the stable category name values from help_desk.json as labels. Keep
near-neighbor categories represented in both splits. The held-out set must be
disjoint from training examples and should include realistic paraphrases,
ambiguous cases, and hard negatives. Do not manufacture held-out examples by
copying or lightly paraphrasing training rows.

For a multilabel task, replace label with a non-empty labels array:

~~~json
{"prompt":"The missing laptop may expose customer data.","labels":["device_loss","security_incident"]}
~~~

calibration/train_heads.py consumes these JSONL files, embeds prompts with
the pooled ModernBERT encoder, trains one task-local projection, and writes
models/modernbert-base/heads/<task>/head.json. The artifact includes output
kind, labels, weights, bias, temperature, and the taskonomy version.

The repository currently has the taxonomy definitions and illustrative
examples, but does not yet have these train/held-out JSONL datasets. That is a
data prerequisite for producing a calibrated help-desk head; no training
examples are fabricated by this implementation.
