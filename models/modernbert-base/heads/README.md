# Task-head artifacts

Each model may load task-local heads from heads/<task>/head.json.

A linear head artifact has this shape:

~~~json
{
  "task_id": "tag",
  "output_kind": "multiclass",
  "labels": ["app_access_req", "account_pw"],
  "weights": [[0.1, 0.2], [-0.1, 0.05]],
  "bias": [0.0, 0.0],
  "temperature": 1.0,
  "taxonomy_version": "1.1.0"
}
~~~

weights is a matrix with one row per label and one column per encoder
dimension. bias has one value per label. Binary and multiclass heads use
softmax; multilabel heads use independent sigmoid probabilities; ordinal heads
use an ordered label list with a softmax distribution. Calibration temperature
must be fit on held-out data.

calibration/train_heads.py writes this format for the ModernBERT pooled
encoder. Harrier can load the same weight artifacts when their dimensions match
its embedding representation. It can also load exemplars.jsonl in a task
directory, with an optional head.json containing output_kind, and will
evaluate those exemplar heads against the one shared prompt embedding.

The existing route model remains available as a fallback. Joint route/tag
execution occurs only when all requested heads are present in the selected
sidecar. A missing head is returned as unsupported; it is never simulated by
running a second encoder call.
