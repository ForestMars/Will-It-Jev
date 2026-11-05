#!/usr/bin/env python3
"""Train a task head over frozen ONNX encoder representations.

Training rows use one JSON object per line:
  {"prompt": "I need access to Salesforce.", "label": "app_access_req"}

For multilabel tasks, use:
  {"prompt": "The laptop was stolen and my account is exposed.",
   "labels": ["device_loss", "security_incident"]}

Example:
  uv run --with numpy --with onnxruntime --with tokenizers --with scikit-learn \
      --with scipy calibration/train_heads.py \
      --task tag --output-kind multiclass \
      --taxonomy taskonomies/help_desk.json \
      --train taskonomies/data/help_desk_train.jsonl \
      --heldout taskonomies/data/help_desk_heldout.jsonl \
      --encoder models/modernbert-base/encoder.onnx \
      --tokenizer models/modernbert-base/tokenizer.json \
      --output models/modernbert-base/heads/tag/head.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def read_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"{path}:{line_number}: prompt must be a non-empty string")
            rows.append(row)
    if not rows:
        raise ValueError(f"no examples found in {path}")
    return rows


def labels_from_taxonomy(path: str | None) -> tuple[list[str] | None, str | None]:
    if path is None:
        return None, None
    with open(path, encoding="utf-8") as handle:
        taxonomy = json.load(handle)
    labels = [item["name"] for item in taxonomy["categories"]]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError(f"{path}: taxonomy labels must be non-empty and unique")
    return labels, str(taxonomy["version"])


def encode_rows(rows, encoder_path: str, tokenizer_path: str, max_length: int) -> np.ndarray:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=max_length)
    session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    input_types = {item.name: item.type for item in session.get_inputs()}
    embeddings = []
    for row in rows:
        encoded = tokenizer.encode(row["prompt"])
        values = {
            "input_ids": encoded.ids,
            "attention_mask": encoded.attention_mask,
            "token_type_ids": encoded.type_ids,
        }
        feed = {}
        for name, dtype in input_types.items():
            if name not in values:
                raise ValueError(f"encoder has unsupported input {name!r}")
            numpy_dtype = np.int64 if dtype == "tensor(int64)" else np.int32
            feed[name] = np.asarray([values[name]], dtype=numpy_dtype)
        outputs = session.run(None, feed)
        if not outputs:
            raise ValueError("encoder returned no outputs")
        vector = np.asarray(outputs[0], dtype=np.float64)
        if vector.ndim == 3 and vector.shape[0] == 1:
            vector = vector[0, 0, :]
        elif vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        else:
            raise ValueError(f"expected pooled encoder output, got shape {vector.shape}")
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("encoder produced an invalid representation")
        embeddings.append(vector)
    return np.vstack(embeddings)


def single_targets(rows, labels):
    index = {label: position for position, label in enumerate(labels)}
    targets = []
    for row in rows:
        label = row.get("label")
        if label not in index:
            raise ValueError(f"unknown or missing label {label!r}")
        targets.append(index[label])
    return np.asarray(targets, dtype=np.int64)


def multilabel_targets(rows, labels):
    index = {label: position for position, label in enumerate(labels)}
    targets = np.zeros((len(rows), len(labels)), dtype=np.int64)
    for row_index, row in enumerate(rows):
        row_labels = row.get("labels")
        if not isinstance(row_labels, list) or not row_labels:
            raise ValueError(f"row {row_index}: multilabel task requires a non-empty labels array")
        for label in row_labels:
            if label not in index:
                raise ValueError(f"row {row_index}: unknown label {label!r}")
            targets[row_index, index[label]] = 1
    return targets


def fit_temperature(logits, targets, output_kind):
    if len(targets) == 0:
        return 1.0
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        return 1.0

    def loss(temperature):
        scaled = logits / temperature
        if output_kind == "multilabel":
            probability = 1.0 / (1.0 + np.exp(-np.clip(scaled, -60, 60)))
            return float(-np.mean(
                targets * np.log(np.clip(probability, 1e-9, 1))
                + (1 - targets) * np.log(np.clip(1 - probability, 1e-9, 1))
            ))
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probability = exp / exp.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(np.clip(
            probability[np.arange(len(targets)), targets], 1e-9, 1
        ))))

    result = minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded")
    return float(result.x) if result.success else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-kind", required=True,
                        choices=["binary", "multiclass", "multilabel", "ordinal"])
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout")
    parser.add_argument("--taxonomy")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    labels, taxonomy_version = labels_from_taxonomy(args.taxonomy)
    if args.output_kind == "multilabel":
        observed = sorted({label for row in train_rows for label in row.get("labels", [])})
    else:
        observed = sorted({row.get("label") for row in train_rows if row.get("label")})
    if labels is None:
        labels = observed
    elif not set(observed).issubset(set(labels)):
        raise ValueError("training data contains labels absent from the taskonomy")

    if len(labels) < 2:
        raise ValueError("a classifier head requires at least two labels")
    X = encode_rows(train_rows, args.encoder, args.tokenizer, args.max_length)

    if args.output_kind == "multilabel":
        Y = multilabel_targets(train_rows, labels)
        for index, label in enumerate(labels):
            if len(np.unique(Y[:, index])) != 2:
                raise ValueError(f"multilabel training needs positive and negative rows for {label!r}")
        estimator = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced"))
        estimator.fit(X, Y)
        weights = [model.coef_[0].tolist() for model in estimator.estimators_]
        bias = [float(model.intercept_[0]) for model in estimator.estimators_]
    else:
        y = single_targets(train_rows, labels)
        if len(np.unique(y)) != len(labels):
            missing = [labels[i] for i in range(len(labels)) if i not in set(y.tolist())]
            raise ValueError(f"training data is missing categories: {missing}")
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
        estimator.fit(X, y)
        if len(labels) == 2 and estimator.coef_.shape[0] == 1:
            # Convert sklearn's binary positive-class logit into symmetric
            # two-logit form so the shared LinearHead can softmax both labels.
            vector = estimator.coef_[0] / 2.0
            offset = estimator.intercept_[0] / 2.0
            weights = [(-vector).tolist(), vector.tolist()]
            bias = [-float(offset), float(offset)]
        else:
            weights = estimator.coef_.tolist()
            bias = estimator.intercept_.astype(float).tolist()

    temperature = 1.0
    if args.heldout:
        heldout_rows = read_rows(args.heldout)
        X_test = encode_rows(heldout_rows, args.encoder, args.tokenizer, args.max_length)
        if args.output_kind == "multilabel":
            y_test = multilabel_targets(heldout_rows, labels)
            logits = X_test @ np.asarray(weights, dtype=np.float64).T + np.asarray(bias)
        else:
            y_test = single_targets(heldout_rows, labels)
            logits = X_test @ np.asarray(weights, dtype=np.float64).T + np.asarray(bias)
        temperature = fit_temperature(logits, y_test, args.output_kind)

    artifact = {
        "task_id": args.task,
        "output_kind": args.output_kind,
        "labels": labels,
        "weights": weights,
        "bias": bias,
        "temperature": temperature,
        "taxonomy_version": taxonomy_version,
        "training_examples": len(train_rows),
        "heldout_examples": 0 if not args.heldout else len(heldout_rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "task": args.task,
        "output_kind": args.output_kind,
        "labels": len(labels),
        "train_examples": len(train_rows),
        "heldout_examples": artifact["heldout_examples"],
        "temperature": temperature,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
