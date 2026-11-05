"""Shared output-head implementations for Tier 3 encoder sidecars.

A head consumes an already-computed representation. Encoders own tokenization and
representation construction; heads own only task-specific projection and output shape.
Model artifact layout: models/<model>/heads/<task>/head.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np


class Head(Protocol):
    task_id: str
    output_kind: str
    taxonomy_version: str | None

    def forward(self, representation: np.ndarray):
        """Return (label, confidence, [(label, score), ...], output_kind)."""


class LinearHead:
    def __init__(
        self,
        task_id: str,
        labels: list[str],
        output_kind: str,
        weights,
        bias=None,
        temperature: float = 1.0,
        taxonomy_version: str | None = None,
    ):
        if not task_id.strip():
            raise ValueError("head task_id must be non-empty")
        if not labels or len(set(labels)) != len(labels) or any(not x.strip() for x in labels):
            raise ValueError(f"{task_id}: labels must be non-empty and unique")
        if output_kind not in {"binary", "multiclass", "multilabel", "ordinal"}:
            raise ValueError(f"{task_id}: unsupported output_kind {output_kind!r}")
        matrix = np.asarray(weights, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(labels):
            raise ValueError(
                f"{task_id}: weights must have shape (labels, hidden_size), got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{task_id}: weights contain non-finite values")
        offset = np.zeros(len(labels), dtype=np.float64) if bias is None else np.asarray(bias, dtype=np.float64)
        if offset.shape != (len(labels),) or not np.isfinite(offset).all():
            raise ValueError(f"{task_id}: bias must contain one finite value per label")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"{task_id}: temperature must be finite and positive")

        self.task_id = task_id
        self.labels = tuple(labels)
        self.output_kind = output_kind
        self.weights = matrix
        self.bias = offset
        self.temperature = float(temperature)
        self.taxonomy_version = taxonomy_version

    def forward(self, representation: np.ndarray):
        vector = np.asarray(representation, dtype=np.float64).reshape(-1)
        if vector.shape != (self.weights.shape[1],):
            raise ValueError(
                f"{self.task_id}: expected representation width {self.weights.shape[1]}, got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"{self.task_id}: representation contains non-finite values")
        logits = (self.weights @ vector + self.bias) / self.temperature

        if self.output_kind == "multilabel":
            clipped = np.clip(logits, -60.0, 60.0)
            probabilities = 1.0 / (1.0 + np.exp(-clipped))
        else:
            shifted = logits - np.max(logits)
            exp = np.exp(shifted)
            probabilities = exp / exp.sum()

        index = int(np.argmax(probabilities))
        scores = [(label, float(probabilities[i])) for i, label in enumerate(self.labels)]
        return self.labels[index], float(probabilities[index]), scores, self.output_kind


class HeadRegistry:
    """Loads independent task heads over a common encoder representation."""

    def __init__(self, heads: dict[str, Head] | None = None):
        self._heads: dict[str, Head] = dict(heads or {})

    @classmethod
    def load(cls, root: str | Path) -> "HeadRegistry":
        root = Path(root)
        heads: dict[str, Head] = {}
        if not root.exists():
            return cls()
        for config_path in sorted(root.glob("*/head.json")):
            with config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            task_id = str(config.get("task_id") or config_path.parent.name)
            # Exemplar-based sidecars consume exemplars.jsonl in their own
            # model adapter; this registry owns only weight-based heads.
            if "weights" not in config:
                continue
            if task_id in heads:
                raise ValueError(f"duplicate head for task {task_id!r}")
            heads[task_id] = LinearHead(
                task_id=task_id,
                labels=list(config["labels"]),
                output_kind=str(config["output_kind"]),
                weights=config["weights"],
                bias=config.get("bias"),
                temperature=float(config.get("temperature", 1.0)),
                taxonomy_version=config.get("taxonomy_version"),
            )
        return cls(heads)

    def register(self, head: Head) -> None:
        if head.task_id in self._heads:
            raise ValueError(f"head already registered for task {head.task_id!r}")
        self._heads[head.task_id] = head

    def get(self, task_id: str) -> Head | None:
        return self._heads.get(task_id)

    def score_tasks(self, representation: np.ndarray, tasks: list[str]):
        readings = {}
        for task_id in tasks:
            head = self._heads.get(task_id)
            if head is not None:
                readings[task_id] = head.forward(representation)
        return readings

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._heads))


class PrototypeHead:
    """Nearest-exemplar head over a shared, already-computed embedding."""

    def __init__(
        self,
        task_id: str,
        vectors,
        output_kind: str = "multiclass",
        taxonomy_version: str | None = None,
    ):
        if output_kind not in {"binary", "multiclass", "multilabel", "ordinal"}:
            raise ValueError(f"{task_id}: unsupported output_kind {output_kind!r}")
        rows = [(np.asarray(vector, dtype=np.float64).reshape(-1), str(label))
                for vector, label in vectors]
        if not rows:
            raise ValueError(f"{task_id}: prototype head needs at least one exemplar")
        width = rows[0][0].shape[0]
        if any(vector.shape != (width,) or not np.isfinite(vector).all() for vector, _ in rows):
            raise ValueError(f"{task_id}: exemplar vectors must have one finite shared width")
        self.task_id = task_id
        self.output_kind = output_kind
        self.taxonomy_version = taxonomy_version
        self.rows = rows
        self.labels = tuple(sorted({label for _, label in rows}))

    def forward(self, representation: np.ndarray):
        query = np.asarray(representation, dtype=np.float64).reshape(-1)
        if query.shape != self.rows[0][0].shape:
            raise ValueError(f"{self.task_id}: embedding width mismatch")
        best = {label: -1.0 for label in self.labels}
        for vector, label in self.rows:
            similarity = float(np.dot(query, vector))
            best[label] = max(best[label], similarity)
        logits = np.asarray([best[label] for label in self.labels], dtype=np.float64)
        if self.output_kind == "multilabel":
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        else:
            shifted = logits - np.max(logits)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        scores = [(label, float(probabilities[i])) for i, label in enumerate(self.labels)]
        return self.labels[index], float(probabilities[index]), scores, self.output_kind
