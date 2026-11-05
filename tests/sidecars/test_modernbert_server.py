import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SIDECAR = Path(__file__).resolve().parents[2] / "sidecars" / "modernbert"
sys.path.insert(0, str(SIDECAR))

from modernbert_model import ModernBertModel
from server import ModernBertServicer
from tier3_server import tier3_pb2


class FakeTokenizer:
    def enable_truncation(self, max_length):
        self.max_length = max_length

    def encode(self, prompt):
        return SimpleNamespace(
            ids=[101, 42, 102],
            attention_mask=[1, 1, 1],
            type_ids=[0, 0, 0],
        )


class FakeSession:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.feed = None
        self.calls = 0

    def get_inputs(self):
        return [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
            SimpleNamespace(name="token_type_ids", type="tensor(int64)"),
        ]

    def run(self, _output_names, feed):
        self.calls += 1
        self.feed = feed
        return [self.logits]


class FakeEncoderSession(FakeSession):
    def __init__(self, representation):
        super().__init__([representation])


def runner(logits, jev_label_index=1):
    session = FakeSession(logits)
    model = ModernBertModel(
        "unused.onnx",
        "unused-tokenizer.json",
        session=session,
        tokenizer=FakeTokenizer(),
        jev_label_index=jev_label_index,
    )
    return model, session


def test_score_maps_two_class_logits_and_builds_batch_inputs():
    model, session = runner([[0.0, 3.0]])

    jev_probability, confidence, label = model.score("test prompt")

    assert jev_probability > 0.9
    assert confidence == pytest.approx(jev_probability)
    assert label == "jev_capable"
    assert session.feed["input_ids"].shape == (1, 3)
    assert session.feed["attention_mask"].tolist() == [[1, 1, 1]]


def test_label_order_is_configurable():
    model, _ = runner([[0.0, 3.0]], jev_label_index=0)

    jev_probability, _, label = model.score("test prompt")

    assert jev_probability < 0.1
    assert label == "needs_llm"


def test_rejects_non_classification_outputs():
    model, _ = runner([[0.1, 0.2, 0.7]])

    with pytest.raises(ValueError, match="two-class logits"):
        model.score("test prompt")


def write_head(root, task, labels, output_kind, weights):
    directory = root / task
    directory.mkdir()
    (directory / "head.json").write_text(json.dumps({
        "task_id": task,
        "labels": labels,
        "output_kind": output_kind,
        "weights": weights,
        "bias": [0.0 for _ in labels],
    }))


def test_score_tasks_runs_one_encoder_for_route_and_tag(tmp_path):
    write_head(tmp_path, "route", ["jev_capable", "needs_llm"], "binary", [[1, 0], [-1, 0]])
    write_head(tmp_path, "tag", ["help_desk", "engineering"], "multiclass", [[1, 0], [0, 1]])
    legacy = FakeSession([[0.0, 2.0]])
    encoder = FakeEncoderSession([[1.0, 0.0]])
    model = ModernBertModel(
        "unused.onnx",
        "unused-tokenizer.json",
        session=legacy,
        tokenizer=FakeTokenizer(),
        encoder_session=encoder,
        heads_dir=str(tmp_path),
    )

    readings = model.score_tasks("help me", ["route", "tag"])

    assert encoder.calls == 1
    assert legacy.calls == 0
    assert readings["route"][3] == "binary"
    assert readings["tag"][0] == "help_desk"
    assert readings["tag"][3] == "multiclass"


def test_modernbert_servicer_uses_common_typed_multi_task_contract(tmp_path):
    write_head(tmp_path, "route", ["jev_capable", "needs_llm"], "binary", [[1, 0], [-1, 0]])
    write_head(tmp_path, "tag", ["help_desk", "engineering"], "multiclass", [[1, 0], [0, 1]])
    encoder = FakeEncoderSession([[1.0, 0.0]])
    model = ModernBertModel(
        "unused.onnx",
        "unused-tokenizer.json",
        session=FakeSession([[0.0, 1.0]]),
        tokenizer=FakeTokenizer(),
        encoder_session=encoder,
        heads_dir=str(tmp_path),
    )
    response = ModernBertServicer(model).ScoreTasks(
        tier3_pb2.ScoreTasksRequest(prompt="help me", tasks=["route", "tag"]),
        context=None,
    )

    rows = {row.task: row for row in response.results}
    assert encoder.calls == 1
    assert rows["route"].output_kind == tier3_pb2.BINARY
    assert rows["tag"].output_kind == tier3_pb2.MULTICLASS
    assert list(response.unsupported) == []


def test_grpc_service_returns_router_contract():
    class FakeModel:
        def score(self, prompt):
            assert prompt == "a request"
            return 0.8, 0.8, "jev_capable"

    response = ModernBertServicer(FakeModel()).Score(
        tier3_pb2.ScoreRequest(prompt="a request"), context=None
    )

    assert response.raw_score == pytest.approx(0.8)
    assert response.confidence == pytest.approx(0.8)
    assert response.label == "jev_capable"
    assert response.model_id == "modernbert-onnx"
