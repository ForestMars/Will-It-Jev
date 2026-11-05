from pathlib import Path
import sys

COMMON = Path(__file__).resolve().parents[2] / "sidecars" / "common"
HARRIER = Path(__file__).resolve().parents[2] / "sidecars" / "harrier"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(HARRIER))

from harrier_model import HarrierModel
from tier3_server import tier3_pb2
from tier3_server.heads import HeadRegistry, LinearHead, PrototypeHead
from tier3_server.server import Tier3Servicer


class CountingEncoder:
    def __init__(self):
        self.calls = 0

    def create_embedding(self, text):
        self.calls += 1
        assert text.startswith(HarrierModel.INSTRUCTION)
        return {"data": [{"embedding": [1.0, 0.0]}]}


class Identity:
    def map(self, raw):
        return raw


def model():
    instance = HarrierModel.__new__(HarrierModel)
    instance.llm = CountingEncoder()
    instance.heads = HeadRegistry({
        "route": PrototypeHead(
            "route",
            [([1.0, 0.0], "jev_capable"), ([-1.0, 0.0], "needs_llm")],
            "binary",
        ),
        "tag": LinearHead(
            "tag",
            ["help_desk", "engineering"],
            "multiclass",
            [[1.0, 0.0], [0.0, 1.0]],
        ),
    })
    return instance


def test_harrier_encodes_once_for_two_heterogeneous_heads():
    instance = model()
    result = instance.score_tasks("help me", ["route", "tag"])

    assert instance.llm.calls == 1
    assert result["route"][0] == "jev_capable"
    assert result["route"][3] == "binary"
    assert result["tag"][0] == "help_desk"
    assert result["tag"][3] == "multiclass"


def test_harrier_server_serializes_task_specific_output_kinds():
    instance = model()
    servicer = Tier3Servicer(instance, Identity(), "harrier-test")
    response = servicer.ScoreTasks(
        tier3_pb2.ScoreTasksRequest(prompt="help me", tasks=["route", "tag"]),
        context=None,
    )

    by_task = {row.task: row for row in response.results}
    assert instance.llm.calls == 1
    assert by_task["route"].output_kind == tier3_pb2.BINARY
    assert by_task["tag"].output_kind == tier3_pb2.MULTICLASS
    assert list(response.unsupported) == []
