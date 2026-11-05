import sys
from concurrent import futures
from pathlib import Path

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sidecars" / "common"))

from tier3_server import tier3_pb2, tier3_pb2_grpc
from tier3_server.server import Tier3Servicer


class Identity:
    def map(self, raw):
        return raw


class RouteOnly:
    def score(self, prompt):
        return 0.9


class Joint:
    """Embeds once and answers route and tag together."""

    def __init__(self):
        self.calls = 0

    def score(self, prompt):
        raise AssertionError("ScoreTasks must not call score() when score_tasks exists")

    def score_tasks(self, prompt, tasks):
        self.calls += 1
        readings = {
            "route": ("needs_llm", 0.8, [("jev_capable", 0.2), ("needs_llm", 0.8)]),
            "tag": ("help_desk", 0.7, [("help_desk", 0.7), ("engineering", 0.3)]),
        }
        return {t: readings[t] for t in tasks if t in readings}


def call(model, tasks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    tier3_pb2_grpc.add_Tier3ScorerServicer_to_server(
        Tier3Servicer(model, Identity(), "m"), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = tier3_pb2_grpc.Tier3ScorerStub(channel)
            return stub.ScoreTasks(tier3_pb2.ScoreTasksRequest(prompt="x", tasks=tasks))
    finally:
        server.stop(0)


def test_joint_model_serves_both_tasks_with_one_call():
    model = Joint()
    resp = call(model, ["route", "tag"])
    assert model.calls == 1
    by_task = {r.task: r for r in resp.results}
    assert by_task["route"].label == "needs_llm"
    assert by_task["tag"].label == "help_desk"
    assert [(s.label, round(s.score, 2)) for s in by_task["tag"].scores] == [
        ("help_desk", 0.7),
        ("engineering", 0.3),
    ]
    assert list(resp.unsupported) == []


def test_unknown_task_is_listed_as_unsupported():
    resp = call(Joint(), ["route", "urgency"])
    assert [r.task for r in resp.results] == ["route"]
    assert list(resp.unsupported) == ["urgency"]


def test_route_only_model_answers_route_and_rejects_the_rest():
    resp = call(RouteOnly(), ["route", "tag"])
    assert [(r.task, r.label) for r in resp.results] == [("route", "jev_capable")]
    assert list(resp.unsupported) == ["tag"]
