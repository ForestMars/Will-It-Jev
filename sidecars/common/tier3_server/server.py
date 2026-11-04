# sidecars/common/tier3_server/server.py 

import grpc
from concurrent import futures
from . import tier3_pb2, tier3_pb2_grpc

class Tier3Servicer(tier3_pb2_grpc.Tier3ScorerServicer):
    def __init__(self, model, calibration, model_id):
        self.model = model
        self.calibration = calibration
        self.model_id = model_id

    def _route(self, prompt):
        raw = self.model.score(prompt)
        conf = self.calibration.map(raw)
        return raw, conf, "jev_capable" if conf >= 0.5 else "needs_llm"

    def Score(self, request, context):
        raw, conf, label = self._route(request.prompt)
        return tier3_pb2.ScoreResponse(
            raw_score=raw, confidence=conf, label=label, model_id=self.model_id
        )

    def ScoreTasks(self, request, context):
        """Joint scoring. A model that implements `score_tasks(prompt, tasks)` embeds the
        prompt once and returns {task: (label, confidence, [(label, score), ...])} for the
        tasks it has a head for. A model without it can only answer the route task."""
        tasks = list(request.tasks)
        score_tasks = getattr(self.model, "score_tasks", None)
        if score_tasks is not None:
            readings = score_tasks(request.prompt, tasks)
        elif "route" in tasks:
            _, conf, label = self._route(request.prompt)
            readings = {"route": (label, conf, [])}
        else:
            readings = {}

        results = [
            tier3_pb2.TaskScore(
                task=task,
                label=label,
                confidence=confidence,
                scores=[tier3_pb2.LabelScore(label=l, score=s) for l, s in scores],
            )
            for task, (label, confidence, scores) in readings.items()
            if task in tasks
        ]
        return tier3_pb2.ScoreTasksResponse(
            model_id=self.model_id,
            results=results,
            unsupported=[task for task in tasks if task not in readings],
        )


def serve(model, calibration, model_id, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tier3_pb2_grpc.add_Tier3ScorerServicer_to_server(
        Tier3Servicer(model, calibration, model_id), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    print(f"harrier server listening on port {port}", flush=True)
    server.wait_for_termination()