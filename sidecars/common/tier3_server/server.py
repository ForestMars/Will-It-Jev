# sidecars/common/tier3_server/server.py

import grpc
from concurrent import futures

from . import tier3_pb2, tier3_pb2_grpc


class Tier3Servicer(tier3_pb2_grpc.Tier3ScorerServicer):
    """Shared Tier 3 transport contract for sidecars with one or many heads."""

    def __init__(self, model, calibration, model_id):
        self.model = model
        self.calibration = calibration
        self.model_id = model_id

    def _route(self, prompt):
        output = self.model.score(prompt)
        # ModernBERT's compatibility adapter already returns calibrated
        # (P(jev), selected-class confidence, label). Scalar models such as
        # Harrier return raw P(jev) and use the common calibration path below.
        if isinstance(output, (tuple, list)) and len(output) == 3:
            raw, confidence, label = output
            p_jev = min(1.0, max(0.0, float(raw)))
            confidence = min(1.0, max(0.0, float(confidence)))
            label = str(label)
        else:
            raw = float(output)
            p_jev = float(self.calibration.map(raw))
            p_jev = min(1.0, max(0.0, p_jev))
            label = "jev_capable" if p_jev >= 0.5 else "needs_llm"
            # The cascade threshold is confidence in the selected class.
            confidence = p_jev if label == "jev_capable" else 1.0 - p_jev
        scores = [("jev_capable", p_jev), ("needs_llm", 1.0 - p_jev)]
        return raw, confidence, label, scores

    def Score(self, request, context):
        try:
            raw, confidence, label, _ = self._route(request.prompt)
        except Exception as error:
            context.abort(grpc.StatusCode.INTERNAL, str(error))
        return tier3_pb2.ScoreResponse(
            raw_score=raw,
            confidence=confidence,
            label=label,
            model_id=self.model_id,
        )

    def ScoreTasks(self, request, context):
        """Run every requested head over one model-owned shared representation."""
        specs = {spec.task: spec for spec in request.task_specs}
        requested = list(request.tasks) or [spec.task for spec in request.task_specs]
        tasks = list(dict.fromkeys(requested))
        score_tasks = getattr(self.model, "score_tasks", None)

        try:
            if callable(score_tasks):
                readings = score_tasks(request.prompt, tasks)
            elif "route" in tasks:
                # Legacy route-only sidecar. It answers route once and marks
                # every other task explicitly unsupported.
                _, confidence, label, scores = self._route(request.prompt)
                readings = {"route": (label, confidence, scores, "binary")}
            else:
                readings = {}
        except Exception as error:
            context.abort(grpc.StatusCode.INTERNAL, str(error))

        results = []
        unsupported = []
        for task in tasks:
            reading = readings.get(task)
            if reading is None:
                unsupported.append(task)
                continue
            try:
                if len(reading) == 3:
                    label, confidence, scores = reading
                    output_kind = tier3_pb2.OUTPUT_KIND_UNSPECIFIED
                else:
                    label, confidence, scores, kind = reading
                    output_kind = {
                        "binary": tier3_pb2.BINARY,
                        "multiclass": tier3_pb2.MULTICLASS,
                        "multilabel": tier3_pb2.MULTILABEL,
                        "ordinal": tier3_pb2.ORDINAL,
                    }.get(str(kind).lower(), tier3_pb2.OUTPUT_KIND_UNSPECIFIED)
                spec = specs.get(task)
                if spec is not None:
                    # Reject stale artifacts whose output shape or taxonomy label
                    # space differs from the task contract sent by the cascade.
                    if spec.output_kind and output_kind != spec.output_kind:
                        unsupported.append(task)
                        continue
                    if spec.taxonomy_version:
                        registry = getattr(self.model, "heads", None)
                        head = registry.get(task) if registry is not None else None
                        if head is None or getattr(head, "taxonomy_version", None) != spec.taxonomy_version:
                            unsupported.append(task)
                            continue
                    expected_labels = {item.id for item in spec.labels}
                    actual_labels = {str(name) for name, _ in scores}
                    if expected_labels and expected_labels != actual_labels:
                        unsupported.append(task)
                        continue
                results.append(
                    tier3_pb2.TaskScore(
                        task=task,
                        label=str(label),
                        confidence=float(confidence),
                        scores=[
                            tier3_pb2.LabelScore(label=str(name), score=float(score))
                            for name, score in scores
                        ],
                        output_kind=output_kind,
                    )
                )
            except Exception as error:
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"invalid reading for task {task!r}: {error}",
                )

        return tier3_pb2.ScoreTasksResponse(
            model_id=self.model_id,
            results=results,
            unsupported=unsupported,
        )


def serve(model, calibration, model_id, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tier3_pb2_grpc.add_Tier3ScorerServicer_to_server(
        Tier3Servicer(model, calibration, model_id), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    print(f"tier3 server listening on port {port}", flush=True)
    server.wait_for_termination()
