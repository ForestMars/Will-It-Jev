import os
import sys
from concurrent import futures
from pathlib import Path

import grpc

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "sidecars" / "common"))

from tier3_server import tier3_pb2, tier3_pb2_grpc
from modernbert_model import ModernBertModel


class ModernBertServicer(tier3_pb2_grpc.Tier3ScorerServicer):
	def __init__(self, model):
		self.model = model

	def Score(self, request, context):
		try:
			jev_probability, confidence, label = self.model.score(request.prompt)
		except Exception as error:
			context.abort(grpc.StatusCode.INTERNAL, str(error))
		return tier3_pb2.ScoreResponse(
			raw_score=jev_probability,
			confidence=confidence,
			label=label,
			model_id="modernbert-onnx",
		)


def build_model():
	model_path = Path(
		os.environ.get(
			"MODERNBERT_ONNX_PATH",
			REPO_ROOT / "models" / "modernbert-base" / "model_int8.onnx",
		)
	)
	tokenizer_path = Path(
		os.environ.get(
			"MODERNBERT_TOKENIZER_PATH",
			REPO_ROOT / "models" / "modernbert-base" / "tokenizer.json",
		)
	)
	return ModernBertModel(
		str(model_path),
		str(tokenizer_path),
		max_length=int(os.environ.get("MODERNBERT_MAX_LENGTH", "512")),
		jev_label_index=int(os.environ.get("MODERNBERT_JEV_LABEL_INDEX", "1")),
	)


def serve(model, port: int = 50052):
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
	tier3_pb2_grpc.add_Tier3ScorerServicer_to_server(
		ModernBertServicer(model), server
	)
	server.add_insecure_port(f"[::]:{port}")
	server.start()
	print(f"ModernBERT ONNX sidecar listening on port {port}", flush=True)
	server.wait_for_termination()


if __name__ == "__main__":
	port = int(os.environ.get("MODERNBERT_PORT", "50052"))
	serve(build_model(), port)
