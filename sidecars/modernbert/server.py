import os
import sys
from concurrent import futures
from pathlib import Path

import grpc

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "sidecars" / "common"))

from tier3_server import tier3_pb2, tier3_pb2_grpc
from tier3_server.server import Tier3Servicer
from modernbert_model import ModernBertModel


class IdentityCalibration:
    def map(self, raw):
        return raw


class ModernBertServicer(Tier3Servicer):
    """ModernBERT adapter using the same Score/ScoreTasks server contract."""

    def __init__(self, model):
        super().__init__(model, IdentityCalibration(), "modernbert-onnx")


def build_model():
    model_root = REPO_ROOT / "models" / "modernbert-base"
    model_path = Path(os.environ.get("MODERNBERT_ONNX_PATH", model_root / "model_int8.onnx"))
    tokenizer_path = Path(
        os.environ.get("MODERNBERT_TOKENIZER_PATH", model_root / "tokenizer.json")
    )
    encoder_path = Path(
        os.environ.get("MODERNBERT_ENCODER_ONNX_PATH", model_root / "encoder.onnx")
    )
    heads_dir = Path(
        os.environ.get("MODERNBERT_HEADS_DIR", model_root / "heads")
    )
    return ModernBertModel(
        str(model_path),
        str(tokenizer_path),
        max_length=int(os.environ.get("MODERNBERT_MAX_LENGTH", "512")),
        jev_label_index=int(os.environ.get("MODERNBERT_JEV_LABEL_INDEX", "1")),
        encoder_path=str(encoder_path),
        heads_dir=str(heads_dir),
    )


def serve(model, port: int = 50052):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tier3_pb2_grpc.add_Tier3ScorerServicer_to_server(
        ModernBertServicer(model), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"ModernBERT shared-encoder sidecar listening on port {port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve(build_model(), int(os.environ.get("MODERNBERT_PORT", "50052")))
