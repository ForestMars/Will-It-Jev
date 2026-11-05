from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from tier3_server.heads import HeadRegistry


class ModernBertModel:
    """ModernBERT encoder plus task-specific linear heads.

    The existing fused route ONNX remains a compatibility fallback. Joint
    multi-task inference requires an encoder ONNX whose first output is the
    pooled CLS representation and one head.json per requested task.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_length: int = 512,
        jev_label_index: int = 1,
        session=None,
        tokenizer=None,
        encoder_path: str | None = None,
        encoder_session=None,
        heads_dir: str | None = None,
    ):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if jev_label_index not in (0, 1):
            raise ValueError("jev_label_index must be 0 or 1")

        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)
        if session is None:
            self._require_artifact(model_path, "legacy route ONNX model")
            import onnxruntime as ort
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

        if tokenizer is None:
            self._require_artifact(tokenizer_path, "tokenizer")
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        if encoder_session is None and encoder_path:
            encoder_file = Path(encoder_path)
            if encoder_file.is_file() and encoder_file.stat().st_size > 0:
                import onnxruntime as ort
                encoder_session = ort.InferenceSession(
                    str(encoder_file), providers=["CPUExecutionProvider"]
                )

        self.session = session
        self.encoder_session = encoder_session
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.jev_label_index = jev_label_index
        self.inputs = self._input_types(session)
        self.encoder_inputs = self._input_types(encoder_session) if encoder_session else {}
        self.heads = HeadRegistry.load(
            heads_dir or (model_path.parent / "heads")
        )

        tokenizer.enable_truncation(max_length=max_length)

    @staticmethod
    def _require_artifact(path: Path, description: str) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{description} is missing or empty: {path}")

    @staticmethod
    def _input_types(session):
        if session is None:
            return {}
        inputs = {item.name: item.type for item in session.get_inputs()}
        supported = {"input_ids", "attention_mask", "token_type_ids"}
        if "input_ids" not in inputs:
            raise ValueError("ONNX model must accept an input_ids tensor")
        unsupported = set(inputs) - supported
        if unsupported:
            raise ValueError(f"unsupported ONNX inputs: {sorted(unsupported)}")
        return inputs

    def _features(self, prompt: str):
        encoding = self.tokenizer.encode(prompt)
        return {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask,
            "token_type_ids": encoding.type_ids,
        }

    @staticmethod
    def _tensors(features, input_types):
        import numpy as np
        tensors = {}
        for name, tensor_type in input_types.items():
            if tensor_type == "tensor(int64)":
                dtype = np.int64
            elif tensor_type == "tensor(int32)":
                dtype = np.int32
            else:
                raise ValueError(f"unsupported ONNX input type for {name}: {tensor_type}")
            tensors[name] = np.asarray([features[name]], dtype=dtype)
        return tensors

    def encode(self, prompt: str):
        """Return one pooled representation for every requested task head."""
        if self.encoder_session is None:
            raise RuntimeError(
                "ModernBERT shared encoder artifact is unavailable; set "
                "MODERNBERT_ENCODER_ONNX_PATH to an exported pooled encoder"
            )
        import numpy as np
        outputs = self.encoder_session.run(
            None, self._tensors(self._features(prompt), self.encoder_inputs)
        )
        if not outputs:
            raise ValueError("ModernBERT encoder returned no outputs")
        representation = np.asarray(outputs[0], dtype=np.float64)
        if representation.ndim == 3 and representation.shape[0] == 1:
            # The exporter exposes the pooled CLS token as sequence position 0.
            representation = representation[0, 0, :]
        elif representation.ndim == 2 and representation.shape[0] == 1:
            representation = representation[0]
        else:
            raise ValueError(
                "expected encoder output shape (1, hidden) or (1, sequence, hidden), "
                f"got {representation.shape}"
            )
        if representation.ndim != 1 or not np.isfinite(representation).all():
            raise ValueError("encoder returned an invalid pooled representation")
        return representation

    def _legacy_route(self, prompt: str):
        import numpy as np
        outputs = self.session.run(
            None, self._tensors(self._features(prompt), self.inputs)
        )
        if not outputs:
            raise ValueError("ONNX route model returned no outputs")
        logits = np.asarray(outputs[0], dtype=np.float64)
        if logits.shape == (1, 2):
            logits = logits[0]
        if logits.shape != (2,) or not np.isfinite(logits).all():
            raise ValueError(f"expected finite two-class logits, got shape {logits.shape}")

        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        p_jev = float(probabilities[self.jev_label_index])
        label = "jev_capable" if p_jev >= 0.5 else "needs_llm"
        confidence = p_jev if label == "jev_capable" else 1.0 - p_jev
        return p_jev, confidence, label

    def score(self, prompt: str) -> tuple[float, float, str]:
        # When available, the route head also consumes the shared encoder.
        if self.encoder_session is not None and self.heads.get("route") is not None:
            label, confidence, scores, _ = self.heads.get("route").forward(self.encode(prompt))
            p_jev = next((score for name, score in scores if name == "jev_capable"), 0.0)
            return float(p_jev), float(confidence), str(label)
        return self._legacy_route(prompt)

    def score_tasks(self, prompt: str, tasks: list[str]):
        """Encode once and execute all requested heads against that vector.

        If a requested head is absent, this model answers only the legacy route
        head (when requested) and marks the remaining task IDs unsupported. This
        avoids quietly doing a second encoder pass for an incomplete head set.
        """
        requested = list(dict.fromkeys(tasks))
        if self.encoder_session is None:
            # Legacy fused route model is available for compatibility, but it
            # cannot impersonate a multi-head encoder. Other requested tasks
            # remain explicitly unsupported.
            if "route" not in requested:
                return {}
            p_jev, confidence, label = self._legacy_route(prompt)
            return {
                "route": (
                    label,
                    confidence,
                    [("jev_capable", p_jev), ("needs_llm", 1.0 - p_jev)],
                    "binary",
                )
            }
        # Encode once even when only a subset of requested heads is installed.
        # The shared server will mark missing heads unsupported; it must not
        # rerun the legacy route model to manufacture a second path.
        representation = self.encode(prompt)
        return self.heads.score_tasks(representation, requested)
