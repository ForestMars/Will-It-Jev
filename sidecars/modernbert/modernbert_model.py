from pathlib import Path


class ModernBertModel:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_length: int = 512,
        jev_label_index: int = 1,
        session=None,
        tokenizer=None,
    ):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if jev_label_index not in (0, 1):
            raise ValueError("jev_label_index must be 0 or 1")

        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)
        if session is None:
            self._require_artifact(model_path, "ONNX model")
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        if tokenizer is None:
            self._require_artifact(tokenizer_path, "tokenizer")
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        self.session = session
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.jev_label_index = jev_label_index
        self.inputs = {item.name: item.type for item in session.get_inputs()}
        supported_inputs = {"input_ids", "attention_mask", "token_type_ids"}
        if "input_ids" not in self.inputs:
            raise ValueError("ONNX model must accept an input_ids tensor")
        unsupported = set(self.inputs) - supported_inputs
        if unsupported:
            raise ValueError(f"unsupported ONNX inputs: {sorted(unsupported)}")

        tokenizer.enable_truncation(max_length=max_length)

    @staticmethod
    def _require_artifact(path: Path, description: str) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{description} is missing or empty: {path}")

    def score(self, prompt: str) -> tuple[float, float, str]:
        import numpy as np

        encoding = self.tokenizer.encode(prompt)
        features = {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask,
            "token_type_ids": encoding.type_ids,
        }
        tensors = {}
        for name, tensor_type in self.inputs.items():
            if tensor_type == "tensor(int64)":
                dtype = np.int64
            elif tensor_type == "tensor(int32)":
                dtype = np.int32
            else:
                raise ValueError(f"unsupported type for ONNX input {name}: {tensor_type}")
            tensors[name] = np.asarray([features[name]], dtype=dtype)

        outputs = self.session.run(None, tensors)
        if not outputs:
            raise ValueError("ONNX model returned no outputs")
        logits = np.asarray(outputs[0], dtype=np.float64)
        if logits.shape == (1, 2):
            logits = logits[0]
        if logits.shape != (2,) or not np.isfinite(logits).all():
            raise ValueError(f"expected finite two-class logits, got shape {logits.shape}")

        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        jev_probability = float(probabilities[self.jev_label_index])
        confidence = float(probabilities.max())
        label = "jev_capable" if jev_probability >= 0.5 else "needs_llm"
        return jev_probability, confidence, label