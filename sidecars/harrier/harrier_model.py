# sidecars/harrier/harrier_model.py
"""Harrier embedding encoder with task-specific heads over one shared vector."""
import hashlib
import json
import os
import time
from pathlib import Path

from llama_cpp import Llama

from tier3_server.heads import HeadRegistry, PrototypeHead

CACHE_VERSION = 2


class HarrierModel:
    def __init__(
        self,
        model_path: str,
        exemplars_path: str,
        n_ctx: int = 512,
        heads_dir: str | None = None,
    ):
        print("loading model", flush=True)
        started = time.perf_counter()
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            embedding=True,
            verbose=False,
            pooling_type=3,
            n_gpu_layers=1,
        )
        print(f"model loaded in {time.perf_counter() - started:.2f}s", flush=True)

        self.exemplar_vectors = self._load_vector_cache(model_path, exemplars_path, n_ctx)
        head_root = Path(heads_dir or os.environ.get(
            "HARRIER_HEADS_DIR",
            str(Path(model_path).parent / "heads"),
        ))
        self.heads = HeadRegistry.load(head_root)

        # The existing route exemplar set remains a valid binary head. Additional
        # task heads can be linear weights in head.json or task-local exemplars.
        if self.heads.get("route") is None:
            self.heads.register(PrototypeHead("route", self.exemplar_vectors, "binary"))

        for task_dir in (sorted(head_root.iterdir()) if head_root.exists() else []):
            if not task_dir.is_dir() or task_dir.name == "route":
                continue
            if self.heads.get(task_dir.name) is not None:
                continue
            exemplar_file = task_dir / "exemplars.jsonl"
            if not exemplar_file.is_file():
                continue
            config_file = task_dir / "head.json"
            output_kind = "multiclass"
            taxonomy_version = None
            if config_file.is_file():
                with config_file.open(encoding="utf-8") as handle:
                    head_config = json.load(handle)
                    output_kind = head_config.get("output_kind", output_kind)
                    taxonomy_version = head_config.get("taxonomy_version")
            vectors = self._load_vector_cache(model_path, str(exemplar_file), n_ctx)
            self.heads.register(PrototypeHead(
                task_dir.name, vectors, output_kind, taxonomy_version
            ))

        print(f"loaded task heads: {', '.join(self.heads.task_ids)}", flush=True)

    @staticmethod
    def _load_exemplars(path):
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row.get("prompt"), str) or not row.get("prompt"):
                        raise ValueError(f"invalid exemplar prompt in {path}")
                    if not isinstance(row.get("label"), str) or not row.get("label"):
                        raise ValueError(f"invalid exemplar label in {path}")
                    rows.append(row)
        if not rows:
            raise ValueError(f"no exemplars loaded from {path}")
        return rows

    def _load_vector_cache(self, model_path, exemplars_path, n_ctx):
        started = time.perf_counter()
        exemplars = self._load_exemplars(exemplars_path)
        key = self._cache_key(model_path, n_ctx, exemplars)
        cache_path = f"{exemplars_path}.{key[:16]}.embcache.json"
        cached = self._read_cache(cache_path, key)
        if cached is not None:
            print(
                f"loaded {len(cached)} cached vectors from {exemplars_path} "
                f"in {time.perf_counter() - started:.2f}s",
                flush=True,
            )
            return cached

        vectors = []
        for index, row in enumerate(exemplars):
            vectors.append((self._embed(row["prompt"]), row["label"]))
            if index and index % 100 == 0:
                print(f"embedded {index} exemplars from {exemplars_path}", flush=True)
        self._write_cache(cache_path, key, vectors)
        print(
            f"embedded {len(vectors)} exemplars from {exemplars_path} "
            f"in {time.perf_counter() - started:.2f}s",
            flush=True,
        )
        return vectors

    @staticmethod
    def _cache_key(model_path, n_ctx, exemplars):
        stat = os.stat(model_path)
        digest = hashlib.sha256()
        digest.update(
            f"v{CACHE_VERSION}|{os.path.abspath(model_path)}|{stat.st_size}|"
            f"{stat.st_mtime_ns}|{n_ctx}|".encode()
        )
        for row in exemplars:
            digest.update(
                json.dumps([row["prompt"], row["label"]], ensure_ascii=False).encode()
            )
        return digest.hexdigest()

    @staticmethod
    def _read_cache(path, key):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("key") != key:
                return None
            return [(row["vec"], row["label"]) for row in data["rows"]]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _write_cache(path, key, vectors):
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(
                    {"key": key, "rows": [{"vec": vector, "label": label}
                                            for vector, label in vectors]},
                    handle,
                )
            os.replace(tmp, path)
        except OSError as error:
            print(f"embedding cache write failed for {path}: {error}", flush=True)

    INSTRUCTION = "Instruct: Retrieve semantically similar text\nQuery: "

    def _embed(self, text, is_query=False):
        if is_query:
            text = self.INSTRUCTION + text
        response = self.llm.create_embedding(text)
        vector = response["data"][0]["embedding"]
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector

    def score_tasks(self, prompt: str, tasks: list[str]):
        """Encode prompt once, then invoke each requested task's own head."""
        representation = self._embed(prompt, is_query=True)
        return self.heads.score_tasks(representation, list(dict.fromkeys(tasks)))

    def score(self, prompt: str) -> float:
        """Legacy Score contract: return P(jev_capable) for route calibration."""
        reading = self.score_tasks(prompt, ["route"]).get("route")
        if reading is None:
            raise RuntimeError("Harrier route head is unavailable")
        _, _, scores, _ = reading
        return next((score for label, score in scores if label == "jev_capable"), 0.0)
