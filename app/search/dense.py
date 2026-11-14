"""Embedding models for dense retrieval.

A model is named "<backend>:<model>":
    fastembed:BAAI/bge-small-en-v1.5    384-dim, ONNX on CPU (default; see DECISIONS.md)
    model2vec:minishlab/potion-base-8M  256-dim static embeddings, tiny and very fast
A bare name is accepted too (names starting with "minishlab/" mean model2vec).

Models are loaded lazily and shared. If one cannot be loaded (offline first run, missing
package) search degrades to BM25 only and /api/health says why.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_lock = threading.Lock()
_models: dict[str, "Embedder"] = {}
_errors: dict[str, str] = {}


def cache_dir() -> str:
    path = Path(os.getenv("HANDBOOK_MODEL_CACHE", Path.home() / ".cache" / "handbook-qa" / "models"))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def parse_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        backend, name = spec.split(":", 1)
        return backend.strip().lower(), name.strip()
    return ("model2vec" if spec.startswith("minishlab/") else "fastembed"), spec


def _normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class Embedder:
    def __init__(self, spec: str):
        self.spec = spec
        self.backend, self.name = parse_spec(spec)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if self.backend == "model2vec":
            from model2vec import StaticModel
            self._model = StaticModel.from_pretrained(self.name)
        elif self.backend == "fastembed":
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.name, cache_dir=cache_dir())
        else:
            raise ValueError(f"Unknown embedding backend {self.backend!r} (use fastembed: or model2vec:)")
        self.dim = int(self.encode_docs(["dimension probe"]).shape[1])

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 1)), dtype=np.float32)
        if self.backend == "model2vec":
            return _normalize(self._model.encode(texts))
        return _normalize(np.array(list(self._model.embed(texts, batch_size=64))))

    def encode_query(self, text: str) -> np.ndarray:
        if self.backend == "model2vec":
            return _normalize(self._model.encode([text]))[0]
        # bge-style models embed queries with their retrieval instruction prefix
        return _normalize(np.array(list(self._model.query_embed([text]))))[0]


def get_model(spec: str) -> Embedder | None:
    with _lock:
        if spec not in _models and spec not in _errors:
            try:
                _models[spec] = Embedder(spec)
            except Exception as exc:  # noqa: BLE001 - any failure means "no dense search"
                _errors[spec] = f"{type(exc).__name__}: {exc}"
                log.warning("Embedding model %s unavailable, falling back to BM25 only: %s", spec, _errors[spec])
        return _models.get(spec)


def load_error(spec: str | None = None) -> str | None:
    if spec is not None:
        return _errors.get(spec)
    return next(iter(_errors.values()), None)
