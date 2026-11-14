"""Cross-encoder re-ranking of the top search candidates.

A cross-encoder reads the question and a passage *together* and scores how well the
passage answers it: slower than comparing two separately computed vectors, but more
precise. It re-orders the top RERANK_CANDIDATES fused results; everything below keeps its
fused order. Whether it is on by default was decided by measurement (DECISIONS.md).
"""
from __future__ import annotations

import logging
import math
import threading

from app.search.dense import cache_dir

log = logging.getLogger(__name__)

RERANK_CANDIDATES = 20

_lock = threading.Lock()
_models: dict[str, "CrossEncoderReranker"] = {}
_errors: dict[str, str] = {}


class CrossEncoderReranker:
    def __init__(self, name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.name = name
        self._model = TextCrossEncoder(name, cache_dir=cache_dir())

    def scores(self, query: str, passages: list[str]) -> list[float]:
        """Relevance in (0, 1) for each passage."""
        if not passages:
            return []
        return [1.0 / (1.0 + math.exp(-float(s))) for s in self._model.rerank(query, passages, batch_size=32)]


def get_reranker(name: str) -> CrossEncoderReranker | None:
    with _lock:
        if name not in _models and name not in _errors:
            try:
                _models[name] = CrossEncoderReranker(name)
            except Exception as exc:  # noqa: BLE001
                _errors[name] = f"{type(exc).__name__}: {exc}"
                log.warning("Re-ranker %s unavailable, continuing without it: %s", name, _errors[name])
        return _models.get(name)


def load_error(name: str) -> str | None:
    return _errors.get(name)
