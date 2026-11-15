"""The knowledge base: documents, chunks and the search indexes over them.

Keyword index (BM25) lives in memory; chunk embeddings live in Qdrant when it is running
(docker compose up -d), otherwise in memory (see app/search/vectors.py).

Search = BM25 + dense embeddings, fused with Reciprocal Rank Fusion, then a
lifecycle-aware re-rank that pushes superseded / deprecated / not-yet-effective
documents below the version in force (unless the question asks for them).

The index is an immutable snapshot. Uploads build a new snapshot and swap it in under a
lock, so requests in flight never see a half-built index and nothing needs a restart.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import Settings
from app.ingest.pipeline import (
    build_document, chunk_document, compute_lifecycles, family_key, find_missing_references, load_corpus,
    quality_report,
)
from app.models import Chunk, Document, Lifecycle
from app.search import dense
from app.search.bm25 import BM25
from app.search.text import expand_query, strip_question_frame, thesaurus_words
from app.search.rerank import RERANK_CANDIDATES, get_reranker
from app.search.spelling import SpellingCorrector
from app.search.vectors import make_vector_index

log = logging.getLogger(__name__)

RRF_K = 60
LIFECYCLE_PENALTY = {"in_force": 1.0, "not_yet_effective": 0.85, "superseded": 0.6, "deprecated": 0.5}
_HISTORY_WORDS = re.compile(r"\b(old|older|previous|prior|former|deprecated|superseded|used to|history|v1)\b", re.I)
_FUTURE_WORDS = re.compile(r"\b(upcoming|future|next year|revision|from 20\d\d|after 20\d\d|in 20\d\d|20\d\d)\b", re.I)


@dataclass(frozen=True)
class SearchConfig:
    use_bm25: bool = True
    use_dense: bool = True
    lifecycle_rerank: bool = True
    expand_query: bool = True
    dense_weight: float = 2.0      # RRF weight of dense vs BM25, tuned for bge-small on dev + paraphrase (DECISIONS.md)
    rerank: bool | None = None     # cross-encoder re-rank; None = follow settings (HANDBOOK_RERANK)
    rerank_model: str | None = None

    @property
    def name(self) -> str:
        parts = [p for p, on in (("bm25", self.use_bm25), ("dense", self.use_dense)) if on]
        if self.expand_query and self.use_bm25:
            parts[0] += "(+thesaurus)"
        return "+".join(parts) + (" +lifecycle-rerank" if self.lifecycle_rerank else "")


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    bm25: float = 0.0
    dense: float = 0.0
    lifecycle: str = "in_force"


@dataclass
class Snapshot:
    docs: dict[str, Document]
    chunks: list[Chunk]
    bm25: BM25
    chunk_index: dict[str, int] = field(default_factory=dict)          # chunk_id -> position in `chunks`
    load_report: list[str] = field(default_factory=list)
    missing_refs: dict[str, list[str]] = field(default_factory=dict)   # referenced but absent documents
    speller: SpellingCorrector | None = None


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._snap: Snapshot | None = None
        self._dense_model = None
        self._vindex = None            # QdrantVectorIndex or MemoryVectorIndex (app/search/vectors.py)

    # -- building ------------------------------------------------------------

    def load(self) -> "KnowledgeBase":
        docs, report = load_corpus(self.settings.corpus_dir)
        docs += self._load_uploads(report)
        if self.settings.use_dense:
            self._dense_model = dense.get_model(self.settings.dense_model)
            if self._dense_model is not None:
                s = self.settings
                self._vindex = make_vector_index(self._dense_model, s.dense_model, s.vector_store,
                                                 s.qdrant_url, s.qdrant_collection)
        with self._lock:
            self._snap = self._build(docs, report)
        log.info("Indexed %d documents / %d chunks (vectors: %s)", len(docs), len(self._snap.chunks),
                 self.vector_store_name)
        return self

    def _build(self, docs: list[Document], report: list[str]) -> Snapshot:
        chunks = [c for d in docs for c in chunk_document(d)]
        texts = [c.index_text for c in chunks]
        if self._vindex is not None:
            self._vindex.sync(chunks)       # Qdrant: embeds only new/changed chunks, deletes stale ones
        self._lifecycles.cache_clear()
        return Snapshot(docs={d.doc_id: d for d in docs}, chunks=chunks, bm25=BM25(texts),
                        chunk_index={c.chunk_id: i for i, c in enumerate(chunks)},
                        load_report=report, missing_refs=find_missing_references(docs),
                        speller=SpellingCorrector([d.text for d in docs], tuple(d.meta.title for d in docs),
                                                  thesaurus_words()))

    def _load_uploads(self, report: list[str]) -> list[Document]:
        manifest = self.settings.uploads_dir / "manifest.json"
        if not manifest.exists():
            return []
        out = []
        for entry in json.loads(manifest.read_text(encoding="utf-8")):
            path = self.settings.uploads_dir / entry["path"]
            if path.exists():
                out.append(build_document(path, entry, source="upload"))
            else:
                report.append(f"[missing upload] {entry['path']}")
        return out

    def add_document(self, path: Path, meta: dict) -> Document:
        """Run a new file through the full ingestion pipeline and make it searchable now."""
        doc = build_document(path, meta, source="upload")
        if not doc.blocks:
            raise ValueError("No text could be extracted from this file.")
        with self._lock:
            snap = self.snapshot
            docs = [d for d in snap.docs.values() if d.doc_id != doc.doc_id] + [doc]
            self._snap = self._build(docs, snap.load_report)
        return doc

    # -- reading -------------------------------------------------------------

    @property
    def snapshot(self) -> Snapshot:
        if self._snap is None:
            raise RuntimeError("KnowledgeBase.load() has not been called")
        return self._snap

    @property
    def dense_enabled(self) -> bool:
        return self._vindex is not None

    @property
    def vector_store_name(self) -> str:
        return self._vindex.name if self._vindex is not None else "none"

    @property
    def vector_store_stats(self) -> dict:
        return dict(self._vindex.stats) if self._vindex is not None else {}

    def lifecycles(self, as_of: date) -> dict[str, Lifecycle]:
        return self._lifecycles(as_of, id(self.snapshot))

    @lru_cache(maxsize=32)  # noqa: B019 - keyed on snapshot id, cleared on rebuild
    def _lifecycles(self, as_of: date, _snap_id: int) -> dict[str, Lifecycle]:
        return compute_lifecycles([d.meta for d in self.snapshot.docs.values()], as_of)

    def report(self, as_of: date) -> list[str]:
        snap = self.snapshot
        return quality_report(list(snap.docs.values()), self.lifecycles(as_of), snap.load_report)

    def correct_spelling(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        speller = self.snapshot.speller
        return speller.correct(text) if speller is not None else (text, [])

    def family_members(self, doc_id: str) -> list[str]:
        snap = self.snapshot
        key = family_key(snap.docs[doc_id].meta.title)
        return [d for d, doc in snap.docs.items() if family_key(doc.meta.title) == key]

    # -- search --------------------------------------------------------------

    def search(self, query: str, as_of: date, k: int = 8, config: SearchConfig = SearchConfig()) -> list[ScoredChunk]:
        snap = self.snapshot
        n = len(snap.chunks)
        if n == 0 or not query.strip():
            return []
        fused = np.zeros(n)
        bm_query = strip_question_frame(expand_query(query) if config.expand_query else query)
        bm = snap.bm25.scores(bm_query) if config.use_bm25 else np.zeros(n)
        if config.use_bm25:
            order = [i for i in np.argsort(-bm) if bm[i] > 0][:50]
            for rank, i in enumerate(order):
                fused[i] += 1.0 / (RRF_K + rank + 1)
        sims = np.zeros(n)
        if config.use_dense and self._vindex is not None:
            qv = self._dense_model.encode_query(query)
            rank = 0
            for chunk_id, score in self._vindex.search(qv, limit=50):
                i = snap.chunk_index.get(chunk_id)
                if i is None:            # vector store briefly ahead of this snapshot during an upload
                    continue
                sims[i] = score
                fused[i] += config.dense_weight / (RRF_K + rank + 1)
                rank += 1

        lcs = self.lifecycles(as_of)
        results = []
        for i in np.argsort(-fused)[: max(k * 4, 30)]:
            if fused[i] <= 0:
                break
            c = snap.chunks[i]
            state = lcs.get(c.doc_id, Lifecycle("in_force")).state
            score = fused[i]
            if config.lifecycle_rerank and not self.explicitly_requested(query, c.doc_id, state):
                score *= LIFECYCLE_PENALTY[state]
            results.append(ScoredChunk(c, float(score), float(bm[i]), float(sims[i]), state))
        results.sort(key=lambda r: -r.score)
        use_rerank = self.settings.use_reranker if config.rerank is None else config.rerank
        if use_rerank:
            results = self._rerank(query, results, config)
        return results[:k]

    def _rerank(self, query: str, results: list[ScoredChunk], config: SearchConfig) -> list[ScoredChunk]:
        """Re-order the top candidates by cross-encoder relevance, keeping the lifecycle
        penalty so a stale version still sits below the current one."""
        model = get_reranker(config.rerank_model or self.settings.rerank_model)
        if model is None:
            return results
        head, tail = results[:RERANK_CANDIDATES], results[RERANK_CANDIDATES:]
        relevance = model.scores(query, [h.chunk.index_text for h in head])
        for h, rel in zip(head, relevance):
            penalty = 1.0
            if config.lifecycle_rerank and not self.explicitly_requested(query, h.chunk.doc_id, h.lifecycle):
                penalty = LIFECYCLE_PENALTY[h.lifecycle]
            h.score = 1.0 + rel * penalty           # > any fused score, so the head stays on top
        head.sort(key=lambda r: -r.score)
        return head + tail

    def best_chunk_in_doc(self, query: str, doc_id: str) -> Chunk | None:
        snap = self.snapshot
        idx = [i for i, c in enumerate(snap.chunks) if c.doc_id == doc_id]
        if not idx:
            return None
        bm = snap.bm25.scores(strip_question_frame(expand_query(query)))[idx]
        if self._vindex is not None:
            qv = self._dense_model.encode_query(query)
            by_id = dict(self._vindex.search(qv, limit=len(idx), doc_id=doc_id))
            dn = np.array([by_id.get(snap.chunks[i].chunk_id, 0.0) for i in idx])
            bm = bm / (bm.max() or 1.0) + dn
        return snap.chunks[idx[int(np.argmax(bm))]]

    def explicitly_requested(self, query: str, doc_id: str, state: str) -> bool:
        """Don't bury an old/future version when the user is asking about it specifically."""
        if state == "in_force":
            return True
        title = self.snapshot.docs[doc_id].meta.title
        markers = set(re.findall(r"\b(?:19|20)\d{2}\b|\bv\d+\b", f"{title} {doc_id}".lower()))
        if markers & set(re.findall(r"\b(?:19|20)\d{2}\b|\bv\d+\b", query.lower())):
            return True
        if state in {"deprecated", "superseded"}:
            return bool(_HISTORY_WORDS.search(query))
        return bool(_FUTURE_WORDS.search(query))
