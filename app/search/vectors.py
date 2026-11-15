"""Where chunk embeddings live: Qdrant (default when it is running) or in memory.

Both implement the same two operations:
    sync(chunks)                 make the stored vectors match the current chunks
    search(qv, limit, doc_id)    nearest chunks to a query vector -> [(chunk_id, score)]

Qdrant keeps vectors across restarts. Each point stores a hash of the embedding model
plus the chunk's indexed text, so `sync` embeds only chunks that are new or changed and
deletes points whose chunk no longer exists (an edited or removed document). An upload
therefore embeds only the new document's chunks.
"""
from __future__ import annotations

import hashlib
import logging
import uuid

import numpy as np

from app.models import Chunk

log = logging.getLogger(__name__)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"handbook-chunk:{chunk_id}"))


def _content_hash(model_name: str, chunk: Chunk) -> str:
    return hashlib.sha1(f"{model_name}\n{chunk.index_text}".encode("utf-8")).hexdigest()


class MemoryVectorIndex:
    """Exact search over a NumPy matrix; rebuilt on every sync."""

    name = "memory"

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self._ids: list[str] = []
        self._doc_ids: list[str] = []
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        self.stats: dict = {}

    def sync(self, chunks: list[Chunk]) -> None:
        matrix = self.model.encode_docs([c.index_text for c in chunks])
        self._ids, self._doc_ids, self._matrix = [c.chunk_id for c in chunks], [c.doc_id for c in chunks], matrix
        self.stats = {"points": len(chunks), "embedded": len(chunks), "reused": 0, "deleted": 0}

    def search(self, qv: np.ndarray, limit: int, doc_id: str | None = None) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        sims = self._matrix @ qv
        order = np.argsort(-sims)
        out = []
        for i in order:
            if doc_id is None or self._doc_ids[i] == doc_id:
                out.append((self._ids[i], float(sims[i])))
                if len(out) >= limit:
                    break
        return out


class QdrantVectorIndex:
    """Persistent vectors in a Qdrant collection (see docker-compose.yml)."""

    name = "qdrant"

    def __init__(self, model, model_name: str, url: str, collection: str):
        from qdrant_client import QdrantClient

        self.model = model
        self.model_name = model_name
        self.collection = collection
        self.url = url
        self.client = QdrantClient(url=url, timeout=10, check_compatibility=False)  # reachability is checked below
        self.client.get_collections()                       # fail fast if the server isn't reachable
        self.dim = model.dim
        self._ensure_collection()
        self.stats: dict = {}

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            size = info.config.params.vectors.size
            if size == self.dim:
                return
            log.warning("Qdrant collection %s has dimension %s, model needs %s: recreating", self.collection, size, self.dim)
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE))
        self.client.create_payload_index(self.collection, field_name="doc_id",
                                         field_schema=models.PayloadSchemaType.KEYWORD)

    def _stored(self) -> dict[str, str]:
        """point id -> content hash for everything currently in the collection."""
        stored: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, limit=1000, offset=offset,
                                                with_payload=["hash"], with_vectors=False)
            for p in points:
                stored[str(p.id)] = (p.payload or {}).get("hash", "")
            if offset is None:
                return stored

    def sync(self, chunks: list[Chunk]) -> None:
        from qdrant_client import models

        stored = self._stored()
        wanted = {_point_id(c.chunk_id): (c, _content_hash(self.model_name, c)) for c in chunks}
        changed = [(pid, c, h) for pid, (c, h) in wanted.items() if stored.get(pid) != h]
        for i in range(0, len(changed), 256):
            batch = changed[i:i + 256]
            vecs = self.model.encode_docs([c.index_text for _, c, _ in batch])
            self.client.upsert(self.collection, wait=True, points=[
                models.PointStruct(id=pid, vector=v.tolist(), payload={
                    "chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section,
                    "text": c.text, "hash": h, "model": self.model_name})
                for (pid, c, h), v in zip(batch, vecs)])
        stale = [pid for pid in stored if pid not in wanted]
        if stale:
            self.client.delete(self.collection, wait=True, points_selector=models.PointIdsList(points=stale))
        self.stats = {"points": len(wanted), "embedded": len(changed), "reused": len(wanted) - len(changed),
                      "deleted": len(stale)}
        log.info("Qdrant sync: %s", self.stats)

    def search(self, qv: np.ndarray, limit: int, doc_id: str | None = None) -> list[tuple[str, float]]:
        from qdrant_client import models

        flt = None
        if doc_id is not None:
            flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        res = self.client.query_points(self.collection, query=qv.tolist(), limit=limit, query_filter=flt,
                                       with_payload=["chunk_id"])
        return [((p.payload or {}).get("chunk_id", ""), float(p.score)) for p in res.points]


def make_vector_index(model, model_name: str, store: str, url: str, collection: str):
    """store: 'qdrant' (required), 'memory', or 'auto' (Qdrant if reachable, else memory)."""
    if store in {"qdrant", "auto"}:
        try:
            return QdrantVectorIndex(model, model_name, url, collection)
        except Exception as exc:  # noqa: BLE001 - connection refused, timeout, missing package...
            if store == "qdrant":
                raise RuntimeError(f"Qdrant is required (HANDBOOK_VECTOR_STORE=qdrant) but unavailable at {url}: {exc}") from exc
            log.warning("Qdrant not reachable at %s (%s); using in-memory vectors. Start it with: docker compose up -d",
                        url, type(exc).__name__)
    return MemoryVectorIndex(model, model_name)
