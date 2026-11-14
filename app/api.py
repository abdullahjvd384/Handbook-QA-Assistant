"""HTTP API and static chat page.

    POST /api/ask                    ask a question (the endpoint external evaluators call)
    GET  /api/documents              list documents with lifecycle status
    GET  /api/documents/{doc_id}     fetch one document (text + structure for highlighting)
    GET  /api/documents/{doc_id}/file  download the original file
    POST /api/documents              upload a new document (multipart); searchable immediately
    GET  /api/report                 data-quality findings from ingestion
    GET  /api/health                 status, index size, which answer mode is active
"""
from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings, settings as default_settings
from app.ingest.loaders import SUPPORTED_EXTENSIONS
from app.ingest.pipeline import slugify
from app.qa.service import QAService
from app.search import dense
from app.search.index import KnowledgeBase

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="The employee's question.")
    history: list[Turn] = Field(default_factory=list, max_length=40,
                                description="Earlier turns, oldest first, for follow-up questions.")
    as_of: date | None = Field(None, description="Answer as of this date (YYYY-MM-DD). Defaults to today.")
    mode: Literal["auto", "llm", "extractive"] | None = Field(None, description="Answer generator; default auto.")
    top_k: int = Field(8, ge=1, le=20, description="Number of passages to retrieve.")


def create_app(settings: Settings = default_settings, kb: KnowledgeBase | None = None,
               qa: QAService | None = None) -> FastAPI:
    state: dict = {}
    upload_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        state["kb"] = kb or KnowledgeBase(settings).load()
        state["qa"] = qa or QAService(state["kb"], settings)
        yield

    app = FastAPI(title="Northgate Handbook Q&A", version="1.0.0", lifespan=lifespan)

    def _kb() -> KnowledgeBase:
        return state["kb"]

    def _qa() -> QAService:
        return state["qa"]

    # -- API -------------------------------------------------------------------

    @app.get("/api/health")
    def health():
        k = _kb()
        snap = k.snapshot
        return {
            "status": "ok",
            "documents": len(snap.docs),
            "chunks": len(snap.chunks),
            "dense_search": k.dense_enabled,
            "dense_model": settings.dense_model if k.dense_enabled else None,
            "dense_error": dense.load_error(settings.dense_model),
            "reranker": settings.rerank_model if settings.use_reranker else None,
            "vector_store": k.vector_store_name,
            "vector_store_stats": k.vector_store_stats,
            "answer_mode": _qa().resolve_mode(None),
            "llm_provider": _qa().llm.provider if _qa().llm_available() else None,
            "llm_model": _qa().llm.model if _qa().llm_available() else None,
            "today": settings.today().isoformat(),
        }

    @app.post("/api/ask")
    def ask(req: AskRequest):
        try:
            return _qa().ask(req.question, [t.model_dump() for t in req.history], req.as_of, req.mode, req.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/documents")
    def list_documents(as_of: date | None = None):
        k = _kb()
        lcs = k.lifecycles(as_of or settings.today())
        return [_meta_json(d, lcs[d.doc_id]) for d in sorted(k.snapshot.docs.values(), key=lambda d: d.doc_id)]

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str, as_of: date | None = None):
        k = _kb()
        doc = k.snapshot.docs.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"No document with id {doc_id!r}")
        lc = k.lifecycles(as_of or settings.today())[doc_id]
        return {
            **_meta_json(doc, lc),
            "text": doc.text,
            "blocks": [{"kind": b.kind, "level": b.level, "page": b.page, "start": b.start, "end": b.end}
                       for b in doc.blocks],
        }

    @app.get("/api/documents/{doc_id}/file")
    def get_document_file(doc_id: str):
        doc = _kb().snapshot.docs.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"No document with id {doc_id!r}")
        base = settings.uploads_dir if doc.meta.source == "upload" else settings.corpus_dir
        path = (base / doc.meta.path).resolve()
        if base.resolve() not in path.parents or not path.exists():
            raise HTTPException(status_code=404, detail="Original file not available")
        return FileResponse(path, filename=doc.meta.path)

    @app.post("/api/documents", status_code=201)
    async def upload_document(
        file: UploadFile = File(...),
        title: str | None = Form(None, max_length=200),
        category: str | None = Form(None, max_length=60),
        effective_date: date | None = Form(None),
        status: Literal["current", "deprecated"] = Form("current"),
        supersedes: str | None = Form(None, max_length=120),
    ):
        name = Path(file.filename or "upload").name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File is larger than 10 MB")
        if not data.strip():
            raise HTTPException(status_code=400, detail="File is empty")
        k = _kb()
        if supersedes and supersedes not in k.snapshot.docs:
            raise HTTPException(status_code=400, detail=f"supersedes refers to unknown document {supersedes!r}")

        with upload_lock:
            base_id = slugify(title or Path(name).stem)[:60]
            doc_id, n = base_id, 2
            while doc_id in k.snapshot.docs:
                doc_id, n = f"{base_id}-{n}", n + 1
            settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            path = settings.uploads_dir / f"{doc_id}{ext}"
            path.write_bytes(data)
            entry = {"doc_id": doc_id, "path": path.name, "title": title, "category": category or "Uploaded",
                     "owner": "Uploaded via web app",
                     "effective_date": (effective_date or settings.today()).isoformat(),
                     "status": status, "supersedes": supersedes}
            try:
                doc = k.add_document(path, entry)
            except Exception as exc:  # noqa: BLE001 - surface parse failures to the user
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=422, detail=f"Could not ingest this file: {exc}") from exc
            manifest_path = settings.uploads_dir / "manifest.json"
            entries = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
            entries.append({k_: v for k_, v in entry.items() if v is not None})
            tmp = manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            tmp.replace(manifest_path)

        chunks = sum(1 for c in k.snapshot.chunks if c.doc_id == doc.doc_id)
        lc = k.lifecycles(settings.today())[doc.doc_id]
        return {**_meta_json(doc, lc), "chunks": chunks, "warnings": doc.warnings}

    @app.get("/api/report")
    def report(as_of: date | None = None):
        return {"as_of": (as_of or settings.today()).isoformat(), "findings": _kb().report(as_of or settings.today())}

    # -- UI --------------------------------------------------------------------

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _meta_json(doc, lc) -> dict:
    m = doc.meta
    return {
        "doc_id": m.doc_id, "title": m.title, "category": m.category, "owner": m.owner,
        "format": m.format, "path": m.path, "source": m.source,
        "effective_date": m.effective_date.isoformat() if m.effective_date else None,
        "declared_status": m.status, "supersedes": m.supersedes,
        "lifecycle": lc.state, "superseded_by": lc.superseded_by, "lifecycle_note": lc.note,
    }


app = create_app()
