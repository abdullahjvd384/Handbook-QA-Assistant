"""Runtime configuration, read once from environment variables.

Everything has a sensible default so `uvicorn app.api:app` works with no setup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env support (KEY=VALUE lines). Real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_date(name: str) -> date | None:
    raw = os.getenv(name)
    return date.fromisoformat(raw) if raw else None


DEFAULT_DENSE_MODEL = "fastembed:BAAI/bge-small-en-v1.5"
DEFAULT_RERANK = False
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = field(default_factory=lambda: Path(os.getenv("HANDBOOK_CORPUS_DIR", ROOT / "data" / "corpus")))
    uploads_dir: Path = field(default_factory=lambda: Path(os.getenv("HANDBOOK_UPLOADS_DIR", ROOT / "data" / "uploads")))
    # "Today" for deciding which policy version is in force. Defaults to the real date;
    # the eval pins it so results are reproducible after 2027-01-01.
    today_override: date | None = field(default_factory=lambda: _env_date("HANDBOOK_TODAY"))
    use_dense: bool = field(default_factory=lambda: _env_bool("HANDBOOK_DENSE", True))
    # "<backend>:<model>", see app/search/dense.py. Chosen by the ablation in DECISIONS.md.
    dense_model: str = field(default_factory=lambda: os.getenv("HANDBOOK_DENSE_MODEL", DEFAULT_DENSE_MODEL))
    use_reranker: bool = field(default_factory=lambda: _env_bool("HANDBOOK_RERANK", DEFAULT_RERANK))
    rerank_model: str = field(default_factory=lambda: os.getenv("HANDBOOK_RERANK_MODEL", DEFAULT_RERANK_MODEL))
    # Where embeddings live: "auto" = Qdrant if reachable, else in memory; or force "qdrant" / "memory".
    vector_store: str = field(default_factory=lambda: os.getenv("HANDBOOK_VECTOR_STORE", "auto").lower())
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "handbook_chunks"))
    # Answer generation: "auto" uses Claude when credentials exist, otherwise extractive.
    answer_mode: str = field(default_factory=lambda: os.getenv("HANDBOOK_ANSWER_MODE", "auto"))
    # "auto" picks whichever key is set (Anthropic first); or force "anthropic" / "openai".
    llm_provider: str = field(default_factory=lambda: os.getenv("HANDBOOK_LLM_PROVIDER", "auto").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("HANDBOOK_LLM_MODEL", "claude-opus-5-5"))
    openai_model: str = field(default_factory=lambda: os.getenv("HANDBOOK_OPENAI_MODEL", "gpt-5.5"))
    llm_effort: str = field(default_factory=lambda: os.getenv("HANDBOOK_LLM_EFFORT", "medium"))
    max_upload_bytes: int = 10 * 1024 * 1024

    def today(self) -> date:
        return self.today_override or date.today()


settings = Settings()
