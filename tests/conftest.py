from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from app.config import ROOT, Settings  # noqa: E402
from app.qa.service import QAService  # noqa: E402
from app.search.index import KnowledgeBase  # noqa: E402

TODAY = date(2026, 11, 14)


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    # Dense search off by default so the suite runs offline and fast; test_search has a
    # separate dense test that skips when the model can't be loaded.
    return Settings(corpus_dir=ROOT / "data" / "corpus",
                    uploads_dir=Path(tmp_path_factory.mktemp("uploads")),
                    today_override=TODAY, use_dense=False, answer_mode="extractive",
                    vector_store="memory")


@pytest.fixture(scope="session")
def kb(settings) -> KnowledgeBase:
    return KnowledgeBase(settings).load()


@pytest.fixture(scope="session")
def qa(kb, settings) -> QAService:
    return QAService(kb, settings)
