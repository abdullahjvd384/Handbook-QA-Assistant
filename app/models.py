"""Core data structures shared by ingestion, search and answering."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

BlockKind = Literal["title", "heading", "paragraph", "list_item", "table_row", "table_header"]


@dataclass
class Block:
    """One structural unit of a document (a heading, paragraph, list item or table row).

    `start`/`end` are character offsets into `Document.text`, so any quote taken from a
    block can be highlighted exactly in the document viewer.
    """

    kind: BlockKind
    text: str
    level: int = 0            # heading depth (1 = top level); 0 for non-headings
    page: int | None = None   # 1-based page for PDFs
    start: int = 0
    end: int = 0


@dataclass
class DocMeta:
    doc_id: str
    title: str
    path: str
    format: str
    category: str = "Uncategorised"
    owner: str = ""
    effective_date: date | None = None
    status: str = "current"            # as declared by the manifest / document header
    supersedes: str | None = None
    source: str = "corpus"             # "corpus" or "upload"
    links: list[str] = field(default_factory=list)   # outgoing links (HTML), for link checking


@dataclass
class Lifecycle:
    """Whether a document's content applies on a given date. Computed, not declared."""

    state: Literal["in_force", "superseded", "deprecated", "not_yet_effective"]
    superseded_by: str | None = None
    note: str = ""

    @property
    def is_current(self) -> bool:
        return self.state == "in_force"


@dataclass
class Document:
    meta: DocMeta
    text: str                     # canonical display text (what the viewer shows)
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.meta.doc_id


@dataclass
class Chunk:
    """The retrieval unit: a section (or part of one) of a document."""

    chunk_id: str
    doc_id: str
    section: str          # heading path, e.g. "4. Limits > 4.2 Hotels and daily spending limits"
    text: str             # exact slice of Document.text
    start: int
    end: int
    index_text: str       # text used for search: title + section + content


@dataclass
class Citation:
    n: int
    doc_id: str
    title: str
    section: str
    quote: str
    start: int
    end: int
    lifecycle: str
    effective_date: str | None
    role: Literal["primary", "supporting", "upcoming", "outdated"] = "primary"
