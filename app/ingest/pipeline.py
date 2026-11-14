"""Turn files into Documents and Chunks, decide which versions are in force, and report
data-quality problems found along the way."""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from app.ingest.loaders import SUPPORTED_EXTENSIONS, Parsed, load_file
from app.models import Block, Chunk, DocMeta, Document, Lifecycle

log = logging.getLogger(__name__)

CHUNK_WORD_BUDGET = 120      # soft cap on words per chunk
LONG_BLOCK_WORDS = 160       # a single block longer than this is split by sentence


# ---------------------------------------------------------------------------
# Document assembly

def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def build_document(path: Path, manifest_entry: dict | None, source: str = "corpus",
                   overrides: dict | None = None) -> Document:
    """Parse one file and merge metadata: manifest < document's own header < explicit overrides."""
    parsed: Parsed = load_file(path)
    entry = dict(manifest_entry or {})
    declared = {k: v for k, v in parsed.declared.items() if not k.startswith("_")}
    warnings = list(parsed.warnings)

    # Cross-check what the manifest says against what the document says about itself.
    for key in ("doc_id", "status", "effective_date", "supersedes"):
        if key in declared and key in entry and str(entry[key]) != declared[key]:
            warnings.append(f"manifest says {key}={entry[key]!r} but the document says {declared[key]!r}")

    merged = {**declared, **{k: v for k, v in entry.items() if v not in (None, "")}}
    merged.update({k: v for k, v in (overrides or {}).items() if v not in (None, "")})
    doc_id = merged.get("doc_id") or slugify(path.stem)
    title = merged.get("title") or parsed.title or path.stem
    meta = DocMeta(
        doc_id=doc_id,
        title=title,
        path=path.name,
        format=path.suffix.lower().lstrip("."),
        category=merged.get("category", "Uncategorised"),
        owner=merged.get("owner", ""),
        effective_date=_parse_date(merged.get("effective_date")),
        status=(merged.get("status") or "current").lower(),
        supersedes=merged.get("supersedes"),
        source=source,
    )

    blocks: list[Block] = []
    parts: list[str] = []
    pos = 0
    for rb in parsed.blocks:
        text = rb.text.strip()
        if not text:
            continue
        if parts:
            parts.append("\n\n")
            pos += 2
        blocks.append(Block(kind=rb.kind, text=text, level=rb.level, page=rb.page, start=pos, end=pos + len(text)))
        parts.append(text)
        pos += len(text)
    if not blocks:
        warnings.append("no text could be extracted")
    meta.links = [ln for ln in parsed.declared.get("_links", "").splitlines() if ln]
    return Document(meta=meta, text="".join(parts), blocks=blocks, warnings=warnings)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "document"


# ---------------------------------------------------------------------------
# Lifecycle: which documents actually apply on a given date

def family_key(title: str) -> str:
    """'Leave Policy v2' / 'Production Deploy Runbook (2023)' -> the policy family they belong to."""
    t = title.lower()
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r"\bv\d+\b", " ", t)
    t = re.sub(r"\b(19|20)\d{2}\b", " ", t)
    t = re.sub(r"\b(revision|revised|draft|old|new)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def compute_lifecycles(metas: list[DocMeta], today: date) -> dict[str, Lifecycle]:
    by_id = {m.doc_id: m for m in metas}
    out: dict[str, Lifecycle] = {}
    for m in metas:
        if m.status in {"deprecated", "archived", "retired"}:
            out[m.doc_id] = Lifecycle("deprecated", note=f"Marked {m.status}; kept for history only.")
        elif m.effective_date and m.effective_date > today:
            out[m.doc_id] = Lifecycle("not_yet_effective", note=f"Takes effect on {m.effective_date.isoformat()}.")
        else:
            out[m.doc_id] = Lifecycle("in_force")
    # Explicit supersession only takes effect once the newer document itself is in force.
    for m in metas:
        old = m.supersedes
        if old and old in by_id and out[m.doc_id].state == "in_force" and out[old].state == "in_force":
            out[old] = Lifecycle("superseded", superseded_by=m.doc_id,
                                 note=f"Superseded by {m.title} from {m.effective_date}.")
    return out


# ---------------------------------------------------------------------------
# Chunking

_ABBREV = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bInc)(?<!\bNo)(?<!\bvs)(?<!\bU\.S)"
_SENT_END = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9\"'(“])")


def sentence_spans(text: str, base: int = 0) -> list[tuple[int, int]]:
    """Split text into sentence spans (absolute offsets)."""
    spans = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.start()
        if text[start:end].strip():
            spans.append((base + start, base + end))
        start = m.end()
    if text[start:].strip():
        spans.append((base + start, base + len(text.rstrip())))
    return spans


def chunk_document(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading_stack: dict[int, str] = {}
    table_header = ""
    group: list[Block] = []

    def section_path() -> str:
        return " > ".join(heading_stack[k] for k in sorted(heading_stack))

    def emit(start: int, end: int, extra: str = ""):
        text = doc.text[start:end]
        sec = section_path()
        index_text = " . ".join(p for p in (doc.meta.title, sec, extra, text) if p)
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}#{len(chunks)}", doc_id=doc.doc_id, section=sec,
                            text=text, start=start, end=end, index_text=index_text))

    def flush_group():
        if not group:
            return
        words = 0
        start = group[0].start
        last_end = group[0].start
        for b in group:
            n = len(b.text.split())
            if n > LONG_BLOCK_WORDS:
                if last_end > start:
                    emit(start, last_end)
                _emit_long_block(b)
                start = last_end = b.end
                words = 0
                continue
            if words and words + n > CHUNK_WORD_BUDGET:
                emit(start, last_end)
                start, words = b.start, 0
            words += n
            last_end = b.end
        if last_end > start:
            emit(start, last_end)
        group.clear()

    def _emit_long_block(b: Block):
        spans = sentence_spans(b.text, b.start)
        cur_start, words = spans[0][0], 0
        for i, (s, e) in enumerate(spans):
            n = len(doc.text[s:e].split())
            if words and words + n > CHUNK_WORD_BUDGET:
                emit(cur_start, spans[i - 1][1])
                cur_start, words = s, 0
            words += n
        emit(cur_start, spans[-1][1])

    lead_in = ""
    for b in doc.blocks:
        if b.kind == "title":
            continue
        if b.kind == "heading":
            flush_group()
            table_header = lead_in = ""
            for lvl in [k for k in heading_stack if k >= b.level]:
                del heading_stack[lvl]
            heading_stack[b.level] = b.text
            continue
        if b.kind == "table_header":
            flush_group()
            table_header = b.text
            continue
        if b.kind == "table_row":
            # Each row is a self-contained fact; index it with its header for context.
            flush_group()
            emit(b.start, b.end, extra=f"Columns: {table_header}" if table_header else "")
            continue
        if b.kind == "list_item":
            # Bullets in a handbook are usually one fact each (a glossary term, a rule, a
            # severity level). Index each on its own, with the sentence that introduces the list.
            flush_group()
            emit(b.start, b.end, extra=" ".join(lead_in.split()[:30]))
            continue
        lead_in = b.text
        group.append(b)
    flush_group()

    if not chunks and doc.blocks:   # title-only document: still make it findable
        b = doc.blocks[0]
        emit(b.start, b.end)
    return chunks


# ---------------------------------------------------------------------------
# Corpus loading + data-quality report

_REF = re.compile(r"\b((?:[A-Z][A-Za-z0-9-]*\s){1,5})(Policy|Runbook|Guide|Procedure|Calendar|Table|Guidelines|Checklist|Matrix|Definitions)\b")
_GENERIC = {"the", "a", "an", "this", "see", "under", "process", "our", "and", "of"}


def _reference_resolves(ref: str, titles: list[str]) -> bool:
    words = [w for w in re.findall(r"[a-z0-9-]+", ref.lower()) if w not in _GENERIC]
    for t in titles:
        tw = set(re.findall(r"[a-z0-9-]+", t.lower()))
        if set(words) <= tw:
            return True
        # "BYOD Policy" -> "Bring Your Own Device Policy"
        initials = "".join(w[0] for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _GENERIC)
        if len(words) >= 2 and len(words[0]) >= 2 and initials.startswith(words[0]) and words[-1] in tw:
            return True
    return False


def load_manifest(corpus_dir: Path) -> list[dict]:
    path = corpus_dir / "manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("documents", [])


def load_corpus(corpus_dir: Path) -> tuple[list[Document], list[str]]:
    """Load every document listed in the manifest plus any unlisted supported file."""
    report: list[str] = []
    manifest = load_manifest(corpus_dir)
    listed = {e["path"]: e for e in manifest}
    docs: list[Document] = []
    for entry in manifest:
        path = corpus_dir / entry["path"]
        if not path.exists():
            report.append(f"[missing file] manifest lists {entry['path']} but it is not in the corpus")
            continue
        docs.append(build_document(path, entry))
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.name not in listed and not path.name.startswith("."):
            report.append(f"[unlisted file] {path.name} is not in the manifest; ingested with inferred metadata")
            docs.append(build_document(path, None))
    return docs, report


def find_missing_references(docs: list[Document]) -> dict[str, list[str]]:
    """Documents the handbook mentions by name that are not in the corpus:
    {'Anti-Harassment Policy': ['hr-grievance-policy'], ...}"""
    titles = [d.meta.title for d in docs]
    missing: dict[str, list[str]] = {}
    for d in docs:
        seen = set()
        for m in _REF.finditer(d.text):
            ref = (m.group(1) + m.group(2)).strip()
            ref = re.sub(r"^(The|See|Under|This|Our)\s+", "", ref)
            if ref.lower() in seen or ref == d.meta.title or ref.split()[0] in {"This", "These"}:
                continue
            seen.add(ref.lower())
            if not _reference_resolves(ref, titles):
                missing.setdefault(ref, []).append(d.doc_id)
    return missing


def quality_report(docs: list[Document], lifecycles: dict[str, Lifecycle], extra: list[str]) -> list[str]:
    report = list(extra)
    files = {d.meta.path for d in docs}
    by_id = {d.doc_id: d for d in docs}
    for d in docs:
        lc = lifecycles[d.doc_id]
        for w in d.warnings:
            report.append(f"[{d.doc_id}] {w}")
        if lc.state == "superseded" and d.meta.status == "current":
            report.append(f"[{d.doc_id}] manifest marks it 'current' but {lc.superseded_by} supersedes it; treated as superseded")
        if lc.state == "not_yet_effective":
            report.append(f"[{d.doc_id}] effective {d.meta.effective_date} is in the future; answers use the version in force today and mention the change")
        if lc.state == "deprecated":
            report.append(f"[{d.doc_id}] deprecated; down-ranked and flagged if cited")
        for link in d.meta.links:
            if link and not link.startswith(("http", "mailto:", "#")) and link not in files:
                report.append(f"[{d.doc_id}] broken link to {link!r} (no such file in the corpus)")
    for ref, doc_ids in find_missing_references(docs).items():
        for doc_id in doc_ids:
            report.append(f"[{doc_id}] refers to '{ref}', which is not in the corpus")
    for d in docs:
        if d.meta.supersedes and d.meta.supersedes not in by_id:
            report.append(f"[{d.doc_id}] supersedes unknown document {d.meta.supersedes!r}")
    return report
