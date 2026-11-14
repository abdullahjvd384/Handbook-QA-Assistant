"""Format-specific parsers.

Every loader returns a `Parsed` object: blocks in reading order plus any metadata the
document declares about itself (front matter, a "Document ID: ... | Status: ..." header
line, an HTML <title>). The pipeline merges that with the manifest.

The goal is to keep the structure the author intended — headings, list items and table
rows — because that structure decides both how we chunk and what we can highlight.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.models import BlockKind

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".csv", ".html", ".htm", ".pdf", ".docx"}


@dataclass
class RawBlock:
    kind: BlockKind
    text: str
    level: int = 0
    page: int | None = None


@dataclass
class Parsed:
    blocks: list[RawBlock] = field(default_factory=list)
    declared: dict[str, str] = field(default_factory=dict)   # metadata found inside the file
    title: str | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers

_HEADER_LINE = re.compile(r"^\s*Document ID:", re.I)


def parse_header_line(line: str) -> dict[str, str]:
    """Parse 'Document ID: x | Owner: y | Effective: 2025-04-01 | Status: current | Supersedes: z'."""
    out: dict[str, str] = {}
    for part in line.split("|"):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if not value:
            continue
        key = {"document_id": "doc_id", "effective": "effective_date"}.get(key, key)
        out[key] = value
    return out


def clean_inline_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)      # links -> link text
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)            # bold
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)  # italics
    text = text.replace("`", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _split_md_table_row(line: str) -> list[str]:
    return [clean_inline_markdown(c) for c in line.strip().strip("|").split("|")]


def table_row_text(headers: list[str], cells: list[str]) -> str:
    """Render a table row so it reads (and searches) on its own: 'Service: Payments | RTO: 4 hours'."""
    parts = []
    for i, cell in enumerate(cells):
        cell = cell.strip()
        if not cell:
            continue
        header = headers[i].strip() if i < len(headers) else ""
        parts.append(f"{header}: {cell}" if header else cell)
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Markdown

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def load_markdown(raw: str) -> Parsed:
    parsed = Parsed()
    m = _FRONTMATTER.match(raw)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                parsed.declared[k.strip()] = v.strip().strip('"').strip("'")
        raw = raw[m.end():]

    lines = raw.splitlines()
    para: list[str] = []
    table: list[str] = []

    def flush_para():
        if para:
            parsed.blocks.append(RawBlock("paragraph", clean_inline_markdown(" ".join(para))))
            para.clear()

    def flush_table():
        if not table:
            return
        rows = [r for r in table if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
        if rows:
            headers = _split_md_table_row(rows[0])
            parsed.blocks.append(RawBlock("table_header", " | ".join(headers)))
            for r in rows[1:]:
                parsed.blocks.append(RawBlock("table_row", table_row_text(headers, _split_md_table_row(r))))
        table.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_para()
            table.append(stripped)
            continue
        flush_table()
        if not stripped:
            flush_para()
            continue
        h = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if h:
            flush_para()
            level = len(h.group(1))
            text = clean_inline_markdown(h.group(2))
            if level == 1 and parsed.title is None:
                parsed.title = text
                parsed.blocks.append(RawBlock("title", text, level=1))
            else:
                parsed.blocks.append(RawBlock("heading", text, level=level))
            continue
        li = re.match(r"^(?:[-*+]|\d+[.)])\s+(.*)$", stripped)
        if li:
            flush_para()
            parsed.blocks.append(RawBlock("list_item", clean_inline_markdown(li.group(1))))
            continue
        para.append(stripped)
    flush_para()
    flush_table()
    return parsed


# ---------------------------------------------------------------------------
# Plain text (meeting notes, SEV definitions)

def load_text(raw: str) -> Parsed:
    parsed = Parsed()
    lines = [ln.rstrip() for ln in raw.splitlines()]
    first = True
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if first:
            parsed.title = stripped.title() if stripped.isupper() else stripped
            parsed.blocks.append(RawBlock("title", stripped, level=1))
            first = False
            continue
        bullet = re.match(r"^[-*•]\s+(.*)$", stripped)
        if bullet:
            parsed.blocks.append(RawBlock("list_item", bullet.group(1)))
        elif len(stripped) < 40 and not re.search(r"[.:;,]$", stripped) and len(stripped.split()) <= 4 \
                and not re.match(r"^SEV\d", stripped):
            parsed.blocks.append(RawBlock("heading", stripped, level=2))
        else:
            # Collapse runs of spaces used for column alignment ("SEV1  Customer-facing ...").
            parsed.blocks.append(RawBlock("paragraph", re.sub(r"^(\S+)\s{2,}", r"\1: ", stripped)))
    return parsed


# ---------------------------------------------------------------------------
# CSV

def load_csv(raw: str) -> Parsed:
    parsed = Parsed()
    reader = csv.reader(io.StringIO(raw))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        parsed.warnings.append("empty CSV")
        return parsed
    headers = [h.strip().replace("_", " ") for h in rows[0]]
    parsed.blocks.append(RawBlock("table_header", " | ".join(headers)))
    for r in rows[1:]:
        if len(r) != len(headers):
            parsed.warnings.append(f"CSV row has {len(r)} cells, header has {len(headers)}: {r}")
        cells = [c.replace(";", ",").strip() for c in r]
        parsed.blocks.append(RawBlock("table_row", table_row_text(headers, cells)))
    return parsed


# ---------------------------------------------------------------------------
# HTML

def load_html(raw: str) -> Parsed:
    from bs4 import BeautifulSoup

    parsed = Parsed()
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    if soup.title and soup.title.get_text(strip=True):
        parsed.title = soup.title.get_text(strip=True)
    body = soup.body or soup
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]):
        if el.name == "p" and el.find_parent("li"):
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if el.name[0] == "h":
            level = int(el.name[1])
            kind: BlockKind = "title" if level == 1 and not any(b.kind == "title" for b in parsed.blocks) else "heading"
            parsed.blocks.append(RawBlock(kind, text, level=level))
        elif el.name == "li":
            parsed.blocks.append(RawBlock("list_item", text))
        elif el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            is_header = bool(el.find("th")) and not el.find("td")
            parsed.blocks.append(RawBlock("table_header" if is_header else "table_row", " | ".join(cells)))
        else:
            parsed.blocks.append(RawBlock("paragraph", text))
    # Record links so the data-quality report can flag ones that point at nothing.
    parsed.declared["_links"] = "\n".join(a.get("href", "") for a in body.find_all("a") if a.get("href"))
    return parsed


# ---------------------------------------------------------------------------
# DOCX — walk the body in order so tables stay inside the section they belong to.

def load_docx(path: Path) -> Parsed:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parsed = Parsed()
    d = docx.Document(str(path))
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = re.sub(r"\s+", " ", p.text).strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            if _HEADER_LINE.match(text):
                parsed.declared.update(parse_header_line(text))
                continue
            if style == "Title":
                parsed.title = text
                parsed.blocks.append(RawBlock("title", text, level=1))
            elif style.startswith("Heading"):
                m = re.search(r"(\d+)", style)
                parsed.blocks.append(RawBlock("heading", text, level=(int(m.group(1)) + 1) if m else 2))
            elif "List" in style:
                parsed.blocks.append(RawBlock("list_item", text))
            else:
                parsed.blocks.append(RawBlock("paragraph", text))
        elif tag == "tbl":
            table = Table(child, d)
            rows = [[re.sub(r"\s+", " ", c.text).strip() for c in row.cells] for row in table.rows]
            if not rows:
                continue
            headers = rows[0]
            parsed.blocks.append(RawBlock("table_header", " | ".join(headers)))
            for r in rows[1:]:
                parsed.blocks.append(RawBlock("table_row", table_row_text(headers, r)))
    return parsed


# ---------------------------------------------------------------------------
# PDF — pypdf gives us lines; rebuild paragraphs, headings and numbered lists.

_PDF_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^.]{0,60})$")


def load_pdf(path: Path) -> Parsed:
    from pypdf import PdfReader

    parsed = Parsed()
    reader = PdfReader(str(path))
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lines = [ln.rstrip() for ln in text.splitlines()]
        if not any(ln.strip() for ln in lines):
            parsed.warnings.append(f"page {page_no} has no extractable text (scanned?)")
            continue
        width = max((len(ln) for ln in lines), default=80)
        i = 0
        para: list[str] = []
        pending_number: str | None = None

        def flush():
            nonlocal para
            if para:
                parsed.blocks.append(RawBlock("paragraph", " ".join(para), page=page_no))
                para = []

        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line:
                flush()
                continue
            if page_no == 1 and parsed.title is None:
                parsed.title = line
                parsed.blocks.append(RawBlock("title", line, level=1, page=page_no))
                continue
            if _HEADER_LINE.match(line):
                # The header can wrap ("... | Supersedes:" / "hr-leave-policy-v1").
                while i < len(lines) and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", lines[i].strip() or "#"):
                    line += " " + lines[i].strip()
                    i += 1
                parsed.declared.update(parse_header_line(line))
                continue
            if re.fullmatch(r"\d{1,2}", line):          # list number on its own line
                flush()
                pending_number = line
                continue
            if pending_number is not None:
                parsed.blocks.append(RawBlock("list_item", line, page=page_no))
                pending_number = None
                # Continuation lines of the list item.
                while i < len(lines) and lines[i].strip() and not re.fullmatch(r"\d{1,2}", lines[i].strip()) \
                        and not _PDF_HEADING.match(lines[i].strip()) and not parsed.blocks[-1].text.endswith("."):
                    parsed.blocks[-1].text += " " + lines[i].strip()
                    i += 1
                continue
            hm = _PDF_HEADING.match(line)
            if hm and len(line) < 60:
                flush()
                parsed.blocks.append(RawBlock("heading", line, level=2, page=page_no))
                continue
            para.append(line)
            # A short line ending a sentence closes the paragraph (pypdf drops blank lines).
            if re.search(r"[.!?:]$", line) and len(line) < 0.8 * width:
                flush()
        flush()
    return parsed


def load_file(path: Path) -> Parsed:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if ext in {".md", ".markdown"}:
        return load_markdown(raw)
    if ext == ".csv":
        return load_csv(raw)
    if ext in {".html", ".htm"}:
        return load_html(raw)
    return load_text(raw)
