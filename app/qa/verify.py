"""Locate a quoted passage inside a document's text.

A citation is only kept if its quote can be found in the source. Matching tolerates
whitespace, curly quotes, dashes, case and markdown emphasis; anything looser goes
through a fuzzy sentence match whose result is replaced by the *real* source text, so
what the user sees highlighted is always verbatim.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.ingest.pipeline import sentence_spans

_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                            "–": "-", "—": "-", " ": " ", "*": " ", "`": " "})


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase + collapse whitespace, remembering where each output char came from."""
    out: list[str] = []
    index: list[int] = []
    prev_space = True
    for i, ch in enumerate(text.translate(_TRANSLATE)):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                index.append(i)
            prev_space = True
            continue
        out.append(ch.lower())
        index.append(i)
        prev_space = False
    if out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def _clean_quote(quote: str) -> str:
    q = quote.strip().strip('"\'“”‘’').strip()
    q = re.sub(r"^(\.\.\.|…)\s*|\s*(\.\.\.|…)$", "", q)
    return q


def locate_quote(quote: str, text: str, lo: int = 0, hi: int | None = None) -> tuple[int, int] | None:
    """Return (start, end) of `quote` within text[lo:hi] (absolute offsets), or None."""
    hi = len(text) if hi is None else hi
    window = text[lo:hi]
    q = _clean_quote(quote)
    if len(q) < 3:
        return None
    norm_src, idx = _normalize_with_map(window)
    # Exact (normalised) match, allowing an ellipsis to join two in-order fragments.
    parts = [p for p in re.split(r"\s*(?:\.\.\.|…)\s*", q) if p.strip()]
    pos, first, last = 0, None, None
    for part in parts:
        norm_q, _ = _normalize_with_map(part)
        norm_q = norm_q.rstrip(" .")
        if not norm_q:
            continue
        found = norm_src.find(norm_q, pos)
        if found < 0:
            first = None
            break
        first = found if first is None else first
        last = found + len(norm_q)
        pos = last
    if first is not None and last is not None:
        start = lo + idx[first]
        end = lo + idx[last - 1] + 1
        # Extend to include a trailing period if the source has one.
        if end < len(text) and text[end] == "." and not quote.rstrip().endswith("."):
            end += 1
        return start, end

    # Fuzzy: best single sentence / line in the window.
    norm_q, _ = _normalize_with_map(q)
    best, best_span = 0.0, None
    for s, e in _units(window):
        cand, _ = _normalize_with_map(window[s:e])
        ratio = SequenceMatcher(None, norm_q, cand, autojunk=False).ratio()
        if ratio > best:
            best, best_span = ratio, (lo + s, lo + e)
    return best_span if best >= 0.85 else None


def _units(window: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for block in window.split("\n\n"):
        spans.extend(sentence_spans(block, pos))
        pos += len(block) + 2
    return spans
