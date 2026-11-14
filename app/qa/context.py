"""Turn search hits into the passages an answerer sees."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.models import Lifecycle
from app.search.index import KnowledgeBase, ScoredChunk

SECTION_EXPAND_WORDS = 200


@dataclass
class Passage:
    pid: str
    doc_id: str
    title: str
    section: str
    text: str
    start: int
    end: int
    lifecycle: Lifecycle
    effective_date: str | None
    score: float

    def status_label(self) -> str:
        lc = self.lifecycle
        eff = f"effective {self.effective_date}" if self.effective_date else "no effective date"
        if lc.state == "in_force":
            return f"IN FORCE ({eff})"
        if lc.state == "superseded":
            return f"SUPERSEDED by {lc.superseded_by} - no longer applies ({eff})"
        if lc.state == "deprecated":
            return f"DEPRECATED - kept for history, must not be followed ({eff})"
        return f"NOT YET IN EFFECT - takes effect {self.effective_date}"


def with_current_versions(kb: KnowledgeBase, query: str, hits: list[ScoredChunk], as_of: date) -> list[ScoredChunk]:
    """Make sure the answerer sees every version of a retrieved policy that matters today:
    - a superseded/deprecated document was retrieved but its in-force sibling wasn't -> add it;
    - an in-force document has a scheduled revision (not yet effective) -> add the revision,
      so the answer can mention the upcoming change."""
    lcs = kb.lifecycles(as_of)
    present = {h.chunk.doc_id for h in hits}
    out: list[ScoredChunk] = []
    for i, h in enumerate(hits):
        out.append(h)
        if i >= 5:
            continue
        if h.lifecycle in {"superseded", "deprecated"}:
            current = lcs[h.chunk.doc_id].superseded_by
            siblings = [current] if current else [
                d for d in kb.family_members(h.chunk.doc_id) if lcs[d].state == "in_force"]
        elif h.lifecycle == "in_force":
            siblings = [d for d in kb.family_members(h.chunk.doc_id) if lcs[d].state == "not_yet_effective"]
        else:
            continue
        for sib in siblings:
            if sib and sib not in present:
                best = kb.best_chunk_in_doc(query, sib)
                if best is not None:
                    # Right after the hit that triggered it, so the passage limit can't cut it off.
                    out.append(ScoredChunk(best, h.score * 0.99, lifecycle=lcs[sib].state))
                    present.add(sib)
    return out


def fuse_rankings(rankings: list[list[ScoredChunk]], k: int) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion of several result lists (e.g. the user's wording and the
    LLM's handbook-vocabulary restatement)."""
    scores: dict[str, float] = {}
    first: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
            first.setdefault(cid, h)
    order = sorted(scores, key=lambda c: -scores[c])[: 2 * k]
    return [ScoredChunk(first[c].chunk, scores[c], first[c].bm25, first[c].dense, first[c].lifecycle) for c in order]


WHOLE_DOC_WORDS = 700


def build_passages(kb: KnowledgeBase, hits: list[ScoredChunk], as_of: date, limit: int = 8,
                   whole_short_docs: bool = False) -> list[Passage]:
    """Expand each hit to its whole section when the section is short, and dedupe.
    whole_short_docs: when a short document supplies two or more hits, pass the whole
    document instead of fragments (used for the LLM, which can read and combine it)."""
    snap = kb.snapshot
    lcs = kb.lifecycles(as_of)
    per_doc: dict[str, int] = {}
    for h in hits:
        per_doc[h.chunk.doc_id] = per_doc.get(h.chunk.doc_id, 0) + 1
    by_section: dict[tuple[str, str], list] = {}
    for c in snap.chunks:
        by_section.setdefault((c.doc_id, c.section), []).append(c)

    passages: list[Passage] = []
    covered: list[tuple[str, int, int]] = []
    for h in hits:
        c = h.chunk
        doc = snap.docs[c.doc_id]
        sec_chunks = by_section[(c.doc_id, c.section)]
        start, end = c.start, c.end
        section = c.section
        if whole_short_docs and per_doc[c.doc_id] >= 2 and len(doc.text.split()) <= WHOLE_DOC_WORDS:
            start, end, section = 0, len(doc.text), "whole document"
        elif c.section:
            s, e = min(x.start for x in sec_chunks), max(x.end for x in sec_chunks)
            if len(doc.text[s:e].split()) <= SECTION_EXPAND_WORDS:
                start, end = s, e
        if any(d == c.doc_id and s <= start and end <= e for d, s, e in covered):
            continue
        covered.append((c.doc_id, start, end))
        passages.append(Passage(
            pid=f"P{len(passages) + 1}", doc_id=c.doc_id, title=doc.meta.title, section=section,
            text=doc.text[start:end], start=start, end=end, lifecycle=lcs[c.doc_id],
            effective_date=doc.meta.effective_date.isoformat() if doc.meta.effective_date else None,
            score=h.score,
        ))
        if len(passages) >= limit:
            break
    return passages
