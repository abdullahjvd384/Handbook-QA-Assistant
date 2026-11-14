"""Deterministic answerer used when no LLM is configured (and as the LLM's safety net).

It never writes new prose about the policy: it picks the sentences / table rows that best
cover the question and quotes them verbatim, then adds the lifecycle notes the passage
statuses call for (an upcoming change, an outdated version). It abstains when no sentence
covers enough of the question, or when the question's key term appears nowhere in the
handbook.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingest.pipeline import family_key, sentence_spans
from app.qa.context import Passage
from app.search.bm25 import BM25
from app.search.text import (
    ANSWER_TYPE_STEMS, NUMBER_WORDS, capitalized_terms, content_terms, core_question, expand_query, expansions, tokenize,
)

ABSTAIN_COVERAGE = 0.45
CONTEXT_CREDIT = 0.75          # a term found only in the document title / section heading
LOW_WEIGHT = 0.3               # answer-type words and numbers in the question
HEADING_BONUS = 0.08           # unit sits under a section heading that matches the question ("2. Annual leave")
UNKNOWN_TERM_ABSTAIN = 0.35    # share of the question's weight carried by words absent from the corpus
_QUANTITY_Q = re.compile(r"^\s*(how (many|much|long|often|fast|quickly|soon|far)|when|what time|what is the (max|min|limit|deadline))", re.I)
_QUANTITY_A = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(minute|hour|day|week|month|year|working|business|percent)", re.I)
_WHO_Q = re.compile(r"^\s*who\b", re.I)
_PERSON = re.compile(r"\b(?!(?:The|This|That|Our|An?|Each|Every)\b)[A-Z][a-z]+ "
                     r"(?!(?:Team|Policy|Office|Lead|Manager|Operations|Procedure|Runbook|Guide|Table|Calendar)\b)[A-Z][a-z]+\b")
_DEFINE_Q = re.compile(r"what (?:does|do|is|are) (?:an? |the )?([\w-]+) (?:mean|stand for)|define (?:an? |the )?([\w-]+)"
                       r"|meaning of (?:an? |the )?([\w-]+)|what (?:is|are) (?:an? )?((?-i:[A-Z]{2,6}))\b", re.I)
_TERM_LINE = re.compile(r"^[\w\s/().-]{1,40}:\s")   # "RTO: Recovery Time Objective. ..." / "Tier 1: ..."
_DEPENDENT = re.compile(r"^\s*(and|what about|how about|what if|also|same|so|then|ok)\b|\b(it|that|this|those|these|they|them|there|its|their)\b", re.I)
_ANAPHORIC = re.compile(r"^(this|that|these|those|it|they|after that|before that|until then|in that case|otherwise)\b", re.I)


@dataclass
class Unit:
    passage: Passage
    start: int
    end: int
    text: str
    coverage: float = 0.0
    score: float = 0.0
    terms: frozenset = frozenset()


@dataclass
class ExtractiveResult:
    answerable: bool
    picks: list[tuple[Unit, str]]     # (unit, role)
    reason: str = ""


def followup_query(question: str, history: list[dict]) -> tuple[str, list[str]]:
    """Heuristic standalone query for a follow-up: prepend the previous user question when the
    new one leans on it. Returns (retrieval query, terms to weight up)."""
    prev = [h["content"] for h in history if h.get("role") == "user" and h.get("content")]
    if prev and (_DEPENDENT.search(question) or len(content_terms(question)) <= 3):
        return f"{prev[-1]} {question}", content_terms(question)
    return question, []


# ---------------------------------------------------------------------------
# quotable units

def _units(p: Passage, doc) -> list[Unit]:
    """Sentences, whole table rows, whole short bullets and definitions. A sentence that
    leans on the previous one ('After that window, ...') is merged into it, and a lead-in
    ending in ':' absorbs the rows/items it introduces, so every quote stands on its own."""
    text = doc.text
    kinds = {b.start: b.kind for b in doc.blocks}
    blocks: list[tuple[int, int]] = []
    pos = p.start
    for block in text[p.start:p.end].split("\n\n"):
        if block.strip():
            blocks.append((pos, pos + len(block)))
        pos += len(block) + 2

    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(blocks):
        s, e = blocks[i]
        block = text[s:e]
        whole = (kinds.get(s) in {"table_row", "table_header"} or (" | " in block and ":" in block)
                 or (kinds.get(s) == "list_item" and len(block.split()) <= 45)
                 or (_TERM_LINE.match(block) and len(block.split()) <= 45))
        if whole:
            spans.append((s, e))
        else:
            for ss, se in sentence_spans(block, s):
                if spans and _ANAPHORIC.match(text[ss:se]) and spans[-1][1] >= s - 2:
                    spans[-1] = (spans[-1][0], se)
                else:
                    spans.append((ss, se))
        if block.rstrip().endswith(":"):
            j = i + 1
            while j < len(blocks) and j - i <= 6 and (" | " in text[blocks[j][0]:blocks[j][1]]
                                                      or len(text[blocks[j][0]:blocks[j][1]].split()) < 25):
                j += 1
            if j > i + 1:
                spans[-1] = (spans[-1][0], blocks[j - 1][1])
                i = j
                continue
        i += 1
    return [Unit(p, s, e, text[s:e]) for s, e in spans]


# ---------------------------------------------------------------------------
# scoring

def _term_weights(question: str, boost_terms: list[str], bm25: BM25) -> tuple[dict, dict]:
    """(ranking weights over all query terms, coverage weights over the user's own core terms)."""
    def w(t: str) -> float:
        base = bm25.term_idf(t)
        return base * LOW_WEIGHT if (t in ANSWER_TYPE_STEMS or t.isdigit() or t in NUMBER_WORDS) else base

    rank = {t: w(t) for t in content_terms(question)}
    for t in boost_terms:
        rank[t] = rank.get(t, w(t)) * 2.0
    for t in content_terms(expand_query(question)):
        rank.setdefault(t, 0.5 * w(t))           # thesaurus words help ranking, at half weight
    # Coverage: the actual question sentence counts fully, surrounding sentences
    # ("I'm on an active SEV1.", or an injected instruction) at half weight.
    # Words from those surrounding sentences that exist nowhere in the handbook can never be
    # supported, so they are left out of coverage instead of counting against it.
    core = set(content_terms(core_question(question))) | set(boost_terms)
    cov = {t: rank[t] * (1.0 if t in core else 0.5) for t in content_terms(question) + list(boost_terms)
           if t in core or t in bm25.idf}
    return rank, cov


def answer(question: str, passages: list[Passage], docs: dict, bm25: BM25,
           boost_terms: list[str] | None = None, explicit_docs: set[str] | None = None) -> ExtractiveResult:
    """explicit_docs: non-current documents the question asks about by name (e.g. 'the 2023 runbook')."""
    boost_terms = boost_terms or []
    explicit_docs = explicit_docs or set()
    rank_w, cov_w = _term_weights(question, boost_terms, bm25)
    rank_total = sum(rank_w.values()) or 1.0
    cov_total = sum(cov_w.values()) or 1.0

    # A word the thesaurus maps to handbook vocabulary ("vacation" -> "annual leave") is covered
    # when any of the mapped words appears.
    synonyms: dict[str, set[str]] = {}
    for typed, mapped in expansions(question):
        for t in content_terms(typed):
            synonyms.setdefault(t, set()).update(content_terms(mapped))

    # The question names something (a capitalised term: 'CEO', 'London', 'Wi-Fi') that appears
    # nowhere in the handbook -> the handbook doesn't cover it.
    named = capitalized_terms(question)
    unknown = [t for t in cov_w if t in named and t not in bm25.idf and not t.isdigit() and t not in synonyms]
    if cov_w and sum(cov_w[t] for t in unknown) / cov_total >= UNKNOWN_TERM_ABSTAIN:
        return ExtractiveResult(False, [], reason=f"terms not in the handbook: {unknown}")

    top_score = max((p.score for p in passages), default=1.0) or 1.0
    dm = _DEFINE_Q.search(question)
    define_term = next((g for g in dm.groups() if g), None) if dm else None
    wants_quantity = bool(_QUANTITY_Q.search(core_question(question)))
    wants_person = bool(_WHO_Q.search(core_question(question)))

    units: list[Unit] = []
    for p in passages:
        doc = docs[p.doc_id]
        context = set(tokenize(f"{p.title} {p.section}"))
        # the section heading the unit sits under (documents without sections get no bonus:
        # their title is already credited through `context`)
        heading = [t for t in tokenize(p.section.split(">")[-1]) if not t.isdigit() and t not in ANSWER_TYPE_STEMS]
        heading_match = sum(t in rank_w for t in heading) / len(heading) if heading else 0.0
        for u in _units(p, doc):
            toks = set(tokenize(u.text))
            hit = {t for t in rank_w if t in toks}
            ctx = {t for t in rank_w if t in context and t not in toks}
            for t, alts in synonyms.items():
                # the mapped phrase must really be there: "resign" -> "resignation notice" is not
                # satisfied by a sentence that only says "notice"
                need = len(alts) if len(alts) <= 2 else -(-len(alts) // 2)
                if t not in hit and len(alts & toks) >= need:
                    hit.add(t)
                elif t not in hit and t not in ctx and len(alts & (toks | context)) >= need:
                    ctx.add(t)
            u.terms = frozenset(hit | ctx)
            u.coverage = (sum(cov_w.get(t, 0) for t in hit) + CONTEXT_CREDIT * sum(cov_w.get(t, 0) for t in ctx)) / cov_total
            ranking_cov = (sum(rank_w[t] for t in hit) + CONTEXT_CREDIT * sum(rank_w[t] for t in ctx)) / rank_total
            u.score = 0.65 * ranking_cov + 0.35 * (p.score / top_score)
            u.score += HEADING_BONUS * heading_match
            if wants_quantity and _QUANTITY_A.search(u.text):
                u.score += 0.1
            if wants_person and _PERSON.search(u.text):
                u.score += 0.1
            if define_term and re.match(rf"\s*{re.escape(define_term)}\b\s*(:|is\b|means\b|\()", u.text, re.I):
                u.score += 0.25
            if p.lifecycle.state != "in_force" and p.doc_id not in explicit_docs:
                u.score *= 0.6
            units.append(u)

    eligible = [u for u in units if u.passage.lifecycle.state == "in_force" or u.passage.doc_id in explicit_docs] or units
    if not eligible:
        return ExtractiveResult(False, [], reason="no passages")
    eligible.sort(key=lambda u: -u.score)
    best = eligible[0]
    if best.coverage < ABSTAIN_COVERAGE:
        return ExtractiveResult(False, [], reason=f"best coverage {best.coverage:.2f}")

    best = _extend_with_next_sentence(best, eligible, cov_w)
    picks: list[tuple[Unit, str]] = [(best, "primary")]
    covered = set(best.terms)
    for u in eligible[1:]:
        if len(picks) >= 3:
            break
        if u.passage.doc_id == best.passage.doc_id and best.start <= u.start < best.end:
            continue
        new = {t for t in u.terms if t not in covered}
        gain = sum(rank_w[t] for t in new) / rank_total
        if gain >= 0.15 and u.coverage >= 0.3 and u.score >= 0.5 * best.score:
            picks.append((u, "supporting"))
            covered |= new

    # Lifecycle notes: other versions of the cited documents that were retrieved.
    cited = {u.passage.doc_id for u, _ in picks}
    families = {family_key(u.passage.title) for u, _ in picks}
    primary_current = picks[0][0].passage.lifecycle.state == "in_force"
    for state, role in (("not_yet_effective", "upcoming"), ("superseded", "outdated"), ("deprecated", "outdated")):
        if role == "outdated" and not primary_current:
            continue
        cands = [u for u in units if u.passage.lifecycle.state == state and u.passage.doc_id not in cited
                 and family_key(u.passage.title) in families and u.coverage >= 0.3]
        if cands:
            u = max(cands, key=lambda x: x.score)
            picks.append((u, role))
            cited.add(u.passage.doc_id)
    return ExtractiveResult(True, picks)


def _extend_with_next_sentence(best: Unit, units: list[Unit], cov_w: dict) -> Unit:
    """Two-part answers often sit in consecutive sentences ('Up to 8 days may be carried...
    Carried days must be used by March 31'). Append the next sentence of the same paragraph
    when it is about the same thing."""
    nxt = [u for u in units if u.passage is best.passage and 0 <= u.start - best.end <= 1]
    if not nxt:
        return best
    n = nxt[0]
    shared = {t for t in n.terms if t in cov_w} & set(best.terms)
    if shared and len(best.text.split()) + len(n.text.split()) <= 60:
        p = best.passage
        text = p.text[best.start - p.start: n.end - p.start]
        return Unit(p, best.start, n.end, text, best.coverage, best.score, best.terms | n.terms)
    return best


def compose(picks: list[tuple[Unit, str]]) -> str:
    """Render picks as text with [n] markers matching citation order."""
    lines = []
    for n, (u, role) in enumerate(picks, start=1):
        p = u.passage
        where = f"{p.title}" + (f" ({p.section})" if p.section else "")
        quote = u.text.strip()
        if role == "primary":
            lines.append(f"According to the {where}: “{quote}” [{n}]")
        elif role == "supporting":
            lines.append(f"Also relevant, from the {where}: “{quote}” [{n}]")
        elif role == "upcoming":
            lines.append(f"Upcoming change: the {p.title} takes effect on {p.effective_date} and says: “{quote}” [{n}] "
                         f"Until then the rule above applies.")
        else:
            label = "is deprecated" if p.lifecycle.state == "deprecated" else "has been superseded"
            lines.append(f"Note: the {p.title} {label} and no longer applies; it said: “{quote}” [{n}]")
    if picks and picks[0][0].passage.lifecycle.state != "in_force":
        p = picks[0][0].passage
        lines.insert(0, f"Caution: the best match is from the {p.title}, which is {p.status_label().lower()}.")
    return "\n\n".join(lines)
