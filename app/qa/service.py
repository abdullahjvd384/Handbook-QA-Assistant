"""Question answering: follow-up resolution -> retrieval -> answer -> verified citations."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict
from datetime import date

from app.config import Settings
from app.ingest.pipeline import sentence_spans
from app.models import Citation
from app.qa import extractive
from app.qa.context import Passage, build_passages, fuse_rankings, with_current_versions
from app.qa.llm import LLMAnswerer
from app.qa.verify import locate_quote
from app.search.index import KnowledgeBase, SearchConfig

log = logging.getLogger(__name__)

NOT_FOUND = ("I couldn't find an answer to that in the handbook. It may not be covered, "
             "or it may be worded differently: try rephrasing, or ask People Operations.")
LLM_PASSAGES = 10

# Greetings and chit-chat get a short reply without searching (and without LLM calls).
_SMALL_TALK = re.compile(
    r"(hi|hii+|hello|hey|hey there|yo|good (morning|afternoon|evening)|greetings|"
    r"thanks?|thank you( very much| so much)?|thx|ty|cheers|great|ok|okay|cool|nice|"
    r"bye|goodbye|see you|"
    r"help|who are you|what are you|what can you do|what do you do|how does this work)"
    r"( there| bot| assistant)?", re.I)
_THANKS = re.compile(r"(thanks?|thank you|thx|ty|cheers|great|ok|okay|cool|nice|bye|goodbye|see you)\b", re.I)
HELLO = ("Hi! I answer questions about the Northgate Robotics handbook (HR, IT, engineering, finance and travel "
         "policies), and every answer quotes the passage it comes from. Try: \"How many days of annual leave do I get?\"")
YOU_ARE_WELCOME = "You're welcome. Ask me anything else about the handbook."
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class QAService:
    def __init__(self, kb: KnowledgeBase, settings: Settings, answerer: LLMAnswerer | None = None):
        self.kb = kb
        self.settings = settings
        self.llm = answerer or LLMAnswerer(settings)

    def llm_available(self) -> bool:
        return self.llm.available()

    def resolve_mode(self, requested: str | None) -> str:
        mode = (requested or self.settings.answer_mode or "auto").lower()
        if mode == "auto":
            return "llm" if self.llm_available() else "extractive"
        if mode not in {"llm", "extractive"}:
            raise ValueError("mode must be 'auto', 'llm' or 'extractive'")
        if mode == "llm" and not self.llm_available():
            raise ValueError("LLM mode requested but OPENAI_API_KEY is not configured")
        return mode

    # ------------------------------------------------------------------

    def ask(self, question: str, history: list[dict] | None = None, as_of: date | None = None,
            mode: str | None = None, k: int = 8, search_config: SearchConfig = SearchConfig()) -> dict:
        t0 = time.perf_counter()
        question = " ".join(question.split())
        if not question:
            raise ValueError("question must not be empty")
        history = history or []
        as_of = as_of or self.settings.today()
        mode_used = self.resolve_mode(mode)
        if _SMALL_TALK.fullmatch(question.strip(" !.?,")):
            reply = YOU_ARE_WELCOME if _THANKS.match(question) else HELLO
            return self._response(question, question, None, {"answer": reply, "answerable": False, "citations": []},
                                  "small-talk", as_of, [], 0, t0, [], [])
        calls_before = self.llm.calls
        warnings: list[str] = []

        # 1. Make the question searchable: resolve follow-ups and, with an LLM, restate it in
        #    handbook vocabulary ("complain about my manager" -> "grievance policy").
        boost: list[str] = []
        # Typos ("vacaton") are fixed for search and the extractive answerer; the LLM planner
        # reads the original wording and does its own correction.
        corrected, spelling = self.kb.correct_spelling(question)
        standalone, search_query = corrected, None
        if mode_used == "llm":
            try:
                titles = sorted(d.meta.title for d in self.kb.snapshot.docs.values())
                standalone, search_query = self.llm.plan_search(question, history, titles)
            except Exception as exc:  # noqa: BLE001 - any API failure degrades to the heuristic
                log.warning("search planning failed: %s", exc)
                warnings.append(f"search planning failed ({_short(exc)}); used keyword heuristic")
                if history:
                    standalone, boost = extractive.followup_query(corrected, history)
        elif history:
            standalone, boost = extractive.followup_query(corrected, history)

        # 2. Retrieve (both phrasings when we have two, fused by rank).
        hits = self.kb.search(standalone, as_of, k=k, config=search_config)
        if search_query and search_query.lower() != standalone.lower():
            hits = fuse_rankings([hits, self.kb.search(search_query, as_of, k=k, config=search_config)], k)
        if mode_used == "llm":
            # An LLM reads whole short documents better than fragments of them (a city's tier
            # and that tier's hotel cap sit in different sections of the Travel Policy).
            passages = build_passages(self.kb, with_current_versions(self.kb, standalone, hits, as_of), as_of,
                                      limit=LLM_PASSAGES, whole_short_docs=True)
        else:
            passages = build_passages(self.kb, with_current_versions(self.kb, standalone, hits, as_of), as_of, limit=k)

        # Documents the handbook mentions but doesn't contain (e.g. the Anti-Harassment Policy).
        snap = self.kb.snapshot
        seen_text = (standalone + " " + " ".join(p.text for p in passages)).lower()
        missing = [ref for ref in snap.missing_refs if ref.lower() in seen_text]
        asked_missing = [ref for ref in missing if ref.lower() in standalone.lower()]

        # 3. Answer.
        result = None
        if mode_used == "llm" and passages:
            try:
                result = self._answer_llm(question if not history else standalone, history, passages, as_of, missing)
            except Exception as exc:  # noqa: BLE001 - any API failure falls back to extractive
                log.warning("LLM answer failed: %s", exc)
                warnings.append(f"LLM answer failed ({_short(exc)}); fell back to extractive answer")
                mode_used = "extractive (fallback)"
            if result is not None and result["answerable"] and not result["citations"]:
                warnings.append("LLM citations could not be verified against the sources; fell back to extractive answer")
                mode_used, result = "extractive (fallback)", None
        if result is None and asked_missing:
            result = self._answer_missing_doc(asked_missing[0], as_of)
        if result is None:
            result = self._answer_extractive(standalone, passages, boost)

        return self._response(question, standalone, search_query, result, mode_used, as_of, hits,
                              self.llm.calls - calls_before, t0, warnings, spelling)

    def _response(self, question, standalone, search_query, result, mode_used, as_of, hits, calls, t0,
                  warnings, spelling) -> dict:
        snap = self.kb.snapshot
        return {
            "question": question,
            "standalone_question": standalone,
            "search_query": search_query,
            "answer": result["answer"],
            "answerable": result["answerable"],
            "citations": result["citations"],
            "mode": mode_used,
            "model": self.llm.model if mode_used == "llm" else None,
            "as_of": as_of.isoformat(),
            "spelling_corrections": [{"from": a, "to": b} for a, b in spelling],
            "retrieved": [
                {"doc_id": h.chunk.doc_id, "title": snap.docs[h.chunk.doc_id].meta.title, "chunk_id": h.chunk.chunk_id,
                 "section": h.chunk.section, "score": round(h.score, 5), "lifecycle": h.lifecycle}
                for h in hits
            ],
            "llm_calls": calls,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------

    def _citation(self, n: int, p: Passage, start: int, end: int, role: str) -> dict:
        doc = self.kb.snapshot.docs[p.doc_id]
        # The section the quote actually sits in (a passage may span the whole document).
        section = next((c.section for c in self.kb.snapshot.chunks
                        if c.doc_id == p.doc_id and c.start <= start < c.end), p.section)
        c = Citation(n=n, doc_id=p.doc_id, title=p.title, section=section, quote=doc.text[start:end],
                     start=start, end=end, lifecycle=p.lifecycle.state, effective_date=p.effective_date, role=role)
        out = asdict(c)
        out["url"] = f"/?doc={p.doc_id}&start={start}&end={end}"
        return out

    def _answer_extractive(self, question: str, passages: list[Passage], boost: list[str]) -> dict:
        snap = self.kb.snapshot
        lcs = {p.doc_id: p.lifecycle.state for p in passages}
        explicit = {d for d, s in lcs.items() if s != "in_force" and self.kb.explicitly_requested(question, d, s)}
        res = extractive.answer(question, passages, snap.docs, snap.bm25, boost, explicit)
        if not res.answerable:
            return {"answer": NOT_FOUND, "answerable": False, "citations": []}
        citations = [self._citation(n, u.passage, u.start, u.end, role)
                     for n, (u, role) in enumerate(res.picks, start=1)]
        return {"answer": extractive.compose(res.picks), "answerable": True, "citations": citations}

    def _answer_missing_doc(self, ref: str, as_of: date) -> dict:
        """The user asks about a document the handbook names but does not contain."""
        snap = self.kb.snapshot
        citations = []
        for doc_id in snap.missing_refs.get(ref, [])[:2]:
            doc = snap.docs[doc_id]
            i = doc.text.find(ref)
            if i < 0:
                continue
            # Cite the sentence that mentions the missing document.
            span = next(((s, e) for s, e in sentence_spans(doc.text) if s <= i < e), (i, i + len(ref)))
            lc = self.kb.lifecycles(as_of)[doc_id]
            p = Passage("P0", doc_id, doc.meta.title, "", doc.text, 0, len(doc.text), lc,
                        doc.meta.effective_date.isoformat() if doc.meta.effective_date else None, 0.0)
            citations.append(self._citation(len(citations) + 1, p, span[0], span[1], "supporting"))
        refs = "".join(f" [{c['n']}]" for c in citations)
        answer = (f"The handbook refers to the {ref}{refs}, but that document is not part of the handbook, "
                  f"so I can't tell you what it says. People Operations should be able to share it.")
        return {"answer": answer, "answerable": False, "citations": citations}

    def _answer_llm(self, question: str, history: list[dict], passages: list[Passage], as_of: date,
                    missing: list[str] | None = None) -> dict:
        out = self.llm.answer(question, history, passages, as_of, missing)
        by_pid = {p.pid: p for p in passages}
        docs = self.kb.snapshot.docs
        renumber: dict[int, int] = {}
        citations: list[dict] = []
        seen: dict[tuple, int] = {}
        for i, c in enumerate(out.citations, start=1):
            p = by_pid.get(str(c.get("passage_id", "")).strip())
            quote = str(c.get("quote", ""))
            if p is None or not quote.strip():
                continue
            span = locate_quote(quote, docs[p.doc_id].text, p.start, p.end) or locate_quote(quote, docs[p.doc_id].text)
            if span is None:
                log.info("Dropped unverifiable quote from %s: %r", p.doc_id, quote[:80])
                continue
            key = (p.doc_id, span)
            if key in seen:                       # same quote cited twice -> one citation
                renumber[i] = seen[key]
                continue
            role = {"not_yet_effective": "upcoming", "superseded": "outdated", "deprecated": "outdated"}.get(
                p.lifecycle.state, "primary" if not citations else "supporting")
            n = len(citations) + 1
            citations.append(self._citation(n, p, span[0], span[1], role))
            renumber[i] = seen[key] = n

        def fix(m: re.Match) -> str:
            nums = [renumber.get(int(x)) for x in m.group(1).split(",")]
            nums = sorted({x for x in nums if x})
            return "".join(f"[{x}]" for x in nums)

        answer = _MARKER.sub(fix, out.answer).strip()
        answer = re.sub(r"[ \t]+([.,;:])", r"\1", answer)
        if not out.answerable and not answer:
            answer = NOT_FOUND
        return {"answer": answer, "answerable": out.answerable, "citations": citations}


def _short(exc: Exception) -> str:
    text = str(exc) or type(exc).__name__
    return text if len(text) <= 160 else text[:157] + "..."
