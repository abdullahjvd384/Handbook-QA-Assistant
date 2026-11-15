"""LLM-as-judge: faithfulness, answer relevance and semantic correctness.

The regex grader in run_eval.py checks that the key fact is present. It cannot tell whether
*every* sentence of an answer is backed by the cited text, and it gives false negatives when
an answer phrases a correct fact differently. The judge covers both, RAGAS-style:

* faithfulness - the answer is split into atomic factual claims; each is checked against the
                 cited passages only (with some surrounding context). Score = supported / factual
                 claims. Statements about what the handbook does *not* cover are "meta" and not scored.
* relevance    - does the answer address the question that was asked? (full / partial / none)
* correctness  - compared with the evaluator's reference answer (correct / partial / incorrect).

The judge sees the reference answer and the evidence, never the assistant's retrieval list or
its own earlier verdicts. Using the same model family as the answerer risks self-preference, so
run_eval reports how often the judge agrees with the independent regex grader and lists every
disagreement for a human to read.
"""
from __future__ import annotations

from dataclasses import replace

JUDGE_PROMPT = """You are a strict evaluator of a company-handbook question-answering assistant.

You receive: the question, the date it is asked, the evaluator's reference answer, the assistant's answer, and the evidence: the handbook passages the assistant cited, each with some surrounding text and its status on that date. Judge only from the evidence and the reference. Do not use outside knowledge.

1. claims: split the assistant's answer into atomic claims, one fact each.
   - kind "factual": a statement about policy, numbers, dates, people or procedures. supported = true only if the evidence states it or it follows directly from cited facts (simple arithmetic, combining two cited facts, applying an effective date to the question date). If the evidence is empty, factual claims are unsupported.
   - kind "meta": statements about what the handbook does or does not contain, greetings, or advice to contact a team. Mark supported = true; they are not scored.
2. relevance: "full" if the answer addresses what was asked, "partial" if it addresses only part or mostly something adjacent, "none" if it does not address the question.
3. correctness against the reference answer:
   - "correct": gives the reference's key facts (wording may differ; extra correct detail is fine) and does not contradict it. If the reference says the handbook does not cover the question, a clear "not covered / cannot find it" is correct.
   - "partial": some key facts are missing, or the right answer is buried under hedging or a wrong-version value.
   - "incorrect": misses or contradicts the key facts, answers from a superseded or not-yet-effective version as if it applied, invents an answer the reference says is not in the handbook, or declines when the reference has an answer.
4. reason: one short sentence explaining the correctness verdict."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "kind": {"type": "string", "enum": ["factual", "meta"]},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "kind", "supported"],
                "additionalProperties": False,
            },
        },
        "relevance": {"type": "string", "enum": ["full", "partial", "none"]},
        "correctness": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "reason": {"type": "string"},
    },
    "required": ["claims", "relevance", "correctness", "reason"],
    "additionalProperties": False,
}

CONTEXT_CHARS = 300
RELEVANCE_SCORE = {"full": 1.0, "partial": 0.5, "none": 0.0}


def format_evidence(res: dict, doc_text) -> str:
    blocks = []
    for c in res.get("citations", []):
        try:
            text = doc_text(c["doc_id"])
        except Exception:  # noqa: BLE001 - a missing document simply gives no context
            text = ""
        lo, hi = max(0, c["start"] - CONTEXT_CHARS), min(len(text), c["end"] + CONTEXT_CHARS)
        context = (text[lo:c["start"]] + "<<" + text[c["start"]:c["end"]] + ">>" + text[c["end"]:hi]) if text else c["quote"]
        status = c.get("lifecycle", "in_force").replace("_", " ")
        when = f", effective {c['effective_date']}" if c.get("effective_date") else ""
        blocks.append(f"[{c['n']}] {c['title']} / {c.get('section') or '-'} (status: {status}{when})\n"
                      f"...{' '.join(context.split())}...")
    return "\n\n".join(blocks) or "(the assistant cited nothing)"


class Judge:
    def __init__(self, model: str | None = None, effort: str = "medium"):
        from app.config import Settings
        from app.qa.llm import LLMAnswerer

        settings = Settings()
        if model:
            settings = replace(settings, openai_model=model)
        self.llm = LLMAnswerer(settings)
        if not self.llm.available():
            raise SystemExit("--judge needs OPENAI_API_KEY")
        self.effort = effort

    @property
    def label(self) -> str:
        return f"openai:{self.llm.model}"

    @property
    def calls(self) -> int:
        return self.llm.calls

    def grade(self, q: dict, res: dict, as_of: str, doc_text) -> dict:
        history = "".join(f"{t['role']}: {t['content']}\n" for t in q.get("history", []))
        user = (f"Date the question is asked: {as_of}\n\n"
                + (f"Earlier conversation:\n{history}\n" if history else "")
                + f"Question: {q['question']}\n\n"
                f"Reference answer: {q.get('expected_answer', '')}\n\n"
                f"Assistant's answer:\n{res['answer']}\n\n"
                f"Evidence (cited passages; the quoted part is between << >>):\n{format_evidence(res, doc_text)}")
        try:
            data = self.llm._create(JUDGE_PROMPT, [{"role": "user", "content": user}], JUDGE_SCHEMA, self.effort)
        except Exception as exc:  # noqa: BLE001 - a failed judgement is reported, not fatal
            return {"judge_error": str(exc)[:200]}
        factual = [c for c in data.get("claims", []) if c.get("kind") == "factual"]
        return {
            "judge_correctness": data.get("correctness"),
            "judge_relevance": data.get("relevance"),
            "judge_faithfulness": (sum(bool(c["supported"]) for c in factual) / len(factual)) if factual else None,
            "judge_unsupported": [c["claim"] for c in factual if not c["supported"]],
            "judge_claims": len(factual),
            "judge_reason": data.get("reason", ""),
        }


def summarize_judgements(rows: list[dict]) -> dict | None:
    judged = [r for r in rows if r.get("judge_correctness")]
    if not judged:
        return None

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    faith = [r["judge_faithfulness"] for r in judged if r.get("judge_faithfulness") is not None]
    return {
        "judged": len(judged),
        "judge_errors": sum(1 for r in rows if r.get("judge_error")),
        "judge_correct": mean(r["judge_correctness"] == "correct" for r in judged),
        "judge_correct_or_partial": mean(r["judge_correctness"] in ("correct", "partial") for r in judged),
        "faithfulness": mean(faith),
        "answers_fully_faithful": mean(f == 1.0 for f in faith),
        "relevance": mean(RELEVANCE_SCORE.get(r["judge_relevance"], 0.0) for r in judged),
        # calibration against the independent regex grader
        "agreement_with_regex": mean((r["judge_correctness"] == "correct") == bool(r["correct"]) for r in judged),
        "disagreements": [r["id"] for r in judged if (r["judge_correctness"] == "correct") != bool(r["correct"])],
    }
