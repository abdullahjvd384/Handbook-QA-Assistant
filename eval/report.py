"""Build eval/RESULTS.md from the JSON files that run_eval.py writes.

    python -m eval.report

Every number in RESULTS.md comes from eval/results/<set>/results-<mode>.json, so the report
can be regenerated after any run and never drifts from what the harness measured.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETS = [("questions", "dev"), ("heldout", "held-out"), ("paraphrase", "paraphrase")]
MODES = ["llm", "extractive"]


def load(set_name: str, mode: str) -> dict | None:
    p = ROOT / "results" / set_name / f"results-{mode}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(v) -> str:
    return "–" if v is None else f"{v * 100:.1f}%"


def num(v) -> str:
    return "–" if v is None else f"{v * 100:.1f}"


def ms(v) -> str:
    return "–" if v is None else (f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms")


ROWS = [
    ("Answer accuracy (regex grader)", "answer_accuracy", pct),
    ("Abstention accuracy (unanswerable questions)", "abstention_accuracy", pct),
    ("Citation support / precision", ("citation_support", "citation_precision"), pct),
    ("Quotes verified verbatim", "quote_verified", pct),
    ("Stale document as primary citation", "stale_primary_rate", pct),
    ("Retrieval hit@1 / hit@3", ("hit@1", "hit@3"), num),
    ("MRR", "mrr", num),
    ("Recall@3 / recall@5", ("recall@3", "recall@5"), num),
    ("nDCG@5", "ndcg@5", num),
    ("Precision@5 (chunks)", "precision@5", num),
    ("Stale version ranked above current", "stale_above_current", pct),
    ("Latency p50 / p95", ("latency_p50_ms", "latency_p95_ms"), ms),
    ("Answer-model calls", "llm_calls", lambda v: str(v)),
]
JUDGE_ROWS = [
    ("Judge: correct", "judge_correct", pct),
    ("Judge: correct or partial", "judge_correct_or_partial", pct),
    ("Judge: faithfulness (supported claims)", "faithfulness", pct),
    ("Judge: answers with every claim supported", "answers_fully_faithful", pct),
    ("Judge: relevance", "relevance", pct),
    ("Judge agrees with regex grader", "agreement_with_regex", pct),
    ("Judge calls", "judge_calls", lambda v: str(v)),
]


def cell(src: dict | None, key, fmt) -> str:
    if not src:
        return "–"
    if isinstance(key, tuple):
        return " / ".join(fmt(src.get(k)) for k in key)
    return fmt(src.get(key))


def main_table(results: dict) -> list[str]:
    cols = [(s, label, m) for s, label in SETS for m in MODES]
    head = "| metric | " + " | ".join(f"{label} · {m}" for _, label, m in cols) + " |"
    out = [head, "|---" * (len(cols) + 1) + "|"]
    for title, key, fmt in ROWS:
        out.append(f"| {title} | " + " | ".join(cell((results[(s, m)] or {}).get("summary"), key, fmt)
                                                for s, _, m in cols) + " |")
    for title, key, fmt in JUDGE_ROWS:
        out.append(f"| {title} | " + " | ".join(cell((results[(s, m)] or {}).get("judge"), key, fmt)
                                                for s, _, m in cols) + " |")
    return out


def ablation_table(results: dict) -> list[str]:
    dev = (results[("questions", "extractive")] or {}).get("ablation") or []
    para = {a["config"]: a for a in (results[("paraphrase", "extractive")] or {}).get("ablation") or []}
    held = {a["config"]: a for a in (results[("heldout", "extractive")] or {}).get("ablation") or []}
    out = ["| configuration | hit@1 | MRR | recall@5 | nDCG@5 | P@5 | stale above current | extractive accuracy "
           "| paraphrase hit@1 | held-out hit@1 | search p50 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in dev:
        out.append(f"| {a['config']} | {num(a['hit@1'])} | {num(a['mrr'])} | {num(a.get('recall@5'))} | "
                   f"{num(a.get('ndcg@5'))} | {num(a['precision@5'])} | {pct(a['stale_above_current'])} | "
                   f"{pct(a['extractive_accuracy'])} | {num((para.get(a['config']) or {}).get('hit@1'))} | "
                   f"{num((held.get(a['config']) or {}).get('hit@1'))} | {a['search_ms_p50']:.0f} ms |")
    return out


def category_table(results: dict) -> list[str]:
    llm = (results[("questions", "llm")] or {}).get("by_category") or {}
    ext = (results[("questions", "extractive")] or {}).get("by_category") or {}
    out = ["| category | n | LLM | extractive |", "|---|---|---|---|"]
    for cat in sorted(set(llm) | set(ext)):
        n = (llm.get(cat) or ext.get(cat))["n"]
        out.append(f"| {cat} | {n} | {pct((llm.get(cat) or {}).get('accuracy'))} | {pct((ext.get(cat) or {}).get('accuracy'))} |")
    return out


def failures(results: dict) -> list[str]:
    out = []
    for s, label in SETS:
        for m in MODES:
            data = results[(s, m)]
            if not data:
                continue
            rows = data["questions"]
            bad = [r for r in rows if not r["correct"] or r.get("judge_correctness") not in (None, "correct")
                   or r.get("judge_unsupported")]
            if not bad:
                continue
            out.append(f"**{label} · {m}**\n")
            for r in bad:
                verdicts = [f"regex {'pass' if r['correct'] else 'FAIL'}"]
                if r.get("judge_correctness"):
                    verdicts.append(f"judge {r['judge_correctness']}")
                out.append(f"- `{r['id']}` ({', '.join(verdicts)}). Expected: {r['expected']}")
                out.append(f"  - Answer: {' '.join(r['answer'].split())[:400]}")
                if r.get("judge_reason"):
                    out.append(f"  - Judge: {r['judge_reason']}")
                for c in r.get("judge_unsupported", []):
                    out.append(f"  - Unsupported claim: {c}")
            out.append("")
    return out


def notes(results: dict) -> list[str]:
    """Caveats recorded in the result files themselves: failed judge calls and regex regrades."""
    out = []
    for s, label in SETS:
        for m in MODES:
            data = results[(s, m)]
            if not data:
                continue
            errors = [r for r in data["questions"] if r.get("judge_error")]
            if errors:
                out.append(f"* **{label} · {m}: the judge did not run for {len(errors)} of {len(data['questions'])} answers** "
                           f"({errors[0]['judge_error'][:90]}…). Those cells show –.")
            for rg in data.get("regrades", []):
                changed = ", ".join(f"`{c['id']}` {'fail→pass' if c['after'] else 'pass→fail'}" for c in rg["changed"]) or "no change"
                out.append(f"* {label} · {m}: regexes corrected and re-applied to the saved answers on {rg['at'][:10]} "
                           f"(`--regrade`, no new model calls): {changed}.")
    return out


def main() -> int:
    results = {(s, m): load(s, m) for s, _ in SETS for m in MODES}
    meta = next((r for r in results.values() if r), None)
    if meta is None:
        raise SystemExit("No results found; run python -m eval.run_eval first.")
    llm = results[("questions", "llm")] or {}
    judge = (llm.get("judge") or {}).get("judge_model", "–")
    answer_model = ", ".join(llm.get("answer_models") or []) or judge
    lines = [
        "# Evaluation results",
        "",
        "Generated by `python -m eval.report` from `eval/results/*/results-*.json`. Do not edit by hand.",
        "",
        f"* Question sets: dev `questions.jsonl` (69), held-out `heldout.jsonl` (25), paraphrase `paraphrase.jsonl` (12). "
        f"Evaluation date pinned to {meta['as_of']}.",
        f"* LLM mode: answers by `{answer_model}` (search planning + answer = 2 calls per question). "
        "Extractive mode makes no model calls.",
        f"* LLM judge: `{judge}`, one call per answer, run on both modes. Same model family as the answerer, "
        "so it is calibrated against the regex grader (agreement row) and every disagreement is listed below.",
        "* Retrieval: hybrid BM25 + `BAAI/bge-small-en-v1.5` in Qdrant, RRF dense weight 2.0, lifecycle re-rank, no cross-encoder.",
        *notes(results),
        "",
        "## All metrics",
        "",
        *main_table(results),
        "",
        "Retrieval columns (hit, MRR, recall, nDCG, P@5) are per document for answerable questions. "
        "In LLM mode they measure the fused retrieval for the question and the LLM's search restatement.",
        "",
        "## Retrieval ablation",
        "",
        "Extractive answers so no model calls; latency is search only, on CPU. Dense weight was tuned on dev and "
        "paraphrase only.",
        "",
        *ablation_table(results),
        "",
        "## Dev accuracy by question category",
        "",
        *category_table(results),
        "",
        "## Every failure, judge disagreement and unsupported claim",
        "",
        *(failures(results) or ["None."]),
    ]
    out = ROOT / "RESULTS.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
