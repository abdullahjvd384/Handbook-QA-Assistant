"""Run the evaluation set and report how well the assistant did.

    python -m eval.run_eval                       # in-process, answer mode from env (auto)
    python -m eval.run_eval --mode extractive     # force the no-LLM answerer
    python -m eval.run_eval --mode llm            # LLM (needs OPENAI_API_KEY or ANTHROPIC_API_KEY)
    python -m eval.run_eval --base-url http://localhost:8000   # through the HTTP API
    python -m eval.run_eval --ablation            # also compare retrieval configurations

What is measured (per question, then aggregated overall and per category):
  * answer correct    - every `must_include` regex matches, no `must_not_include` matches,
                        and the answerable flag is right. Unanswerable questions pass if the
                        assistant abstains (or matches `accept_if`, e.g. "there is no London office").
  * citation support  - at least one citation comes from a gold or acceptable document.
  * citation precision- share of all citations that come from gold or acceptable documents.
  * quote verified    - the quoted text is exactly the source text at the cited offsets.
  * stale primary     - the first citation comes from a superseded/deprecated document
                        when a current one exists (lower is better).
  * retrieval hit@k / MRR - a gold document appears among the top-k retrieved documents.
  * recall@k              - share of the gold documents that appear among the top-k retrieved
                            documents (differs from hit@k on questions with several gold documents).
  * precision@5           - share of the top-5 retrieved chunks that come from a gold or acceptable document.
  * nDCG@5                - rank-weighted retrieval quality over documents: a gold document at
                            rank 1 counts fully, lower ranks count less (binary relevance).
  * stale above current   - a stale document is ranked above the current gold one.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.judge import Judge, summarize_judgements  # noqa: E402 - needs ROOT on sys.path

DEFAULT_AS_OF = "2026-11-14"   # pinned so results don't change when 2027 policies take effect


def load_questions(path: Path) -> list[dict]:
    qs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            q.setdefault("answerable", True)
            for key in ("must_include", "must_not_include", "supporting_docs", "acceptable_docs", "stale_docs", "accept_if", "history"):
                q.setdefault(key, [])
            qs.append(q)
    return qs


# ---------------------------------------------------------------------------
# clients

class InProcessClient:
    def __init__(self, mode: str | None):
        from app.config import Settings
        from app.qa.service import QAService
        from app.search.index import KnowledgeBase

        self.settings = Settings()
        self.kb = KnowledgeBase(self.settings).load()
        self.qa = QAService(self.kb, self.settings)
        self.mode = mode
        self.label = "in-process"

    def ask(self, q: dict, as_of: str, search_config=None) -> dict:
        from app.search.index import SearchConfig
        return self.qa.ask(q["question"], q["history"], date.fromisoformat(as_of), self.mode,
                           search_config=search_config or SearchConfig())

    def doc_text(self, doc_id: str) -> str:
        return self.kb.snapshot.docs[doc_id].text

    def doc_format(self, doc_id: str) -> str:
        d = self.kb.snapshot.docs.get(doc_id)
        return d.meta.format if d else "?"


class HttpClient:
    def __init__(self, base_url: str, mode: str | None):
        import httpx
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=120)
        self.mode = mode
        self.label = base_url
        self._docs: dict[str, dict] = {}

    def ask(self, q: dict, as_of: str, search_config=None) -> dict:
        body = {"question": q["question"], "history": q["history"], "as_of": as_of}
        if self.mode:
            body["mode"] = self.mode
        r = self.http.post("/api/ask", json=body)
        r.raise_for_status()
        return r.json()

    def _doc(self, doc_id: str) -> dict:
        if doc_id not in self._docs:
            r = self.http.get(f"/api/documents/{doc_id}")
            r.raise_for_status()
            self._docs[doc_id] = r.json()
        return self._docs[doc_id]

    def doc_text(self, doc_id: str) -> str:
        return self._doc(doc_id)["text"]

    def doc_format(self, doc_id: str) -> str:
        try:
            return self._doc(doc_id)["format"]
        except Exception:  # noqa: BLE001
            return "?"


# ---------------------------------------------------------------------------
# grading

def _matches(patterns: list[str], text: str) -> list[bool]:
    return [bool(re.search(p, text, re.I | re.S)) for p in patterns]


def unique_docs(retrieved: list[dict]) -> list[str]:
    out: list[str] = []
    for r in retrieved:
        if r["doc_id"] not in out:
            out.append(r["doc_id"])
    return out


def retrieval_metrics(q: dict, ranked_docs: list[str], chunk_docs: list[str] | None = None) -> dict:
    """ranked_docs: distinct documents in rank order. chunk_docs: the document of each
    retrieved chunk in rank order (for precision@5, which counts chunks)."""
    gold = set(q["supporting_docs"])
    if not gold:
        return {}
    rank = next((i + 1 for i, d in enumerate(ranked_docs) if d in gold), None)
    out = {f"hit@{k}": rank is not None and rank <= k for k in (1, 3, 5)}
    out["rr"] = 1.0 / rank if rank else 0.0
    for k in (3, 5):
        out[f"recall@{k}"] = len(gold & set(ranked_docs[:k])) / len(gold)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked_docs[:5]) if d in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 5)))
    out["ndcg@5"] = dcg / idcg
    top5 = (chunk_docs if chunk_docs is not None else ranked_docs)[:5]
    if top5:
        relevant = gold | set(q["acceptable_docs"])
        out["precision@5"] = sum(d in relevant for d in top5) / len(top5)
    if q["stale_docs"]:
        stale_rank = next((i + 1 for i, d in enumerate(ranked_docs) if d in q["stale_docs"]), None)
        out["stale_above_current"] = stale_rank is not None and (rank is None or stale_rank < rank)
    return out


def answer_verdict(q: dict, text: str, answerable: bool) -> dict:
    """The regex part of grading: depends only on the answer text and its answerable flag."""
    inc = _matches(q["must_include"], text)
    exc = _matches(q["must_not_include"], text)
    if q["answerable"]:
        correct = answerable and all(inc) and not any(exc)
    else:
        accepted = bool(q["accept_if"]) and all(_matches(q["accept_if"], text))
        correct = (not answerable or accepted) and not any(exc)
    return {"correct": correct,
            "missing": [p for p, m in zip(q["must_include"], inc) if not m],
            "forbidden_found": [p for p, m in zip(q["must_not_include"], exc) if m]}


def grade(q: dict, res: dict, client) -> dict:
    text = res["answer"]
    ok_docs = set(q["supporting_docs"]) | set(q["acceptable_docs"])
    cites = res["citations"]
    verdict = answer_verdict(q, text, res["answerable"])

    verified = []
    for c in cites:
        try:
            verified.append(client.doc_text(c["doc_id"])[c["start"]:c["end"]] == c["quote"])
        except Exception:  # noqa: BLE001
            verified.append(False)

    g = {
        "id": q["id"], "category": q["category"], "question": q["question"], "answerable_expected": q["answerable"],
        "answerable_got": res["answerable"], **verdict,
        "citations": [c["doc_id"] for c in cites],
        "citation_ok": [c["doc_id"] in ok_docs for c in cites],
        "quote_verified": verified,
        "answer": text, "expected": q.get("expected_answer", ""),
        "latency_ms": res.get("latency_ms", 0), "llm_calls": res.get("llm_calls", 0), "mode": res.get("mode"),
        "format": client.doc_format(q["supporting_docs"][0]) if q["supporting_docs"] else "-",
        "warnings": res.get("warnings", []),
    }
    if q["answerable"]:
        g["citation_support"] = any(g["citation_ok"])
        if q["stale_docs"] and cites:
            g["stale_primary"] = cites[0]["doc_id"] in q["stale_docs"] and cites[0]["doc_id"] not in q["supporting_docs"]
    retrieved = res.get("retrieved", [])
    ranked = unique_docs(retrieved)
    g["retrieved_docs"] = ranked
    g.update(retrieval_metrics(q, ranked, [r["doc_id"] for r in retrieved]))
    return g


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else None


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable_expected"]]
    unans = [r for r in rows if not r["answerable_expected"]]
    all_ok = [ok for r in rows for ok in r["citation_ok"] if r["answerable_expected"]]
    lat = sorted(r["latency_ms"] for r in rows)
    return {
        "questions": len(rows),
        "answer_accuracy": _mean(r["correct"] for r in rows),
        "answerable_accuracy": _mean(r["correct"] for r in ans),
        "abstention_accuracy": _mean(r["correct"] for r in unans),
        "false_abstention_rate": _mean(not r["answerable_got"] for r in ans),
        "citation_support": _mean(r["citation_support"] for r in ans),
        "citation_precision": _mean(all_ok),
        "quote_verified": _mean(v for r in rows for v in r["quote_verified"]),
        "stale_primary_rate": _mean(r["stale_primary"] for r in rows if "stale_primary" in r),
        "hit@1": _mean(r["hit@1"] for r in rows if "hit@1" in r),
        "hit@3": _mean(r["hit@3"] for r in rows if "hit@3" in r),
        "hit@5": _mean(r["hit@5"] for r in rows if "hit@5" in r),
        "mrr": _mean(r["rr"] for r in rows if "rr" in r),
        "recall@3": _mean(r["recall@3"] for r in rows if "recall@3" in r),
        "recall@5": _mean(r["recall@5"] for r in rows if "recall@5" in r),
        "ndcg@5": _mean(r["ndcg@5"] for r in rows if "ndcg@5" in r),
        "precision@5": _mean(r["precision@5"] for r in rows if "precision@5" in r),
        "stale_above_current": _mean(r["stale_above_current"] for r in rows if "stale_above_current" in r),
        "latency_p50_ms": lat[len(lat) // 2] if lat else None,
        "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
        "llm_calls": sum(r["llm_calls"] for r in rows),
    }


def by_group(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {k: {"n": len(v), "accuracy": _mean(r["correct"] for r in v),
                "hit@5": _mean(r["hit@5"] for r in v if "hit@5" in r)} for k, v in sorted(groups.items())}


def fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.1%}" if x <= 1.0 else f"{x:.0f}"
    return str(x)


# ---------------------------------------------------------------------------
# ablation

# (label, model, tuned dense weight): each embedder is compared at the RRF weight that suits it
EMBEDDERS = [("model2vec", "model2vec:minishlab/potion-base-8M", 0.5), ("bge-small", "fastembed:BAAI/bge-small-en-v1.5", 2.0)]
RERANKERS = [("bge-reranker-base", "BAAI/bge-reranker-base"), ("MiniLM-L6", "Xenova/ms-marco-MiniLM-L-6-v2")]


def ablation(client, questions: list[dict], as_of_default: str) -> list[dict]:
    """Retrieval configurations x embedding models x re-rankers, answered in extractive mode
    (deterministic, no LLM cost). Each embedding model gets its own in-memory index so the
    comparison never touches the app's Qdrant collection."""
    from app.config import Settings
    from app.qa.extractive import followup_query
    from app.qa.service import QAService
    from app.search.index import KnowledgeBase, SearchConfig

    kbs, weights = {}, {}
    for label, spec, weight in EMBEDDERS:
        kb = KnowledgeBase(Settings(dense_model=spec, vector_store="memory", use_reranker=False)).load()
        if kb.dense_enabled:
            kbs[label], weights[label] = (kb, QAService(kb, kb.settings)), weight
    if not kbs:
        return []
    first = next(iter(kbs))
    configs = [
        ("BM25 only", first, SearchConfig(True, False, False, False, rerank=False)),
        ("BM25 + thesaurus", first, SearchConfig(True, False, False, True, rerank=False)),
        ("BM25 + thesaurus + lifecycle", first, SearchConfig(True, False, True, True, rerank=False)),
    ]
    for label in kbs:
        w = weights[label]
        configs += [
            (f"Dense only ({label})", label, SearchConfig(False, True, False, False, dense_weight=w, rerank=False)),
            (f"Hybrid RRF ({label}, w={w})", label, SearchConfig(True, True, False, True, dense_weight=w, rerank=False)),
            (f"Hybrid + lifecycle ({label}, w={w})", label, SearchConfig(True, True, True, True, dense_weight=w, rerank=False)),
        ]
    best = "bge-small" if "bge-small" in kbs else first
    for rlabel, rmodel in RERANKERS:
        configs.append((f"Hybrid + lifecycle ({best}) + rerank {rlabel}", best,
                        SearchConfig(True, True, True, True, dense_weight=weights[best], rerank=True, rerank_model=rmodel)))

    out = []
    for name, label, cfg in configs:
        kb, qa = kbs[label]
        rows, search_ms = [], []
        for q in questions:
            as_of = date.fromisoformat(q.get("as_of", as_of_default))
            query, _ = followup_query(q["question"], q["history"]) if q["history"] else (q["question"], [])
            t0 = time.perf_counter()
            hits = kb.search(query, as_of, k=8, config=cfg)
            search_ms.append((time.perf_counter() - t0) * 1000)
            chunk_docs = [h.chunk.doc_id for h in hits]
            m = retrieval_metrics(q, list(dict.fromkeys(chunk_docs)), chunk_docs)
            res = qa.ask(q["question"], q["history"], as_of, "extractive", search_config=cfg)
            m["correct"] = grade(q, res, client)["correct"]
            rows.append(m)
        search_ms.sort()
        out.append({
            "config": name,
            "hit@1": _mean(r["hit@1"] for r in rows if "hit@1" in r),
            "hit@3": _mean(r["hit@3"] for r in rows if "hit@3" in r),
            "hit@5": _mean(r["hit@5"] for r in rows if "hit@5" in r),
            "mrr": _mean(r["rr"] for r in rows if "rr" in r),
            "recall@3": _mean(r["recall@3"] for r in rows if "recall@3" in r),
            "recall@5": _mean(r["recall@5"] for r in rows if "recall@5" in r),
            "ndcg@5": _mean(r["ndcg@5"] for r in rows if "ndcg@5" in r),
            "precision@5": _mean(r["precision@5"] for r in rows if "precision@5" in r),
            "stale_above_current": _mean(r["stale_above_current"] for r in rows if "stale_above_current" in r),
            "extractive_accuracy": _mean(r["correct"] for r in rows),
            "search_ms_p50": round(search_ms[len(search_ms) // 2], 1),
        })
    return out


# ---------------------------------------------------------------------------

def regrade(questions: list[dict], path: Path) -> int:
    """Grading is deterministic given the answer text, so a corrected regex can be re-applied to a
    saved run without asking the model again. The file records which verdicts changed."""
    data = json.loads(path.read_text(encoding="utf-8"))
    byid = {q["id"]: q for q in questions}
    changed = []
    for row in data["questions"]:
        q = byid.get(row["id"])
        if q is None:
            continue
        verdict = answer_verdict(q, row["answer"], row["answerable_got"])
        if verdict["correct"] != row["correct"]:
            changed.append({"id": row["id"], "before": row["correct"], "after": verdict["correct"]})
        row.update(verdict)
    rows = data["questions"]
    data["summary"] = {**summarize(rows), "latency_p50_ms": data["summary"]["latency_p50_ms"],
                       "latency_p95_ms": data["summary"]["latency_p95_ms"]}
    data["by_category"] = by_group(rows, "category")
    data["by_format"] = by_group([r for r in rows if r["format"] != "-"], "format")
    if data.get("judge"):
        keep = {k: data["judge"][k] for k in ("judge_model", "judge_calls") if k in data["judge"]}
        data["judge"] = {**keep, **(summarize_judgements(rows) or {})}
    data.setdefault("regrades", []).append({"at": datetime.now().isoformat(timespec="seconds"), "changed": changed})
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Regraded {path}: {len(changed)} verdict(s) changed {changed}; "
          f"answer accuracy now {fmt(data['summary']['answer_accuracy'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--mode", choices=["auto", "llm", "extractive"], default=None)
    ap.add_argument("--base-url", default=None, help="Evaluate a running server through its HTTP API")
    ap.add_argument("--as-of", default=DEFAULT_AS_OF)
    ap.add_argument("--only", default=None, help="Comma-separated question ids")
    ap.add_argument("--ablation", action="store_true", help="Compare retrieval configurations (in-process only)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--judge", action="store_true",
                    help="Also grade each answer with an LLM judge (faithfulness, relevance, correctness); 1 call per question")
    ap.add_argument("--judge-model", default=None, help="Judge model (default: the configured answer model)")
    ap.add_argument("--regrade", action="store_true",
                    help="Re-apply the (edited) regexes to the answers stored in --out/results-<mode>.json; no model calls")
    args = ap.parse_args(argv)

    questions = load_questions(Path(args.questions))
    if args.regrade:
        return regrade(questions, Path(args.out) / f"results-{args.mode or 'llm'}.json")
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    client = HttpClient(args.base_url, args.mode) if args.base_url else InProcessClient(args.mode)
    judge = Judge(args.judge_model) if args.judge else None

    rows = []
    answer_models: set[str] = set()
    t0 = time.perf_counter()
    for q in questions:
        as_of = q.get("as_of", args.as_of)
        res = client.ask(q, as_of)
        rows.append(grade(q, res, client))
        if res.get("model"):
            answer_models.add(res["model"])
        if judge is not None:
            rows[-1].update(judge.grade(q, res, as_of, client.doc_text))
        if not args.quiet:
            mark = "PASS" if rows[-1]["correct"] else "FAIL"
            jv = f"  judge={rows[-1].get('judge_correctness')}" if judge is not None else ""
            print(f"  {mark}  {q['id']:<24} {res['mode']:<10} {res.get('latency_ms', 0):>6} ms{jv}", flush=True)
    elapsed = time.perf_counter() - t0

    summary = summarize(rows)
    modes = sorted({r["mode"] for r in rows})
    print(f"\nEvaluated {len(rows)} questions via {client.label} in {elapsed:.1f}s; answer mode(s): {', '.join(modes)}")
    print("\n== Summary ==")
    for k, v in summary.items():
        print(f"  {k:<24} {fmt(v)}")
    print("\n== By category ==")
    cats = by_group(rows, "category")
    for k, v in cats.items():
        print(f"  {k:<22} n={v['n']:<3} accuracy={fmt(v['accuracy']):<7} hit@5={fmt(v['hit@5'])}")
    fmts = by_group([r for r in rows if r["format"] != "-"], "format")
    print("\n== By source format (gold document) ==")
    for k, v in fmts.items():
        print(f"  {k:<6} n={v['n']:<3} accuracy={fmt(v['accuracy']):<7} hit@5={fmt(v['hit@5'])}")
    failures = [r for r in rows if not r["correct"]]
    if failures:
        print("\n== Failures ==")
        for r in failures:
            why = []
            if r["answerable_expected"] != r["answerable_got"]:
                why.append("abstained" if r["answerable_expected"] else "answered an unanswerable question")
            if r["missing"]:
                why.append(f"missing {r['missing']}")
            if r["forbidden_found"]:
                why.append(f"contains forbidden {r['forbidden_found']}")
            print(f"  - {r['id']}: {'; '.join(why)}")
            print(f"      expected: {r['expected']}")
            print(f"      got:      {r['answer'][:220].replace(chr(10), ' ')}")

    abl = None
    if args.ablation:
        if not isinstance(client, InProcessClient):
            print("\n(--ablation runs in-process only; skipped)")
        else:
            print("\n== Retrieval ablation (extractive answers, no LLM calls) ==")
            abl = ablation(client, questions, args.as_of)
            print(f"  {'config':<52} {'hit@1':>6} {'hit@3':>6} {'MRR':>6} {'R@5':>6} {'nDCG@5':>6} {'P@5':>6} {'stale>cur':>9} {'ans acc':>8} {'ms':>6}")
            for a in abl:
                print(f"  {a['config']:<52} {fmt(a['hit@1']):>6} {fmt(a['hit@3']):>6} {fmt(a['mrr']):>6} {fmt(a['recall@5']):>6} {fmt(a['ndcg@5']):>6} "
                      f"{fmt(a['precision@5']):>6} {fmt(a['stale_above_current']):>9} "
                      f"{fmt(a['extractive_accuracy']):>8} {a['search_ms_p50']:>6}")

    judge_summary = None
    if judge is not None:
        judge_summary = summarize_judgements(rows)
        if judge_summary:
            judge_summary = {"judge_model": judge.label, "judge_calls": judge.calls, **judge_summary}
            print(f"\n== LLM judge ({judge.label}, {judge.calls} calls) ==")
            for k, v in judge_summary.items():
                if k not in ("disagreements", "judge_model", "judge_calls"):
                    print(f"  {k:<26} {fmt(v)}")
            byid = {r["id"]: r for r in rows}
            if judge_summary["disagreements"]:
                print("  disagreements with the regex grader (read these by hand):")
                for i in judge_summary["disagreements"]:
                    r = byid[i]
                    print(f"    - {i}: regex={'pass' if r['correct'] else 'fail'}, judge={r['judge_correctness']}: "
                          f"{r.get('judge_reason', '')[:160]}")
            unsupported = [(r["id"], c) for r in rows for c in r.get("judge_unsupported", [])]
            if unsupported:
                print("  claims the judge found unsupported by the cited text:")
                for i, c in unsupported:
                    print(f"    - {i}: {c[:160]}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(sorted({m.split()[0] for m in modes}))
    payload = {"run_at": datetime.now().isoformat(timespec="seconds"), "as_of": args.as_of, "client": client.label,
               "question_set": Path(args.questions).name, "modes": modes, "summary": summary,
               "by_category": cats, "by_format": fmts, "ablation": abl, "judge": judge_summary, "answer_models": sorted(answer_models), "questions": rows}
    path = out_dir / f"results-{tag}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
