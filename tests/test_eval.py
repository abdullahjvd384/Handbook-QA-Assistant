"""The eval harness's own arithmetic: retrieval metrics and judge aggregation (no network)."""
import math

from eval.judge import format_evidence, summarize_judgements
from eval.run_eval import retrieval_metrics


def _q(gold, acceptable=(), stale=()):
    return {"supporting_docs": list(gold), "acceptable_docs": list(acceptable), "stale_docs": list(stale)}


def test_recall_ndcg_and_precision():
    m = retrieval_metrics(_q(["a", "b"]), ["a", "x", "b"], ["a", "a", "x", "b", "y"])
    assert m["hit@1"] and m["rr"] == 1.0
    assert m["recall@3"] == 1.0 and retrieval_metrics(_q(["a", "b"]), ["a", "x", "y", "z"])["recall@3"] == 0.5
    assert math.isclose(m["ndcg@5"], (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert m["precision@5"] == 3 / 5           # counts chunks, not documents


def test_stale_above_current():
    m = retrieval_metrics(_q(["new"], stale=["old"]), ["old", "new"])
    assert m["stale_above_current"] and m["rr"] == 0.5
    assert retrieval_metrics(_q([]), ["x"]) == {}   # unanswerable questions have no retrieval metrics


def test_judge_summary_and_calibration():
    rows = [
        {"id": "a", "correct": True, "judge_correctness": "correct", "judge_relevance": "full", "judge_faithfulness": 1.0},
        {"id": "b", "correct": False, "judge_correctness": "correct", "judge_relevance": "full", "judge_faithfulness": 0.5},
        {"id": "c", "correct": False, "judge_correctness": "incorrect", "judge_relevance": "partial", "judge_faithfulness": None},
        {"id": "d", "correct": True, "judge_error": "timeout"},
    ]
    s = summarize_judgements(rows)
    assert s["judged"] == 3 and s["judge_errors"] == 1
    assert s["faithfulness"] == 0.75 and s["answers_fully_faithful"] == 0.5
    assert s["relevance"] == (1 + 1 + 0.5) / 3
    assert s["disagreements"] == ["b"] and math.isclose(s["agreement_with_regex"], 2 / 3)


def test_judge_evidence_marks_the_quote_in_context():
    text = "Intro. Full-time employees accrue 16 days of annual leave. Outro."
    start = text.index("Full"); end = text.index(" Outro")
    res = {"citations": [{"n": 1, "doc_id": "d", "title": "Leave Policy v2", "section": "2. Annual leave",
                          "quote": text[start:end], "start": start, "end": end, "lifecycle": "in_force"}]}
    ev = format_evidence(res, lambda _id: text)
    assert "<<Full-time employees accrue 16 days of annual leave.>>" in ev and "Intro." in ev
    assert format_evidence({"citations": []}, lambda _id: "") == "(the assistant cited nothing)"
