"""The LLM path, exercised with a fake OpenAI client: no network, no API key."""
import json
from types import SimpleNamespace

from app.qa.llm import LLMAnswerer
from app.qa.service import QAService

from .conftest import TODAY


def _is_planning(kwargs) -> bool:
    return "search_query" in json.dumps(kwargs["response_format"]["json_schema"])


def _echo_plan(kwargs) -> dict:
    """Default planning reply: the latest message, unchanged, as both question and query."""
    content = kwargs["messages"][-1]["content"]
    q = content.rsplit("Latest message: ", 1)[-1]
    return {"question": q, "search_query": q}


def _completion(reply: dict, refusal: str | None = None):
    msg = SimpleNamespace(content=None if refusal else json.dumps(reply), refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


class FakeCompletions:
    """Stands in for client.chat.completions. Replies are consumed in order; a planning
    request takes a reply only if the next one is a plan, otherwise it echoes the question."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []          # answer requests
        self.plan_requests = []     # search-planning requests

    def create(self, **kwargs):
        if _is_planning(kwargs):
            self.plan_requests.append(kwargs)
            if self.replies and isinstance(self.replies[0], dict) and "search_query" in self.replies[0]:
                return _completion(self.replies.pop(0))
            return _completion(_echo_plan(kwargs))
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, SimpleNamespace):
            return reply
        return _completion(reply)


class FakeClient:
    def __init__(self, replies):
        self.completions = FakeCompletions(replies)
        self.chat = SimpleNamespace(completions=self.completions)


def make_service(kb, settings, replies):
    client = FakeClient(replies)
    return QAService(kb, settings, LLMAnswerer(settings, client=client)), client.completions


def test_valid_quotes_are_kept_and_invented_ones_dropped(kb, settings):
    reply = {
        "answerable": True,
        "answer": "Full-time employees get 16 days of annual leave per year [1]. Managers approve it [2]. Bonus fact [3].",
        "citations": [
            {"passage_id": "P1", "quote": "Full-time employees accrue 16 days of annual leave per calendar year"},
            {"passage_id": "P1", "quote": "Leave is approved by the CEO personally."},          # not in source
            {"passage_id": "P99", "quote": "anything"},                                        # unknown passage
        ],
    }
    qa, client = make_service(kb, settings, [reply])
    res = qa.ask("How many days of annual leave do I get?", as_of=TODAY, mode="llm")
    assert res["mode"] == "llm" and res["llm_calls"] == 2      # search planning + answer
    assert len(res["citations"]) == 1
    c = res["citations"][0]
    assert c["doc_id"] == "hr-leave-policy-v2"
    assert kb.snapshot.docs[c["doc_id"]].text[c["start"]:c["end"]] == c["quote"]
    # markers for dropped citations are removed; the kept one stays [1]
    assert "[1]" in res["answer"] and "[2]" not in res["answer"] and "[3]" not in res["answer"]


def test_request_shape(kb, settings):
    reply = {"answerable": True, "answer": "16 days [1].",
             "citations": [{"passage_id": "P1", "quote": "Full-time employees accrue 16 days of annual leave per calendar year"}]}
    qa, client = make_service(kb, settings, [reply])
    qa.ask("How many days of annual leave do I get?", as_of=TODAY, mode="llm")
    req = client.requests[0]
    assert req["model"] == settings.openai_model
    assert req["messages"][0]["role"] == "system"
    assert req["response_format"]["json_schema"]["strict"] is True
    assert "Today's date: 2026-11-14" in req["messages"][-1]["content"]
    assert "SUPERSEDED" in req["messages"][-1]["content"] or "IN FORCE" in req["messages"][-1]["content"]
    assert "temperature" not in req


def test_unverifiable_answer_falls_back_to_extractive(kb, settings):
    reply = {"answerable": True, "answer": "You get 30 days [1].",
             "citations": [{"passage_id": "P1", "quote": "Employees get 30 days of leave."}]}
    qa, _ = make_service(kb, settings, [reply])
    res = qa.ask("How many days of annual leave do I get?", as_of=TODAY, mode="llm")
    assert res["mode"] == "extractive (fallback)"
    assert "16 days" in res["answer"]
    assert any("could not be verified" in w for w in res["warnings"])


def test_refusal_or_error_falls_back(kb, settings):
    refusal = _completion({}, refusal="I can't help with that.")
    qa, _ = make_service(kb, settings, [refusal])
    res = qa.ask("How many days of annual leave do I get?", as_of=TODAY, mode="llm")
    assert res["mode"] == "extractive (fallback)"
    assert res["citations"]


def test_unexpected_client_exception_falls_back(kb, settings):
    """Not only API errors: any failure inside the provider SDK must degrade, never 500."""
    qa, _ = make_service(kb, settings, [RuntimeError("connection reset by peer")])
    res = qa.ask("How many days of annual leave do I get?", as_of=TODAY, mode="llm")
    assert res["mode"] == "extractive (fallback)" and "16 days" in res["answer"]
    assert any("connection reset" in w for w in res["warnings"])


def test_small_talk_makes_no_model_calls(kb, settings):
    qa, client = make_service(kb, settings, [])
    res = qa.ask("Hi there!", as_of=TODAY, mode="llm")
    assert res["mode"] == "small-talk" and res["llm_calls"] == 0 and not res["citations"]
    assert not client.requests and not client.plan_requests


def test_follow_up_is_rewritten_then_answered(kb, settings):
    rewrite = {"question": "What is the per diem for an L4 employee in Seattle?",
               "search_query": "per diem table grade L4 city tier Seattle"}
    answer = {"answerable": True, "answer": "$85 per day [1].",
              "citations": [{"passage_id": "P1", "quote": "grade: L4 | city tier: 2 | example cities: Chicago, Seattle, Los Angeles, Washington DC | per diem usd: 85"}]}
    qa, client = make_service(kb, settings, [rewrite, answer])
    history = [{"role": "user", "content": "What is the per diem for an L4 employee in New York?"},
               {"role": "assistant", "content": "$100."}]
    res = qa.ask("What about Seattle?", history=history, as_of=TODAY, mode="llm")
    assert res["llm_calls"] == 2
    assert res["standalone_question"] == rewrite["question"]
    assert res["citations"][0]["doc_id"] == "fin-per-diem-table"
    # history is passed to the answer call, ending with the new question
    msgs = client.requests[0]["messages"]
    assert [m["role"] for m in msgs[:2]] == ["system", "user"] and msgs[-1]["role"] == "user"
    assert "Handbook documents:" in client.plan_requests[0]["messages"][-1]["content"]


def test_llm_needs_an_openai_key(settings, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not LLMAnswerer(settings).available()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert LLMAnswerer(settings).available()


def test_unanswerable_from_llm(kb, settings):
    reply = {"answerable": False, "answer": "The handbook doesn't say which insurer provides dental cover.", "citations": []}
    qa, _ = make_service(kb, settings, [reply])
    res = qa.ask("Which insurance company provides our dental plan?", as_of=TODAY, mode="llm")
    assert res["answerable"] is False and res["citations"] == [] and res["mode"] == "llm"


def test_search_planning_bridges_vocabulary(kb, settings):
    """'complain about my manager' shares no words with the Grievance Policy; the planned
    handbook-vocabulary query is searched too and fused with the original."""
    plan = {"question": "If I formally complain about my manager, how soon will I hear back?",
            "search_query": "Grievance Policy acknowledgement working days"}
    answer = {"answerable": True, "answer": "Within two working days [1].",
              "citations": [{"passage_id": "P1", "quote": "You will receive an acknowledgement within two working days"}]}
    qa, _ = make_service(kb, settings, [plan, answer])
    res = qa.ask(plan["question"], as_of=TODAY, mode="llm")
    assert res["search_query"] == plan["search_query"]
    assert "hr-grievance-policy" in [r["doc_id"] for r in res["retrieved"]]


def test_short_documents_are_passed_whole_to_the_llm(kb, settings):
    """A city's tier (section 3) and its hotel cap (section 4.2) must both reach the model."""
    answer = {"answerable": True, "answer": "$200 [1].",
              "citations": [{"passage_id": "P1", "quote": "City tier: Tier 2 | Maximum room rate per night: $200"}]}
    qa, client = make_service(kb, settings, [answer])
    res = qa.ask("What's the hotel limit per night in Seattle?", as_of=TODAY, mode="llm")
    prompt = client.requests[0]["messages"][-1]["content"]
    assert "Tier 2: Chicago, Seattle" in prompt and "Maximum room rate per night: $200" in prompt
    assert res["citations"][0]["section"].endswith("4.2 Hotels and daily spending limits")
