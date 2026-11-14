from datetime import date

from app.qa.verify import locate_quote

from .conftest import TODAY


def ask(qa, q, **kw):
    return qa.ask(q, as_of=kw.pop("as_of", TODAY), mode="extractive", **kw)


def assert_citations_are_verbatim(qa, res):
    docs = qa.kb.snapshot.docs
    for c in res["citations"]:
        assert docs[c["doc_id"]].text[c["start"]:c["end"]] == c["quote"]


def test_answer_quotes_current_policy(qa):
    res = ask(qa, "How many days of annual leave do full-time employees get?")
    assert res["answerable"]
    assert "16 days" in res["answer"]
    assert res["citations"][0]["doc_id"] == "hr-leave-policy-v2"
    assert_citations_are_verbatim(qa, res)


def test_future_revision_is_mentioned_but_current_rule_leads(qa):
    res = ask(qa, "How many weeks of paid leave do secondary caregivers get?")
    assert res["citations"][0]["doc_id"] == "hr-parental-leave-policy"
    assert "12 weeks" in res["citations"][0]["quote"]
    upcoming = [c for c in res["citations"] if c["role"] == "upcoming"]
    assert upcoming and upcoming[0]["doc_id"] == "hr-parental-leave-policy-2027"
    assert_citations_are_verbatim(qa, res)


def test_after_effective_date_the_revision_applies(qa):
    res = ask(qa, "How many weeks of paid leave do secondary caregivers get?", as_of=date(2027, 3, 1))
    roles = {c["doc_id"]: c["role"] for c in res["citations"]}
    assert roles.get("hr-parental-leave-policy-2027") != "upcoming"


def test_deprecated_runbook_is_not_the_primary_source(qa):
    res = ask(qa, "How long before a scheduled release does the deploy freeze start?")
    assert res["citations"][0]["doc_id"] == "eng-deploy-runbook"
    assert "72 hours" in res["answer"]


def test_abstains_on_topics_the_handbook_does_not_cover(qa):
    for q in ["What is the vesting schedule for stock options?", "Who is the CEO of Northgate Robotics?",
              "What is the guest Wi-Fi password in the Austin office?"]:
        res = ask(qa, q)
        assert not res["answerable"], q
        assert res["citations"] == []


def test_missing_referenced_document_is_called_out(qa):
    res = ask(qa, "What reporting deadlines does the Anti-Harassment Policy set?")
    assert not res["answerable"]
    assert "not part of the handbook" in res["answer"]
    assert res["citations"][0]["doc_id"] == "hr-grievance-policy"
    assert_citations_are_verbatim(qa, res)


def test_follow_up_uses_conversation(qa):
    history = [{"role": "user", "content": "What is the per diem for an L4 employee in New York?"},
               {"role": "assistant", "content": "$100 per day."}]
    res = ask(qa, "What about Seattle?", history=history)
    assert "per diem usd: 85" in res["citations"][0]["quote"]
    assert res["standalone_question"] != res["question"]


def test_lead_in_sentence_absorbs_its_table(qa):
    res = ask(qa, "What is the maximum hotel room rate per night in San Francisco?")
    assert "$275" in res["answer"]


def test_locate_quote_tolerates_formatting_but_not_invention():
    text = "Claims above $2,500 require approval from a Vice President in addition to your manager."
    end = text.index(" in addition")
    assert locate_quote("claims above $2,500 require approval from a  Vice President", text) == (0, end)
    assert locate_quote("“Claims above $2,500 require approval”", text)[0] == 0
    assert locate_quote("Claims above $5,000 are approved by the CFO", text) is None
    s, e = locate_quote("Claims above $2,500 require approval from a Vice-President in addition to your manager", text)
    assert text[s:e].startswith("Claims above")
