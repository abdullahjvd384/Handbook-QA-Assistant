import pytest
from fastapi.testclient import TestClient

from app.api import create_app


@pytest.fixture(scope="module")
def client(settings, kb, qa):
    with TestClient(create_app(settings, kb=kb, qa=qa)) as c:
        yield c


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["documents"] >= 64 and h["answer_mode"] == "extractive"


def test_ask_returns_answer_with_clickable_citations(client):
    r = client.post("/api/ask", json={"question": "How many days of annual leave do I get?", "as_of": "2026-11-14"})
    assert r.status_code == 200
    body = r.json()
    c = body["citations"][0]
    assert c["doc_id"] == "hr-leave-policy-v2"
    assert c["url"] == f"/?doc={c['doc_id']}&start={c['start']}&end={c['end']}"
    doc = client.get(f"/api/documents/{c['doc_id']}").json()
    assert doc["text"][c["start"]:c["end"]] == c["quote"]


def test_ask_validation(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 3000}).status_code == 422
    assert client.post("/api/ask", json={"question": "hi", "mode": "llm"}).status_code == 400   # no API key


def test_document_endpoints(client):
    docs = client.get("/api/documents").json()
    v1 = next(d for d in docs if d["doc_id"] == "hr-leave-policy-v1")
    assert v1["declared_status"] == "current" and v1["lifecycle"] == "superseded"
    d = client.get("/api/documents/hr-travel-policy").json()
    assert d["format"] == "docx" and d["blocks"]
    assert client.get("/api/documents/does-not-exist").status_code == 404
    assert client.get("/api/documents/..%2F..%2Fapp%2Fconfig.py").status_code == 404
    f = client.get("/api/documents/hr-travel-policy/file")
    assert f.status_code == 200 and f.content[:2] == b"PK"


def test_report(client):
    findings = client.get("/api/report").json()["findings"]
    assert any("Anti-Harassment Policy" in f for f in findings)


def test_upload_is_searchable_immediately(client):
    content = (b"# Bike to Work Scheme\n\n## Allowance\nThe company contributes up to $750 towards a bicycle "
               b"and safety gear, once every three years.\n")
    r = client.post("/api/documents", files={"file": ("bike-scheme.md", content, "text/markdown")},
                    data={"category": "HR"})
    assert r.status_code == 201, r.text
    doc_id = r.json()["doc_id"]
    assert r.json()["chunks"] >= 1
    ans = client.post("/api/ask", json={"question": "How much does the company contribute towards a bicycle?"}).json()
    assert ans["citations"][0]["doc_id"] == doc_id
    assert "$750" in ans["answer"]
    # same title again gets a distinct id rather than overwriting
    r2 = client.post("/api/documents", files={"file": ("bike-scheme.md", content, "text/markdown")})
    assert r2.status_code == 201 and r2.json()["doc_id"] != doc_id


def test_upload_rejects_bad_files(client):
    assert client.post("/api/documents", files={"file": ("x.exe", b"MZ...", "application/octet-stream")}).status_code == 415
    assert client.post("/api/documents", files={"file": ("empty.md", b"   ", "text/markdown")}).status_code == 400
    r = client.post("/api/documents", files={"file": ("a.md", b"# A\n\ntext", "text/markdown")},
                    data={"supersedes": "nope"})
    assert r.status_code == 400


def test_upload_superseding_document_changes_lifecycle(client):
    content = b"# Corporate Card Policy v2\n\nThe monthly limit is $15,000.\n"
    r = client.post("/api/documents", files={"file": ("card-v2.md", content, "text/markdown")},
                    data={"supersedes": "fin-corporate-card", "effective_date": "2026-01-01"})
    assert r.status_code == 201
    old = client.get("/api/documents/fin-corporate-card").json()
    assert old["lifecycle"] == "superseded"
    ans = client.post("/api/ask", json={"question": "What is the monthly limit on a corporate card?"}).json()
    assert "15,000" in ans["citations"][0]["quote"]


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Handbook Assistant" in r.text
    assert client.get("/static/app.js").status_code == 200
