from datetime import date

from app.config import ROOT
from app.ingest.pipeline import (
    build_document, chunk_document, compute_lifecycles, family_key, load_corpus, quality_report,
)

CORPUS = ROOT / "data" / "corpus"


def _doc(name):
    return build_document(CORPUS / name, None)


def test_every_manifest_document_is_ingested_with_text():
    docs, report = load_corpus(CORPUS)
    assert len(docs) == 64
    assert not [r for r in report if r.startswith("[missing file]")]
    for d in docs:
        assert d.blocks, f"{d.doc_id} produced no text"
        for b in d.blocks:
            assert d.text[b.start:b.end] == b.text


def test_markdown_front_matter_is_metadata_not_content():
    d = _doc("hr-leave-policy-v1.md")
    assert d.meta.doc_id == "hr-leave-policy-v1"
    assert "doc_id:" not in d.text
    assert "**" not in d.text                    # inline markdown stripped
    assert "12 days of annual leave" in d.text


def test_pdf_header_line_is_parsed_including_wrapped_supersedes():
    d = _doc("hr-leave-policy-v2.pdf")
    assert d.meta.supersedes == "hr-leave-policy-v1"
    assert d.meta.effective_date == date(2025, 4, 1)
    headings = [b.text for b in d.blocks if b.kind == "heading"]
    assert "2. Annual leave" in headings


def test_pdf_numbered_list_is_rebuilt():
    d = _doc("eng-deploy-runbook-2023.pdf")
    items = [b.text for b in d.blocks if b.kind == "list_item"]
    assert "The change has one approving review and a green CI pipeline." in items
    assert d.meta.status == "deprecated"


def test_docx_table_stays_in_its_section():
    """The hotel caps table sits under 4.2 in the Word file; it must not drift to the end."""
    d = _doc("hr-travel-policy.docx")
    chunks = chunk_document(d)
    row = next(c for c in chunks if "Tier 1 | Maximum room rate per night: $275" in c.text)
    assert row.section.endswith("4.2 Hotels and daily spending limits")
    assert d.text.index("$275") < d.text.index("4.3 Ground transport")


def test_csv_rows_become_self_describing_chunks():
    d = _doc("fin-per-diem-table.csv")
    chunks = chunk_document(d)
    assert len(chunks) == 15
    assert any("grade: L3 | city tier: 2" in c.text and "per diem usd: 75" in c.text for c in chunks)


def test_html_is_parsed_without_markup():
    d = _doc("support-escalation.html")
    assert "<p>" not in d.text
    assert "Head of Product" in d.text


def test_glossary_items_are_separate_chunks():
    chunks = chunk_document(_doc("ref-glossary.md"))
    assert any(c.text.startswith("RTO: Recovery Time Objective") for c in chunks)
    assert len(chunks) >= 10


def test_lifecycle_supersession_deprecation_and_future_dates():
    docs, _ = load_corpus(CORPUS)
    metas = [d.meta for d in docs]
    now = compute_lifecycles(metas, date(2026, 11, 14))
    assert now["hr-leave-policy-v1"].state == "superseded"
    assert now["hr-leave-policy-v1"].superseded_by == "hr-leave-policy-v2"
    assert now["hr-leave-policy-v2"].state == "in_force"
    assert now["eng-deploy-runbook-2023"].state == "deprecated"
    assert now["hr-parental-leave-policy-2027"].state == "not_yet_effective"
    later = compute_lifecycles(metas, date(2027, 2, 1))
    assert later["hr-parental-leave-policy-2027"].state == "in_force"


def test_supersession_waits_for_the_new_version_to_take_effect():
    docs, _ = load_corpus(CORPUS)
    lcs = compute_lifecycles([d.meta for d in docs], date(2024, 6, 1))   # before v2's effective date
    assert lcs["hr-leave-policy-v1"].state == "in_force"
    assert lcs["hr-leave-policy-v2"].state == "not_yet_effective"


def test_family_key_groups_versions():
    assert family_key("Leave Policy v1") == family_key("Leave Policy v2")
    assert family_key("Production Deploy Runbook (2023)") == family_key("Production Deploy Runbook")
    assert family_key("Parental Leave Policy (2027 revision)") == family_key("Parental Leave Policy")


def test_quality_report_flags_the_known_data_problems():
    docs, extra = load_corpus(CORPUS)
    lcs = compute_lifecycles([d.meta for d in docs], date(2026, 11, 14))
    report = "\n".join(quality_report(docs, lcs, extra))
    assert "Holiday Calendar" in report
    assert "Anti-Harassment Policy" in report
    assert "broken link to 'hr-travel-policy.md'" in report
    assert "manifest marks it 'current' but hr-leave-policy-v2 supersedes it" in report
    assert "BYOD Policy" not in report      # resolves to "Bring Your Own Device Policy"
