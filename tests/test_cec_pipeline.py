import pytest
import io
import json
import zipfile
from fastapi.testclient import TestClient
from app.main import app
from app.storage.collection_store import storage_manager
from app.retrieval.query_compiler import QueryCompiler
from app.retrieval.source_discriminator import SourceDiscriminator
from app.retrieval.coverage_engine import ContrastiveCoverageEngine
from app.config import settings
from app.ingestion.archive_processor import ArchiveProcessor
from app.retrieval.bm25 import bm25_scores
from scripts.run_evaluation import RAGMetricsEvaluator

client = TestClient(app)

def create_dummy_pdf_bytes(text: str = "Test document content page 1") -> bytes:
    import pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # Return basic pdf structure
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()

def create_sample_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("test_report_fy25.pdf", create_dummy_pdf_bytes("Gammon India FY 2025 financial disclosures"))
        z.writestr("test_report_fy26.pdf", create_dummy_pdf_bytes("Gammon India FY 2026 financial disclosures"))
    return buf.getvalue()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_compiler_deterministic():
    q = "Compare Gammon India FY 2025 and FY 2026 debt obligations"
    reqs = QueryCompiler.compile_deterministic(q)
    assert "Gammon India" in reqs["entities"]
    assert "FY 2025" in reqs["periods"] or "FY 2026" in reqs["periods"]
    assert reqs["is_comparative"] is True

def test_query_compiler_uses_bounded_token_budget():
    import asyncio

    class QueryProvider:
        api_key = "configured"

        async def generate_completion(self, system_prompt, user_prompt, **kwargs):
            assert kwargs["json_mode"] is True
            assert kwargs["max_tokens"] == 400
            return json.dumps({
                "entities": [],
                "periods": [],
                "is_comparative": False,
                "comparison_targets": [],
                "subquestions": [],
                "key_terms": [],
            })

    requirements = asyncio.run(
        QueryCompiler.compile_query("What is the capital requirement?", QueryProvider())
    )

    assert requirements["required_facets"] == ["What is the capital requirement?"]

def test_source_discrimination():
    docs = [
        {"filename": "Gammon India Annual Report FY25.pdf", "entities": ["Gammon India Limited"], "periods": ["FY 2025"]},
        {"filename": "Inox Wind Report FY26.pdf", "entities": ["Inox Wind Limited"], "periods": ["FY 2026"]},
    ]
    reqs = {"query": "Gammon India debt", "entities": ["gammon india"], "periods": ["fy 2025"], "keywords": ["debt"]}
    ranked = SourceDiscriminator.rank_candidate_documents(docs, reqs)
    assert ranked[0][0]["filename"] == "Gammon India Annual Report FY25.pdf"

def test_bm25_weights_documents_by_content_not_only_filename():
    documents = [
        {"filename": "document_a.pdf"},
        {"filename": "document_b.pdf"},
    ]
    requirements = {
        "query": "spectral calibration coefficient",
        "entities": [],
        "periods": [],
        "keywords": [],
    }

    ranked = SourceDiscriminator.rank_candidate_documents(
        documents,
        requirements,
        max_candidates=1,
        document_texts={
            "document_a.pdf": "General accounting policies and financial statements.",
            "document_b.pdf": "The spectral calibration coefficient is estimated from observed prices.",
        },
    )

    assert ranked[0][0]["filename"] == "document_b.pdf"

def test_bm25_prefers_rare_query_terms_in_relevant_chunk():
    scores = bm25_scores(
        "spectral calibration coefficient",
        [
            "Spectral calibration coefficient determined the fitted parameter.",
            "General market overview and annual financial report.",
        ],
    )

    assert scores[0] > scores[1]

def test_evaluator_reports_answer_f1_loss_and_claim_confusion_counts(monkeypatch):
    evaluator = RAGMetricsEvaluator(client=None)
    answer_detail = {
        "matched_ids": ["bp1", "bp2"],
        "missing_ids": ["bp3"],
        "extra_ids": ["ap3"],
    }
    monkeypatch.setattr(
        evaluator,
        "answer_precision_recall",
        lambda *_: (0.75, 2 / 3, answer_detail),
    )
    monkeypatch.setattr(
        evaluator,
        "context_precision_recall",
        lambda *_: (0.5, 0.25),
    )

    metrics = evaluator.evaluate_single(
        "query",
        "answer",
        "reference",
        ["chunk"],
        ["reference chunk"],
    )

    assert metrics["answer_f1"] == pytest.approx(2 * 0.75 * (2 / 3) / (0.75 + 2 / 3))
    assert metrics["answer_loss"] == pytest.approx(1 - metrics["answer_f1"])
    assert metrics["context_f1"] == pytest.approx(1 / 3)
    assert metrics["context_loss"] == pytest.approx(2 / 3)
    assert metrics["answer_confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": None}

def test_exact_refusal_matches_without_claim_judge():
    evaluator = RAGMetricsEvaluator(client=None)

    precision, recall, detail = evaluator.answer_precision_recall(
        "No information was retrieved for this question.",
        "No information was retrieved for this question.",
    )

    assert precision == 1.0
    assert recall == 1.0
    assert detail["matched_ids"] == ["bp1"]
    assert detail["extra_ids"] == []

def test_context_recall_is_bounded_by_one(monkeypatch):
    evaluator = RAGMetricsEvaluator(client=None)
    monkeypatch.setattr(
        evaluator,
        "_judge_chunk_relevance",
        lambda *_: [{"relevant": True}, {"relevant": True}],
    )

    precision, recall = evaluator.context_precision_recall("query", ["a", "b"], 1)

    assert precision == 1.0
    assert recall == 1.0

def test_coverage_engine_selection():
    chunks = [
        {"chunk_id": "c1", "document": "doc1.pdf", "page": 1, "text": "Debt obligations increased to 500 crore in 2025"},
        {"chunk_id": "c2", "document": "doc1.pdf", "page": 2, "text": "Auditor flagged material uncertainty regarding going concern"},
        {"chunk_id": "c3", "document": "doc2.pdf", "page": 1, "text": "Weather patterns in Northern hemisphere"},
    ]
    reqs = {
        "query": "What are the debt and auditor concerns?",
        "required_facets": ["debt obligations", "auditor concerns"],
        "periods": ["2025"],
        "is_comparative": False,
    }
    selected = ContrastiveCoverageEngine.select_evidence(chunks, reqs, max_chunks=2)
    assert len(selected) == 2
    selected_texts = [c["text"] for c in selected]
    assert any("Debt" in t for t in selected_texts)
    assert any("Auditor" in t for t in selected_texts)

def test_coverage_selection_rejects_chunks_below_relative_cutoff():
    chunks = [
        {"chunk_id": "relevant", "document": "doc.pdf", "page": 1,
         "text": "Debt obligations increased to 500 crore in 2025."},
        {"chunk_id": "unrelated", "document": "doc.pdf", "page": 2,
         "text": "The weather patterns changed across northern regions in 2025."},
    ]
    reqs = {
        "query": "What were debt obligations in 2025?",
        "required_facets": ["debt obligations amount"],
        "periods": ["2025"],
        "is_comparative": False,
    }

    selected = ContrastiveCoverageEngine.select_evidence(chunks, reqs)

    assert [chunk["chunk_id"] for chunk in selected] == ["relevant"]

def test_coverage_selection_stops_after_facets_are_covered():
    chunks = [
        {"chunk_id": "alpha", "document": "doc.pdf", "page": 1,
         "text": "Company Alpha reported revenue of 10 million dollars in FY25."},
        {"chunk_id": "beta", "document": "doc.pdf", "page": 2,
         "text": "Company Beta reported revenue of 25 million dollars in FY25."},
        {"chunk_id": "alpha-extra", "document": "doc.pdf", "page": 3,
         "text": "Company Alpha reported revenue of 12 million dollars in FY25."},
    ]
    reqs = {
        "query": "Compare Company Alpha revenue and Company Beta revenue in FY25",
        "required_facets": ["Company Alpha revenue", "Company Beta revenue"],
        "periods": ["FY25"],
        "is_comparative": True,
        "comparison_targets": ["Company Alpha", "Company Beta"],
    }

    selected = ContrastiveCoverageEngine.select_evidence(chunks, reqs)

    assert {chunk["chunk_id"] for chunk in selected} == {"alpha", "beta"}
    assert len(selected) == 2

def test_selection_limit_scales_with_facets_and_breadth(monkeypatch):
    monkeypatch.setattr(settings, "MAX_EVIDENCE_CHUNKS", 12)

    simple = {"query": "What was revenue?", "required_facets": ["What was revenue?"]}
    multifacet = {
        "query": "Explain the risk and capital impact during reporting periods",
        "required_facets": ["risk", "capital impact", "reporting periods"],
    }
    broad = {
        "query": "List all events across each reporting period",
        "required_facets": ["events", "reporting period"],
    }
    comparative = {
        "query": "How did exposure increase from 2024 to 2025?",
        "required_facets": ["2024 exposure", "2025 exposure"],
        "is_comparative": True,
    }

    assert ContrastiveCoverageEngine.selection_limit(simple, 12) == 3
    assert ContrastiveCoverageEngine.selection_limit(multifacet, 12) == 6
    assert ContrastiveCoverageEngine.selection_limit(broad, 12) == 8
    assert ContrastiveCoverageEngine.selection_limit(comparative, 12) == 8

def test_selection_reserves_chunks_for_uncovered_facets():
    chunks = [
        {"chunk_id": "alpha", "document": "doc.pdf", "page": 1,
         "text": "Company Alpha risk exposure increased with regulatory capital pressure in 2025."},
        {"chunk_id": "alpha-repeat", "document": "doc.pdf", "page": 2,
         "text": "Company Alpha risk exposure increased with regulatory capital pressure in 2025."},
        {"chunk_id": "beta", "document": "doc.pdf", "page": 3,
         "text": "Company Beta risk exposure remained stable in 2025."},
    ]
    requirements = {
        "query": "Compare Company Alpha and Company Beta risk exposure in 2025",
        "required_facets": ["Company Alpha risk exposure", "Company Beta risk exposure"],
        "periods": ["2025"],
    }

    selected = ContrastiveCoverageEngine.select_evidence(chunks, requirements, max_chunks=8)

    assert {chunk["chunk_id"] for chunk in selected} == {"alpha", "beta"}

def test_selection_stops_when_no_facets_can_be_covered(monkeypatch):
    monkeypatch.setattr(settings, "MIN_COVERAGE_THRESHOLD", 1.1)
    chunks = [
        {"chunk_id": str(index), "document": "doc.pdf", "page": index,
         "text": "Debt obligations reported in 2025 with capital risk analysis and financial exposure."}
        for index in range(1, 7)
    ]
    reqs = {
        "query": "Debt obligations reported in 2025 with capital risk analysis",
        "required_facets": ["debt obligations capital risk analysis"],
        "periods": ["2025"],
        "is_comparative": False,
    }

    selected = ContrastiveCoverageEngine.select_evidence(chunks, reqs, max_chunks=8)

    assert len(selected) == 2

def test_chunk_pages_splits_long_paragraphs_with_section_heading_and_overlap(monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_SIZE_CHARS", 750)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP_CHARS", 100)
    body = "The bank reports credit risk exposure for customers and financial institutions. " * 30
    pages = [{"page_number": 1, "text": f"Capital Planning\n\n{body}"}]

    chunks = ArchiveProcessor.chunk_pages(
        pages,
        "report.pdf",
        {"periods": ["FY 2025-26"]},
    )

    assert len(chunks) > 1
    assert all(chunk["text"].startswith("Capital Planning\n") for chunk in chunks)
    assert all(chunk["section_title"] == "Capital Planning" for chunk in chunks)
    assert all(chunk["report_periods"] == ["FY 2025-26"] for chunk in chunks)
    assert all(len(chunk["text"]) <= 750 + len("Capital Planning\n") for chunk in chunks)
    first_body = chunks[0]["text"].split("\n", 1)[1]
    second_body = chunks[1]["text"].split("\n", 1)[1]
    assert first_body[-50:] in second_body

def test_api_index_and_query_flow(tmp_path):
    # Index collection
    zip_bytes = create_sample_zip()
    files = {"file": ("test_archive.zip", zip_bytes, "application/zip")}
    data = {"collection_name": "pytest_test_collection"}

    resp = client.post("/index", data=data, files=files)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "success"
    assert res_data["total_documents"] == 2

    # Query collection
    q_resp = client.post("/query", json={
        "collection_name": "pytest_test_collection",
        "query": "What are the disclosures for Gammon India?"
    })
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert "answer" in q_data
    assert "chunks" in q_data
    assert "trace" in q_data
    assert len(q_data["trace"]) == 4

    # Verify trace fields strictly match: step_name, description, input, output
    for step in q_data["trace"]:
        assert set(step.keys()) == {"step_name", "description", "input", "output"}

def test_unknown_collection_404():
    resp = client.post("/query", json={
        "collection_name": "non_existent_collection_999",
        "query": "Test query"
    })
    assert resp.status_code == 404
