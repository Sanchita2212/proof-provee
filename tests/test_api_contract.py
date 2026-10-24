import asyncio
import io
import zipfile

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import pipeline as pipeline_module
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas import EvidenceChunk, EvidenceChunkMetadata, QueryResponse, TraceStep
from app.storage.collection_store import storage_manager


client = TestClient(app)


def _zip_with_pdf() -> bytes:
    import pypdf

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf = io.BytesIO()
    writer.write(pdf)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Original File Name.pdf", pdf.getvalue())
    return archive.getvalue()


def _trace() -> list[TraceStep]:
    return [TraceStep(step_name="selection", description="test", input="q", output="one chunk")]


def test_index_is_synchronous_and_supports_multiple_collections_and_reindexing():
    archive = _zip_with_pdf()
    for collection_name in ("contract_collection_one", "contract_collection_two"):
        response = client.post(
            "/index",
            data={"collection_name": collection_name},
            files={"file": ("corpus.zip", archive, "application/zip")},
        )
        assert response.status_code == 200
        assert storage_manager.collection_exists(collection_name)

    reindex = client.post(
        "/index",
        data={"collection_name": "contract_collection_one"},
        files={"file": ("corpus.zip", archive, "application/zip")},
    )
    assert reindex.status_code == 200
    assert reindex.json()["collection_name"] == "contract_collection_one"


def test_query_serializes_exact_contract_shape_and_one_based_pages(monkeypatch):
    async def fake_query(*_args, **_kwargs):
        return QueryResponse(
            answer="The evidence supports the answer [Original File Name.pdf p.1].",
            chunks=[EvidenceChunk(
                content="Supporting evidence.",
                metadata=EvidenceChunkMetadata(
                    file_name="Original File Name.pdf",
                    page_numbers=[1],
                ),
            )],
            trace=_trace(),
        )

    monkeypatch.setattr("app.main.retrieval_pipeline.run_query", fake_query)
    response = client.post("/query", json={"collection_name": "any", "query": "test"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The evidence supports the answer [Original File Name.pdf p.1].",
        "chunks": [{
            "content": "Supporting evidence.",
            "metadata": {"file_name": "Original File Name.pdf", "page_numbers": [1]},
        }],
        "trace": [{"step_name": "selection", "description": "test", "input": "q", "output": "one chunk"}],
    }


def test_unknown_collection_uses_contract_error():
    response = client.post("/query", json={"collection_name": "missing_contract_collection", "query": "test"})

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown collection"}


def test_missing_evidence_returns_empty_chunks_with_valid_trace(monkeypatch):
    async def fake_query(*_args, **_kwargs):
        return QueryResponse(
            answer="Insufficient evidence available in the indexed collection to answer the query.",
            chunks=[],
            trace=_trace(),
        )

    monkeypatch.setattr("app.main.retrieval_pipeline.run_query", fake_query)
    response = client.post("/query", json={"collection_name": "any", "query": "unanswerable"})

    assert response.status_code == 200
    assert response.json()["chunks"] == []
    assert response.json()["trace"]
    assert set(response.json()) == {"answer", "chunks", "trace"}


def test_selected_chunk_preserves_ingested_filename_and_page(monkeypatch):
    collection = "contract_pipeline_provenance"
    storage_manager.save_collection(
        collection,
        manifest={"total_documents": 1, "total_pages": 1, "total_chunks": 1},
        documents=[{"filename": "Original File Name.pdf"}],
        chunks=[{
            "chunk_id": "source_p1_c0",
            "document": "Original File Name.pdf",
            "file_name": "Original File Name.pdf",
            "page": 1,
            "page_numbers": [1],
            "text": "Alpha capital ratio was 12 percent.",
        }],
    )

    class FakeProvider:
        api_key = ""

        async def generate_completion_with_usage(self, **_kwargs):
            return "Alpha capital ratio was 12 percent [Original File Name.pdf p.1].", None

    monkeypatch.setattr(pipeline_module, "llm_provider", FakeProvider())
    result = asyncio.run(RetrievalPipeline.run_query(collection, "What was the Alpha capital ratio?"))

    assert result.chunks[0].metadata.file_name == "Original File Name.pdf"
    assert result.chunks[0].metadata.page_numbers == [1]
