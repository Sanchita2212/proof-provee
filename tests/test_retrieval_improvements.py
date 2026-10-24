import asyncio

from app.generation.answer_generator import GroundedAnswerGenerator
from app.retrieval.coverage_engine import ContrastiveCoverageEngine
from app.retrieval.query_compiler import QueryCompiler


def test_deterministic_compiler_keeps_original_query_and_explicit_clauses():
    query = "List the first result; what was the second result; and what conditions apply?"

    requirements = QueryCompiler.compile_deterministic(query)

    assert requirements["required_facets"][0] == query
    assert requirements["required_facets"][1:] == [
        "List the first result",
        "what was the second result",
        "what conditions apply?",
    ]
    assert requirements["synthesis"] is True


def test_coverage_ignores_compiler_routing_facet_when_allocating_evidence():
    query = "What were Alpha revenue and Beta revenue?"
    chunks = [
        {"chunk_id": "alpha", "document": "report.pdf", "page": 1,
         "text": "Alpha revenue was 10 million dollars."},
        {"chunk_id": "beta", "document": "report.pdf", "page": 2,
         "text": "Beta revenue was 20 million dollars."},
    ]
    requirements = {
        "query": query,
        "required_facets": [query, "Alpha revenue", "Beta revenue"],
    }

    selected = ContrastiveCoverageEngine.select_evidence(chunks, requirements, max_chunks=3)

    assert {chunk["chunk_id"] for chunk in selected} == {"alpha", "beta"}


def test_cited_partial_answer_is_not_collapsed_to_refusal():
    class PartialProvider:
        async def generate_completion_with_usage(self, **kwargs):
            return (
                "Revenue was 100 crore [report.pdf p.4]. "
                "No information was retrieved for this specific question.",
                None,
            )

    answer, _ = asyncio.run(
        GroundedAnswerGenerator.generate_grounded_answer(
            query="What was revenue and the CEO's home address?",
            chunks=[{"document": "report.pdf", "page": 4, "text": "Revenue was 100 crore."}],
            llm_provider=PartialProvider(),
            synthesis=True,
        )
    )

    assert "Revenue was 100 crore [report.pdf p.4]." in answer
