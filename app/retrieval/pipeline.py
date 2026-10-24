import time
import logging
from typing import Dict, Any, List
from app.config import settings
from app.schemas import EvidenceChunk, EvidenceChunkMetadata, TraceStep, QueryResponse
from app.storage.collection_store import storage_manager
from app.retrieval.query_compiler import QueryCompiler
from app.retrieval.source_discriminator import SourceDiscriminator
from app.retrieval.coverage_engine import ContrastiveCoverageEngine
from app.generation.provider import llm_provider
from app.generation.answer_generator import GroundedAnswerGenerator
from app.observability.langfuse_tracker import observability

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    """
    Coordinates the full Contrastive Evidence Coverage (CEC) pipeline:
    1. Query Requirement Compilation (Stage C)
    2. Contrastive Source Discrimination (Stage B)
    3. Bounded Evidence Selection & Coverage (Stage D)
    4. Grounded Answer Generation (Stage E)
    5. Trace logging conforming strictly to {step_name, description, input, output}
    """

    @staticmethod
    async def run_query(collection_name: str, query: str) -> QueryResponse:
        start_time = time.time()
        trace_steps: List[TraceStep] = []

        # 0. Validate Collection Existence
        if not storage_manager.collection_exists(collection_name):
            raise FileNotFoundError(f"Collection '{collection_name}' not found. Please index documents first.")

        collection_data = storage_manager.load_collection(collection_name)
        documents = collection_data.get("documents", [])
        all_chunks = collection_data.get("chunks", [])

        # Langfuse trace
        lf_trace = observability.start_trace(
            name="query_execution",
            metadata={"collection": collection_name, "query": query}
        )

        # 1. Query Requirement Compilation
        s1_start = time.time()
        requirements = await QueryCompiler.compile_query(query, llm_provider)
        s1_lat = (time.time() - s1_start) * 1000.0
        step1 = TraceStep(
            step_name="query_compilation",
            description="Compiled user query into structured retrieval facets, entities, periods, and comparison constraints.",
            input=query,
            output=f"Entities: {requirements.get('entities')}, Periods: {requirements.get('periods')}, Comparative: {requirements.get('is_comparative')}, Facets: {len(requirements.get('required_facets', []))}"
        )
        trace_steps.append(step1)
        if lf_trace:
            lf_trace.log_step(step1.step_name, step1.description, step1.input, step1.output, latency_ms=s1_lat)

        # 2. Contrastive Source Discrimination
        s2_start = time.time()
        document_texts: Dict[str, List[str]] = {}
        for chunk in all_chunks:
            document_texts.setdefault(chunk.get("document", ""), []).append(chunk.get("text", ""))
        ranked_docs = SourceDiscriminator.rank_candidate_documents(
            documents=documents,
            requirements=requirements,
            # Broad/synthesis questions often span documents. Keep a wider
            # soft pool, then let facet-aware evidence selection enforce the
            # relevance gate rather than excluding a source prematurely.
            max_candidates=(
                settings.MAX_BROAD_DOC_CANDIDATES
                if requirements.get("synthesis", False)
                else settings.MAX_DOC_CANDIDATES
            ),
            document_texts={
                filename: "\n".join(texts)
                for filename, texts in document_texts.items()
            },
        )
        s2_lat = (time.time() - s2_start) * 1000.0
        candidate_filenames = {d[0]["filename"] for d in ranked_docs}
        step2 = TraceStep(
            step_name="source_discrimination",
            description="Discriminated and prioritized candidate documents using entity matching, period alignment, and document type signals.",
            input=f"Total documents: {len(documents)}; Requirements: {requirements.get('entities')}",
            output=f"Prioritized {len(candidate_filenames)} documents: {list(candidate_filenames)}"
        )
        trace_steps.append(step2)
        if lf_trace:
            lf_trace.log_step(step2.step_name, step2.description, step2.input, step2.output, latency_ms=s2_lat)

        # 3. Evidence Coverage & Bounded Selection
        s3_start = time.time()
        candidate_chunks = [c for c in all_chunks if c.get("document") in candidate_filenames]
        if not candidate_chunks:
            # Fallback to all chunks if no candidate filtered
            candidate_chunks = all_chunks

        selected_chunks = ContrastiveCoverageEngine.select_evidence(
            candidate_chunks=candidate_chunks,
            requirements=requirements,
            max_chunks=settings.MAX_EVIDENCE_CHUNKS,
        )
        s3_lat = (time.time() - s3_start) * 1000.0
        chunk_provenances = [f"{c.get('document')} p.{c.get('page')}" for c in selected_chunks]
        step3 = TraceStep(
            step_name="evidence_coverage",
            description="Selected bounded evidence chunks maximizing marginal requirement coverage while eliminating redundancy.",
            input=f"Evaluated {len(candidate_chunks)} candidate chunks against {len(requirements.get('required_facets', []))} facets.",
            output=f"Selected {len(selected_chunks)} chunks: {chunk_provenances}"
        )
        trace_steps.append(step3)
        if lf_trace:
            lf_trace.log_step(step3.step_name, step3.description, step3.input, step3.output, latency_ms=s3_lat)

        # 4. Grounded Answer Generation
        s4_start = time.time()
        grounded_res = await GroundedAnswerGenerator.generate_grounded_answer(
            query=query,
            chunks=selected_chunks,
            llm_provider=llm_provider,
            synthesis=requirements.get("synthesis", False),
        )
        answer = grounded_res.answer
        usage_dict = grounded_res.usage
        s4_lat = (time.time() - s4_start) * 1000.0

        # Calculate estimated cost if provider returned usage details
        # deepseek-v4.1-flash rates: ~$0.14/1M prompt tokens, ~$0.28/1M completion tokens
        calculated_cost = None
        if usage_dict:
            p_tokens = usage_dict.get("prompt_tokens", 0)
            c_tokens = usage_dict.get("completion_tokens", 0)
            calculated_cost = round((p_tokens * 0.00000014) + (c_tokens * 0.00000028), 6)

        step4 = TraceStep(
            step_name="answer_generation",
            description="Generated grounded answer with explicit document and page citations, adhering to source evidence.",
            input=f"Provided {len(selected_chunks)} evidence chunks to LLM generator.",
            output=f"Generated {len(answer.split())} words with {len(grounded_res.citations)} source citations."
        )
        trace_steps.append(step4)
        if lf_trace:
            lf_trace.log_step(
                step4.step_name,
                step4.description,
                step4.input,
                step4.output,
                latency_ms=s4_lat,
                model=settings.LLM_MODEL,
                usage=usage_dict,
                cost=calculated_cost,
            )

        total_latency = (time.time() - start_time) * 1000.0

        if lf_trace:
            try:
                lf_meta = {
                    "latency_ms": total_latency,
                    "chunks_used": len(selected_chunks),
                }
                if usage_dict:
                    lf_meta["usage"] = usage_dict
                if calculated_cost is not None:
                    lf_meta["cost_usd"] = calculated_cost
                lf_trace.update(
                    output=answer,
                    metadata=lf_meta,
                )
            except Exception:
                pass
        observability.flush()

        # Serialize exactly the evidence presented to the answer generator.
        # Legacy collections lack the explicit fields, so retain their stored
        # document/page values without guessing or changing numbering.
        no_evidence_answer = (
            answer == "No information was retrieved for this question."
            or answer.startswith("Insufficient evidence available")
        )
        response_chunks = []
        if not no_evidence_answer:
            for chunk in selected_chunks:
                raw_pages = chunk.get("page_numbers") or [chunk.get("page")]
                pages = [int(page) for page in raw_pages if isinstance(page, int) and page >= 1]
                if not pages:
                    continue
                response_chunks.append(EvidenceChunk(
                    content=chunk.get("text", ""),
                    metadata=EvidenceChunkMetadata(
                        file_name=chunk.get("file_name") or chunk.get("document", ""),
                        page_numbers=list(dict.fromkeys(pages)),
                    ),
                ))

        response = QueryResponse(
            answer=answer,
            chunks=response_chunks,
            trace=trace_steps,
        )
        response._latency_ms = round(total_latency, 2)
        return response

retrieval_pipeline = RetrievalPipeline()
