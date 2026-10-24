import time
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import QueryRequest, QueryResponse, IndexResponse
from app.ingestion.archive_processor import ArchiveProcessor
from app.storage.collection_store import storage_manager
from app.retrieval.pipeline import retrieval_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smart_kb")

app = FastAPI(
    title="Smart Knowledge Base for Agents",
    description="Contrastive Evidence Coverage (CEC) grounded retrieval and QA engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Smart Knowledge Base for Agents",
        "version": "1.0.0",
        "available_collections": storage_manager.list_collections(),
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "collections_count": len(storage_manager.list_collections()),
    }

@app.post("/index", response_model=IndexResponse)
async def index_collection(
    file: UploadFile = File(..., description="ZIP archive containing documents to index"),
    collection_name: str = Form(..., description="Unique name for the isolated collection"),
):
    """
    POST /index
    Ingests an uploaded ZIP of documents, extracts text and discovered metadata,
    and indexes it under an isolated collection name.
    """
    start_time = time.time()
    if not collection_name or not collection_name.strip():
        raise HTTPException(status_code=400, detail="collection_name cannot be empty.")

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a ZIP archive (.zip).")

    try:
        zip_bytes = await file.read()
        documents, chunks, manifest, warnings = ArchiveProcessor.process_zip(zip_bytes)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"No valid documents could be extracted from archive. Warnings: {warnings}"
            )

        # Persist collection in isolated directory
        storage_manager.save_collection(
            collection_name=collection_name.strip(),
            manifest=manifest,
            documents=documents,
            chunks=chunks,
        )

        duration = time.time() - start_time
        logger.info(f"Successfully indexed collection '{collection_name}' with {len(documents)} docs in {duration:.2f}s")

        return IndexResponse(
            status="success",
            collection_name=collection_name.strip(),
            total_documents=len(documents),
            total_pages=manifest.get("total_pages", 0),
            total_chunks=len(chunks),
            warnings=warnings,
            duration_seconds=round(duration, 3),
        )

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(payload: QueryRequest):
    """
    POST /query
    Accepts collection_name and query.
    Optionally accepts benchmark_answer, benchmark_chunks, or evaluate=True.
    Executes Contrastive Evidence Coverage (CEC) pipeline and returns:
    - answer: grounded answer with inline citations
    - chunks: evidence chunks actually used
    - trace: structured steps with exact fields [step_name, description, input, output]
    - evaluation: (optional) LLM judge precision, recall, F1, and loss
    """
    if not payload.collection_name or not payload.collection_name.strip():
        raise HTTPException(status_code=400, detail="collection_name is required.")

    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty.")

    try:
        response = await retrieval_pipeline.run_query(
            collection_name=payload.collection_name.strip(),
            query=payload.query.strip(),
        )

        # If evaluation requested or benchmark_answer provided, run the official LLM judge
        if payload.evaluate or payload.benchmark_answer is not None:
            try:
                from scripts.run_evaluation import OpenCodeClient, RAGMetricsEvaluator
                import os

                judge_key = settings.LLM_API_KEY or os.getenv("LLM_API_KEY", "")
                judge_url = settings.LLM_BASE_URL or os.getenv("LLM_BASE_URL", "")
                judge_model = settings.LLM_MODEL or os.getenv("LLM_MODEL", "")

                client = OpenCodeClient(api_key=judge_key, model=judge_model, base_url=judge_url)
                # The public response is contract-limited to answer, chunks,
                # and trace. Keep optional local evaluation out of that
                # response and do not alter its official scoring function.
                evaluator = RAGMetricsEvaluator(client)
                evaluator.evaluate_single(
                    query=payload.query.strip(),
                    ai_answer=response.answer,
                    benchmark_answer=payload.benchmark_answer or "",
                    retrieved_chunks=[chunk.content for chunk in response.chunks],
                    benchmark_chunks=payload.benchmark_chunks or ["dummy_placeholder"],
                )
            except Exception as eval_err:
                logger.warning(f"Inline LLM judge evaluation skipped/failed: {eval_err}")


        return response
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail="unknown collection")
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

