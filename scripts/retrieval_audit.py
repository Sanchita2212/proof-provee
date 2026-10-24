import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from app.config import settings
from app.retrieval.coverage_engine import ContrastiveCoverageEngine
from app.retrieval.query_compiler import QueryCompiler
from app.retrieval.source_discriminator import SourceDiscriminator
from app.generation.provider import llm_provider
from app.storage.collection_store import storage_manager


PagePair = Tuple[str, str]


def _page_pairs(document: str, page_record: Dict[str, Any]) -> Set[PagePair]:
    metadata = page_record.get("metadata") or {}
    pages = metadata.get("page_numbers") or []
    if not pages:
        section = str(page_record.get("section_name", ""))
        match = re.search(
            r"(?:^|_)p?(\d+)(?:_|$)",
            section,
            flags=re.IGNORECASE,
        )
        pages = [match.group(1)] if match else []
    return {(document, str(page)) for page in pages if page is not None}


async def audit(
    benchmark_path: Path,
    prior_results_path: Path,
    collection_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    if not storage_manager.collection_exists(collection_name):
        raise FileNotFoundError(f"Collection '{collection_name}' not found.")

    benchmark_by_id: Dict[str, Dict[str, Any]] = {}
    for line in benchmark_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            benchmark_by_id[item["id"]] = item

    prior_results = json.loads(prior_results_path.read_text(encoding="utf-8"))
    collection = storage_manager.load_collection(collection_name)
    documents = collection.get("documents", [])
    all_chunks = collection.get("chunks", [])
    document_texts: Dict[str, List[str]] = {}
    for chunk in all_chunks:
        document_texts.setdefault(chunk.get("document", ""), []).append(chunk.get("text", ""))

    rows: List[Dict[str, Any]] = []
    for prior in prior_results.get("per_question", []):
        question_id = prior["id"]
        item = benchmark_by_id[question_id]
        query = item["question"]
        requirements = await QueryCompiler.compile_query(query, llm_provider)
        ranked_docs = SourceDiscriminator.rank_candidate_documents(
            documents=documents,
            requirements=requirements,
            max_candidates=settings.MAX_DOC_CANDIDATES,
            document_texts={
                filename: "\n".join(texts)
                for filename, texts in document_texts.items()
            },
        )
        candidate_names = {entry[0]["filename"] for entry in ranked_docs}
        candidates = [chunk for chunk in all_chunks if chunk.get("document") in candidate_names]
        if not candidates:
            candidates = all_chunks
        limit = ContrastiveCoverageEngine.selection_limit(
            requirements,
            settings.MAX_EVIDENCE_CHUNKS,
        )
        selected = ContrastiveCoverageEngine.select_evidence(
            candidates,
            requirements,
            max_chunks=settings.MAX_EVIDENCE_CHUNKS,
        )

        gold_pairs: Set[PagePair] = set()
        gold_chunk_count = 0
        for document, records in item.get("benchmark_chunks_w_metadata", {}).items():
            for record in records:
                if isinstance(record, dict):
                    gold_chunk_count += 1
                    gold_pairs.update(_page_pairs(document, record))

        retrieved_pairs = {
            (str(chunk.get("document", "")), str(chunk.get("page")))
            for chunk in selected
            if chunk.get("document") is not None and chunk.get("page") is not None
        }
        page_hits = gold_pairs & retrieved_pairs
        gold_claim_count = len(
            (prior.get("answer_detail") or {}).get("benchmark_points", {})
        )
        rows.append({
            "id": question_id,
            "question": query,
            "gold_claims": gold_claim_count,
            "gold_chunks": gold_chunk_count,
            "gold_document_pages": len(gold_pairs),
            "retrieval_budget": limit,
            "retrieved_chunks": len(selected),
            "retrieved_document_pages": sorted(
                [{"document": document, "page": page} for document, page in retrieved_pairs],
                key=lambda pair: (pair["document"], pair["page"]),
            ),
            "gold_pages_hit": len(page_hits),
            "gold_page_recall": len(page_hits) / len(gold_pairs) if gold_pairs else None,
            "answer_recall_previous_evaluation": prior.get("answer_recall"),
        })

    page_recalls = [
        row["gold_page_recall"]
        for row in rows
        if row["gold_page_recall"] is not None
    ]
    gold_pair_counts = [row["gold_document_pages"] for row in rows]
    result = {
        "per_question": rows,
        "summary": {
            "questions_audited": len(rows),
            "questions_with_gold_pages": len(page_recalls),
            "average_gold_page_recall": statistics.mean(page_recalls) if page_recalls else 0.0,
            "median_gold_document_pages": statistics.median(gold_pair_counts) if gold_pair_counts else 0,
            "questions_with_gold_pages_above_budget": sum(
                row["gold_document_pages"] > row["retrieval_budget"] for row in rows
            ),
            "global_evidence_cap": settings.MAX_EVIDENCE_CHUNKS,
            "document_candidate_limit": settings.MAX_DOC_CANDIDATES,
        },
        "method": {
            "gold_page_match": "Exact (benchmark document, page) pair intersection; no LLM judge.",
            "answer_recall_note": (
                "Carried forward from evaluation_results_sample20.json. "
                "The deployed training_full_corpus collection is reindexed and matches the 31,994-chunk "
                "segmentation measured in the evaluation run."
            ),
            "collection": collection_name,
            "benchmark": str(benchmark_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit retrieved document/page pairs against the selected Train benchmark questions."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/Train/Benchmarks/benchmark_100.jsonl"),
    )
    parser.add_argument(
        "--prior-results",
        type=Path,
        default=Path("docs/evaluation_results_sample20.json"),
    )
    parser.add_argument("--collection", default="training_full_corpus")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/retrieval_audit_sample20.json"),
    )
    args = parser.parse_args()
    result = asyncio.run(
        audit(args.benchmark, args.prior_results, args.collection, args.output)
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote per-question retrieval audit to {args.output}")


if __name__ == "__main__":
    main()
