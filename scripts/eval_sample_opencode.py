import asyncio
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv

load_dotenv(".env", override=True)
from app.retrieval.pipeline import retrieval_pipeline
from scripts.run_evaluation import OpenCodeClient, RAGMetricsEvaluator

test_qids = ["B32", "C31", "A35", "A34", "A33"]

async def run_batch():
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    client = OpenCodeClient(api_key=api_key, model=model, base_url=base_url)
    evaluator = RAGMetricsEvaluator(client)

    records = {}
    with open("data/Train/Benchmarks/benchmark_100.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in test_qids:
                records[r["id"]] = r

    results = []
    print(f"Starting official evaluation over {len(test_qids)} benchmark questions...")
    for qid in test_qids:
        item = records[qid]
        q = item["question"]
        gold_ans = item["benchmark_answer"]
        gold_chunks = []
        for doc_k, doc_pages in item.get("benchmark_chunks_w_metadata", {}).items():
            for p in doc_pages:
                if isinstance(p, dict) and "content" in p:
                    gold_chunks.append(p["content"])

        res = await retrieval_pipeline.run_query("training_full_corpus", q)
        metrics = evaluator.evaluate_single(
            query=q,
            ai_answer=res.answer,
            benchmark_answer=gold_ans,
            retrieved_chunks=[chunk.content for chunk in res.chunks],
            benchmark_chunks=gold_chunks or ["dummy"],
        )
        # Calculate approximate cost based on OpenCode / DeepSeek model rates
        # deepseek-v4.1-flash: ~$0.14/1M input tokens, ~$0.28/1M output tokens
        tokens_est = len(q.split()) + len(res.answer.split()) + sum(len(c.split()) for c in res.chunks)
        cost_est = (tokens_est / 1000.0) * 0.0002

        metrics["id"] = qid
        metrics["latency_ms"] = res.latency_ms
        metrics["tokens_est"] = tokens_est
        metrics["cost_usd_est"] = round(cost_est, 6)
        results.append(metrics)
        print(f"[{qid}] AnsP={metrics['answer_precision']:.2f} AnsR={metrics['answer_recall']:.2f} CtxP={metrics['context_precision']:.2f} CtxR={metrics['context_recall']:.2f} Latency={res.latency_ms:.0f}ms Cost=${cost_est:.6f}")

    avg_ap = sum(r["answer_precision"] for r in results) / len(results)
    avg_ar = sum(r["answer_recall"] for r in results) / len(results)
    avg_af1 = sum(r["answer_f1"] for r in results) / len(results)
    avg_aloss = sum(r["answer_loss"] for r in results) / len(results)
    avg_cp = sum(r["context_precision"] for r in results) / len(results)
    avg_cr = sum(r["context_recall"] for r in results) / len(results)
    avg_cf1 = sum(r["context_f1"] for r in results) / len(results)
    avg_closs = sum(r["context_loss"] for r in results) / len(results)
    claim_tp = sum(r["answer_confusion"]["tp"] for r in results)
    claim_fp = sum(r["answer_confusion"]["fp"] for r in results)
    claim_fn = sum(r["answer_confusion"]["fn"] for r in results)
    micro_precision = claim_tp / (claim_tp + claim_fp) if claim_tp + claim_fp else 0.0
    micro_recall = claim_tp / (claim_tp + claim_fn) if claim_tp + claim_fn else 0.0
    micro_f1 = evaluator.f1_score(micro_precision, micro_recall)
    avg_lat = sum(r["latency_ms"] for r in results) / len(results)
    total_cost = sum(r["cost_usd_est"] for r in results)

    print("\n" + "=" * 45)
    print("OFFICIAL 5-QUESTION BENCHMARK SUMMARY (OPENCODE):")
    print("=" * 45)
    print(f"Average Answer Precision : {avg_ap:.4f}")
    print(f"Average Answer Recall    : {avg_ar:.4f}")
    print(f"Average Answer F1        : {avg_af1:.4f}")
    print(f"Average Answer Loss      : {avg_aloss:.4f}")
    print(f"Average Context Precision: {avg_cp:.4f}")
    print(f"Average Context Recall   : {avg_cr:.4f}")
    print(f"Average Context F1       : {avg_cf1:.4f}")
    print(f"Average Context Loss     : {avg_closs:.4f}")
    print(f"Answer claim counts      : TP={claim_tp} FP={claim_fp} FN={claim_fn} TN=N/A")
    print(f"Micro claim P/R/F1/loss  : {micro_precision:.4f}/{micro_recall:.4f}/{micro_f1:.4f}/{1.0 - micro_f1:.4f}")
    print(f"Average Latency (ms)     : {avg_lat:.1f} ms")
    print(f"Total Est. Cost (USD)    : ${total_cost:.6f}")
    print("=" * 45)

    with open("docs/evaluation_results_sample5_current.json", "w", encoding="utf-8") as out_f:
        json.dump({
            "summary": {
                "avg_answer_precision": avg_ap,
                "avg_answer_recall": avg_ar,
                "avg_answer_f1": avg_af1,
                "avg_answer_loss": avg_aloss,
                "avg_context_precision": avg_cp,
                "avg_context_recall": avg_cr,
                "avg_context_f1": avg_cf1,
                "avg_context_loss": avg_closs,
                "answer_claim_counts": {
                    "tp": claim_tp,
                    "fp": claim_fp,
                    "fn": claim_fn,
                    "tn": None,
                },
                "micro_answer_precision": micro_precision,
                "micro_answer_recall": micro_recall,
                "micro_answer_f1": micro_f1,
                "micro_answer_loss": 1.0 - micro_f1,
                "avg_latency_ms": avg_lat,
                "total_cost_usd_est": total_cost,
            },
            "per_question": results
        }, out_f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_batch())
