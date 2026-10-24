import os
import json
import time
import requests
from typing import List, Dict, Any
from app.generation.answer_generator import NO_INFORMATION_RESPONSE

# =========================================================================
# The Official Evaluation Logic (Extracted directly from Evaluation.ipynb)
# DO NOT ALTER THE EVALUATION FORMULA OR PROMPTS
# =========================================================================

class OpenCodeClient:
    """
    Client for calling LLM judge with structured JSON output.
    """
    def __init__(self, api_key: str, model: str = "deepseek-v4.1-flash", base_url: str = "https://opencode.ai/inference/openai/v1/chat/completions"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def call_structured(self, system_prompt: str, user_prompt: str, max_tokens: int = 10000,
                         temperature: float = 0, max_retries: int = 3) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(max_retries):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError("Empty content returned from judge")
                return self._safe_json_parse(content)
            except Exception as e:
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenCode call failed after {max_retries} attempts.")

    @staticmethod
    def _safe_json_parse(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        return json.loads(text)

class RAGMetricsEvaluator:
    """
    Official evaluator class from Section 11.2 of the problem statement.
    Computes Answer Precision/Recall and Context Precision/Recall.
    """
    def __init__(self, client: OpenCodeClient):
        self.client = client

    def _decompose_points(self, text: str) -> list:
        system = "You are a precise fact-extraction assistant. Respond only with JSON."
        user = (
            "Break the following answer into a list of discrete, atomic factual points. "
            "Each point should be a single, independently checkable claim. If the answer "
            "is empty or contains no checkable claims, return an empty list.\n\n"
            f"Answer:\n{text}\n\n"
            'Respond as JSON exactly in this shape: {"points": ["point 1", "point 2", ...]}'
        )
        result = self.client.call_structured(system, user)
        return result.get("points", []) or []

    def _compare_answer_points(self, ai_points: list, benchmark_points: list) -> dict:
        system = (
            "You are a strict evaluator comparing two labeled sets of factual points: "
            "a benchmark (reference) answer's points and an AI-generated response's points. "
            "Respond only with JSON."
        )
        labeled_bp = {f"bp{i + 1}": p for i, p in enumerate(benchmark_points)}
        labeled_ap = {f"ap{i + 1}": p for i, p in enumerate(ai_points)}

        user = (
            "Benchmark points:\n"
            f"{json.dumps(labeled_bp, indent=2)}\n\n"
            "Agent response points:\n"
            f"{json.dumps(labeled_ap, indent=2)}\n\n"
            "Compare them and identify, using the ids shown above:\n"
            "1. matched_ids: ids of benchmark points (bpX) that are also present/covered in the agent response\n"
            "2. missing_ids: ids of benchmark points (bpX) that are NOT present in the agent response\n"
            "3. extra_ids: ids of agent response points (apX) that are NOT supported by any benchmark point\n\n"
            "Every bpX id must appear in exactly one of matched_ids or missing_ids. "
            "Every apX id that has no matching benchmark point should appear in extra_ids.\n\n"
            'Respond as JSON exactly in this shape: '
            '{"matched_ids": ["bp1", "bp3"], "missing_ids": ["bp2"], "extra_ids": ["ap4"]}'
        )
        result = self.client.call_structured(system, user)
        return {
            "matched_ids": result.get("matched_ids", []) or [],
            "missing_ids": result.get("missing_ids", []) or [],
            "extra_ids": result.get("extra_ids", []) or [],
            "benchmark_points": labeled_bp,
            "agent_points": labeled_ap,
        }

    def answer_precision_recall(self, ai_answer: str, benchmark_answer: str) -> tuple:
        if (
            ai_answer.strip() == NO_INFORMATION_RESPONSE
            and benchmark_answer.strip() == NO_INFORMATION_RESPONSE
        ):
            return 1.0, 1.0, {
                "matched_ids": ["bp1"],
                "missing_ids": [],
                "extra_ids": [],
                "benchmark_points": {"bp1": NO_INFORMATION_RESPONSE},
                "agent_points": {"ap1": NO_INFORMATION_RESPONSE},
            }

        ai_points = self._decompose_points(ai_answer)
        benchmark_points = self._decompose_points(benchmark_answer)

        if not ai_points and not benchmark_points:
            return 0.0, 0.0, {}
        if not ai_points:
            return 0.0, 0.0, {"missing_ids": [f"bp{i+1}" for i in range(len(benchmark_points))]}

        detail = self._compare_answer_points(ai_points, benchmark_points)
        matched_bp = {i for i in detail["matched_ids"] if i in detail["benchmark_points"]}
        extra_ap = {i for i in detail["extra_ids"] if i in detail["agent_points"]}

        precision = (len(ai_points) - len(extra_ap)) / len(ai_points) if ai_points else 0.0
        recall = len(matched_bp) / len(benchmark_points) if benchmark_points else 0.0
        return precision, recall, detail

    def _judge_chunk_relevance(self, query: str, chunks: list) -> list:
        system = "You are a strict evaluator judging whether retrieved text chunks are relevant to a user's query. Respond only with JSON."
        numbered = [{"id": i, "text": c} for i, c in enumerate(chunks)]
        user = (
            f"User query:\n{query}\n\n"
            "Retrieved chunks:\n"
            f"{json.dumps(numbered, indent=2)}\n\n"
            "For each chunk, decide if it is genuinely relevant and useful for answering the query.\n"
            'Respond as JSON exactly in this shape: '
            '{"verdicts": [{"id": 0, "relevant": true}, {"id": 1, "relevant": false}]}'
        )
        result = self.client.call_structured(system, user)
        return result.get("verdicts", []) or []

    def context_precision_recall(self, query: str, retrieved_chunks: list, benchmark_chunk_count: int) -> tuple:
        if not retrieved_chunks:
            return 0.0, 0.0
        verdicts = self._judge_chunk_relevance(query, retrieved_chunks)
        relevant_count = sum(1 for v in verdicts if v.get("relevant") is True)
        precision = relevant_count / len(retrieved_chunks)
        recall = min(1.0, relevant_count / benchmark_chunk_count) if benchmark_chunk_count else 0.0
        return precision, recall

    def evaluate_single(self, query: str, ai_answer: str, benchmark_answer: str,
                         retrieved_chunks: list, benchmark_chunks: list) -> dict:
        ans_p, ans_r, ans_detail = self.answer_precision_recall(ai_answer, benchmark_answer)
        ctx_p, ctx_r = self.context_precision_recall(query, retrieved_chunks, len(benchmark_chunks))
        answer_f1 = self.f1_score(ans_p, ans_r)
        context_f1 = self.f1_score(ctx_p, ctx_r)
        return {
            "answer_precision": ans_p,
            "answer_recall": ans_r,
            "answer_f1": answer_f1,
            "answer_loss": 1.0 - answer_f1,
            "context_precision": ctx_p,
            "context_recall": ctx_r,
            "context_f1": context_f1,
            "context_loss": 1.0 - context_f1,
            "answer_confusion": {
                "tp": len(ans_detail.get("matched_ids", [])),
                "fp": len(ans_detail.get("extra_ids", [])),
                "fn": len(ans_detail.get("missing_ids", [])),
                "tn": None,
            },
            "answer_detail": ans_detail,
        }

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        if precision <= 0.0 or recall <= 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

def run_evaluation(
    benchmark_path: str = "data/Train/Benchmarks/benchmark_100.jsonl",
    api_url: str = "http://127.0.0.1:8000/query",
    collection_name: str = "training_corpus",
    sample_size: int = 5,
    judge_api_key: str = "",
    judge_base_url: str = "https://api.groq.com/openai/v1/chat/completions",
    judge_model: str = "llama-3.3-70b-versatile",
):
    print(f"Loading benchmark from {benchmark_path}...")
    records = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    eval_subset = records[:sample_size]
    print(f"Running evaluation on {len(eval_subset)} benchmark questions against collection '{collection_name}'...")

    # Initialize evaluator
    client = OpenCodeClient(api_key=judge_api_key, model=judge_model, base_url=judge_base_url)
    evaluator = RAGMetricsEvaluator(client)

    results = []
    for idx, item in enumerate(eval_subset):
        q = item.get("question")
        gold_ans = item.get("benchmark_answer")
        chunks_dict = item.get("benchmark_chunks_w_metadata", {})
        # Flatten gold benchmark chunks
        gold_chunks = []
        for doc_k, doc_pages in chunks_dict.items():
            for p in doc_pages:
                if isinstance(p, dict) and "content" in p:
                    gold_chunks.append(p["content"])

        print(f"\n[{idx+1}/{len(eval_subset)}] Query: {q[:75]}...")

        # Query API
        try:
            resp = requests.post(api_url, json={"collection_name": collection_name, "query": q}, timeout=120)
            resp.raise_for_status()
            api_data = resp.json()
            ai_ans = api_data.get("answer", "")
            retrieved_chunks = [
                chunk.get("content", "") if isinstance(chunk, dict) else chunk
                for chunk in api_data.get("chunks", [])
            ]
            latency = api_data.get("latency_ms", 0)

            # Evaluate with official function
            metrics = evaluator.evaluate_single(
                query=q,
                ai_answer=ai_ans,
                benchmark_answer=gold_ans,
                retrieved_chunks=retrieved_chunks,
                benchmark_chunks=gold_chunks or ["dummy_placeholder"],
            )
            metrics["query"] = q
            metrics["latency_ms"] = latency
            results.append(metrics)

            print(f"  -> AnsP={metrics['answer_precision']:.2f} AnsR={metrics['answer_recall']:.2f} "
                  f"CtxP={metrics['context_precision']:.2f} CtxR={metrics['context_recall']:.2f} (Latency: {latency}ms)")
        except Exception as e:
            print(f"  -> Evaluation error on item {idx+1}: {e}")

    if results:
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
        micro_f1 = RAGMetricsEvaluator.f1_score(micro_precision, micro_recall)
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Questions Evaluated: {len(results)}")
        print(f"Average Answer Precision : {avg_ap:.4f}")
        print(f"Average Answer Recall    : {avg_ar:.4f}")
        print(f"Average Answer F1        : {avg_af1:.4f}")
        print(f"Average Answer Loss      : {avg_aloss:.4f}")
        print(f"Average Context Precision: {avg_cp:.4f}")
        print(f"Average Context Recall   : {avg_cr:.4f}")
        print(f"Average Context F1       : {avg_cf1:.4f}")
        print(f"Average Context Loss     : {avg_closs:.4f}")
        print("Answer claim confusion counts:")
        print(f"  TP={claim_tp} FP={claim_fp} FN={claim_fn} "
              "TN=N/A (no defined negative-claim universe)")
        print(f"Micro claim precision/recall/F1/loss: {micro_precision:.4f}/"
              f"{micro_recall:.4f}/{micro_f1:.4f}/{1.0 - micro_f1:.4f}")
        print("=" * 50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/Train/Benchmarks/benchmark_100.jsonl")
    parser.add_argument("--api_url", default="http://127.0.0.1:8000/query")
    parser.add_argument("--collection", default="training_corpus")
    parser.add_argument("--sample_size", type=int, default=3)
    parser.add_argument("--judge_key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--judge_url", default="https://api.groq.com/openai/v1/chat/completions")
    parser.add_argument("--judge_model", default="llama-3.3-70b-versatile")
    args = parser.parse_args()

    run_evaluation(
        benchmark_path=args.benchmark,
        api_url=args.api_url,
        collection_name=args.collection,
        sample_size=args.sample_size,
        judge_api_key=args.judge_key,
        judge_base_url=args.judge_url,
        judge_model=args.judge_model,
    )
