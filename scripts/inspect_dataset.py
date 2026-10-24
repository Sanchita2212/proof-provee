import os
import sys
import json
import zipfile
import pypdf

def inspect_dataset(data_dir: str = "data/Train"):
    docs_dir = os.path.join(data_dir, "Documents")
    benchmarks_file = os.path.join(data_dir, "Benchmarks", "benchmark_100.jsonl")
    eval_file = os.path.join(data_dir, "Evaluation", "Evaluation.ipynb")

    print("=" * 60)
    print("DATASET INSPECTION & INTEGRITY REPORT")
    print("=" * 60)

    # 1. Document Collection
    if not os.path.exists(docs_dir):
        print(f"ERROR: Documents directory not found at {docs_dir}")
        return

    doc_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    print(f"Total PDF Documents: {len(doc_files)}")

    total_pages = 0
    total_size_bytes = 0
    corrupt_files = []
    encrypted_files = []
    empty_files = []

    sample_summary = []

    for idx, fname in enumerate(sorted(doc_files)):
        fpath = os.path.join(docs_dir, fname)
        size = os.path.getsize(fpath)
        total_size_bytes += size

        try:
            reader = pypdf.PdfReader(fpath)
            if reader.is_encrypted:
                encrypted_files.append(fname)
                continue
            pages_count = len(reader.pages)
            total_pages += pages_count

            # Extract small sample text from page 1
            sample_text = ""
            if pages_count > 0:
                sample_text = (reader.pages[0].extract_text() or "").strip()[:100]

            if pages_count == 0:
                empty_files.append(fname)
            elif idx < 5:
                sample_summary.append({
                    "filename": fname,
                    "pages": pages_count,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "page_1_excerpt": sample_text.replace("\n", " "),
                })
        except Exception as e:
            corrupt_files.append((fname, str(e)))

    print(f"Total Corpus Size: {total_size_bytes / (1024 * 1024):.2f} MB")
    print(f"Total Corpus Pages: {total_pages}")
    print(f"Encrypted PDFs: {len(encrypted_files)}")
    print(f"Corrupted PDFs: {len(corrupt_files)}")
    print(f"Empty PDFs: {len(empty_files)}")

    print("\nSample Documents:")
    for s in sample_summary:
        print(f" - {s['filename']} ({s['pages']} pages, {s['size_mb']} MB): \"{s['page_1_excerpt']}...\"")

    # 2. Benchmark Collection
    print("-" * 60)
    if os.path.exists(benchmarks_file):
        with open(benchmarks_file, "r", encoding="utf-8") as bf:
            benchmark_lines = bf.readlines()
        print(f"Total Benchmark Questions: {len(benchmark_lines)}")
        if benchmark_lines:
            sample_q = json.loads(benchmark_lines[0])
            print("Benchmark Sample:")
            print(f"  ID: {sample_q.get('id')}")
            print(f"  Question: {sample_q.get('question')[:120]}...")
            print(f"  Answer length: {len(sample_q.get('benchmark_answer', ''))} characters")
    else:
        print(f"WARNING: Benchmark file not found at {benchmarks_file}")

    # 3. Evaluation Notebook
    print("-" * 60)
    if os.path.exists(eval_file):
        print(f"Official Evaluation Notebook confirmed at: {eval_file}")
    else:
        print(f"WARNING: Evaluation notebook not found at {eval_file}")
    print("=" * 60)

if __name__ == "__main__":
    inspect_dataset()
