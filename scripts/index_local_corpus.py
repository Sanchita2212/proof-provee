import os
import io
import time
import zipfile
import requests

def build_zip_and_index(
    docs_dir: str = "data/Train/Documents",
    api_url: str = "http://127.0.0.1:8000/index",
    collection_name: str = "training_corpus",
    max_docs: int = 10,
):
    doc_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    if max_docs and max_docs > 0:
        doc_files = doc_files[:max_docs]
        print(f"Preparing in-memory ZIP from {docs_dir} ({len(doc_files)} sample documents)...")
    else:
        print(f"Preparing in-memory ZIP from {docs_dir} (ALL {len(doc_files)} documents in corpus)...")
    
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in doc_files:
            fpath = os.path.join(docs_dir, fname)
            z.write(fpath, arcname=fname)
            print(f"  Added {fname} ({os.path.getsize(fpath) / (1024*1024):.2f} MB)")

    zip_bytes = zip_buf.getvalue()
    print(f"Archive built: {len(zip_bytes) / (1024*1024):.2f} MB. Posting to {api_url}...")

    files = {"file": ("documents.zip", zip_bytes, "application/zip")}
    data = {"collection_name": collection_name}

    t0 = time.time()
    resp = requests.post(api_url, data=data, files=files, timeout=300)
    elapsed = time.time() - t0

    if resp.status_code == 200:
        res = resp.json()
        print(f"Indexing Succeeded in {elapsed:.2f}s:")
        print(f"  Collection: {res.get('collection_name')}")
        print(f"  Documents: {res.get('total_documents')}")
        print(f"  Pages: {res.get('total_pages')}")
        print(f"  Chunks: {res.get('total_chunks')}")
        if res.get('warnings'):
            print(f"  Warnings ({len(res.get('warnings'))}): {res.get('warnings')[:3]}")
    else:
        print(f"Indexing Failed (HTTP {resp.status_code}): {resp.text}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default="data/Train/Documents")
    parser.add_argument("--api_url", default="http://127.0.0.1:8000/index")
    parser.add_argument("--collection", default="training_corpus")
    parser.add_argument("--max_docs", type=int, default=10)
    args = parser.parse_args()

    build_zip_and_index(
        docs_dir=args.docs_dir,
        api_url=args.api_url,
        collection_name=args.collection,
        max_docs=args.max_docs,
    )
