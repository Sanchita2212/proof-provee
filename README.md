# Smart Knowledge Base for Agents

A production-ready, traceable knowledge base that ingests unlabeled document archives and delivers grounded, page-cited answers using **Contrastive Evidence Coverage (CEC)**.

---

## Key Features

- **Contrastive Evidence Coverage (CEC):** A distinctive, bounded retrieval approach avoiding standard embeddings + rerankers, parent-doc retrieval, or recursive agent loops.
- **Dynamic Metadata Discovery:** Extracts entities, reporting periods, scopes, and document types on the fly without hardcoded dictionaries or document names.
- **Strict Provenance & Citations:** Every claim is tied to its source document and page number (`[doc.pdf p.X]`).
- **Official Trace Contract:** Returns traces where every entry strictly has `step_name`, `description`, `input`, and `output`.
- **Collection Isolation:** Complete data separation across named collections with atomic writes.
- **Flexible LLM Adapters:** Seamlessly switch between Groq, OpenCode, and generic OpenAI-compatible providers.
- **Observability:** Integrated Langfuse tracing and local latency tracking.

---

## Quickstart

### 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file:
```bash
cp .env.example .env
```
Set your LLM credentials:
```env
LLM_PROVIDER=groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_KEY=your_key_here
```

### 3. Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

### 1. POST `/index`
Ingests a ZIP file containing PDF documents under a collection name.

```bash
curl -X POST "http://localhost:8000/index" \
  -F "file=@documents.zip" \
  -F "collection_name=annual_reports_2025"
```

**Response:**
```json
{
  "status": "success",
  "collection_name": "annual_reports_2025",
  "total_documents": 10,
  "total_pages": 420,
  "total_chunks": 1250,
  "warnings": [],
  "duration_seconds": 8.42
}
```

### 2. POST `/query`
Answers questions with grounded citations, evidence chunks, and execution traces.

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "annual_reports_2025",
    "query": "What were the key auditor concerns for Gammon India?"
  }'
```

**Response Schema:**
```json
{
  "answer": "The auditors flagged material uncertainties regarding going concern... [Gammon_India_IAR.pdf p.45]",
  "chunks": ["...raw chunk text..."],
  "citations": [
    {
      "document": "Gammon_India_IAR.pdf",
      "page": 45,
      "text": "..."
    }
  ],
  "trace": [
    {
      "step_name": "query_compilation",
      "description": "Compiled user query into structured retrieval facets...",
      "input": "What were the key auditor concerns...",
      "output": "Entities: ['Gammon India'], Periods: ['FY 2025']..."
    }
  ],
  "latency_ms": 1420.5
}
```

---

## Running Tests

Run the full automated test suite:
```bash
pytest tests/ -v
```

---

## Running Evaluation

To evaluate against the benchmark using the official evaluation logic:
```bash
python scripts/run_evaluation.py --benchmark data/Train/Benchmarks/benchmark_100.jsonl --sample_size 5
```

---

## Docker Deployment

Build and run using Docker Compose:
```bash
docker-compose up --build -d
```
