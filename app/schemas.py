from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing import List, Dict, Any, Optional

class TraceStep(BaseModel):
    """
    Mandatory official schema: exactly step_name, description, input, output.
    No extra fields permitted in individual trace entries.
    """
    step_name: str
    description: str
    input: str
    output: str

class ChunkCitation(BaseModel):
    document: str
    page: int
    text: str
    section: Optional[str] = None
    confidence: Optional[float] = 1.0

class EvaluationMetrics(BaseModel):
    answer_precision: float
    answer_recall: float
    answer_f1: float
    answer_loss: float
    context_precision: float
    context_recall: float
    context_f1: float
    context_loss: float
    answer_confusion: Dict[str, Any]
    answer_detail: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    collection_name: str
    query: str
    benchmark_answer: Optional[str] = None
    benchmark_chunks: Optional[List[str]] = None
    evaluate: Optional[bool] = False


class EvidenceChunkMetadata(BaseModel):
    """Original source provenance for one returned evidence excerpt."""
    file_name: str
    page_numbers: List[int]


class EvidenceChunk(BaseModel):
    content: str
    metadata: EvidenceChunkMetadata


class QueryResponse(BaseModel):
    """Public /query contract: only answer, evidence, and four-field trace."""
    model_config = ConfigDict(extra="forbid")
    answer: str
    chunks: List[EvidenceChunk]
    trace: List[TraceStep]
    _latency_ms: Optional[float] = PrivateAttr(default=None)

    @property
    def latency_ms(self) -> Optional[float]:
        """Internal-only timing retained for local evaluation scripts."""
        return self._latency_ms

class IndexResponse(BaseModel):
    status: str
    collection_name: str
    total_documents: int
    total_pages: int
    total_chunks: int
    warnings: List[str] = Field(default_factory=list)
    duration_seconds: float
