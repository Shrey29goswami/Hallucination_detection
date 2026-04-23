from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str


class IngestResponse(BaseModel):
    files_ingested: int
    chunks_indexed: int
    document_ids: List[str]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    use_graph: bool = True
    top_k: Optional[int] = None


class EvidenceChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float
    retrieval_type: str


class AskResponse(BaseModel):
    answer: str
    citations: List[EvidenceChunk]
    reasoning_trace: List[str]
    validation: Dict[str, Any]
    guardrails: Dict[str, Any]
    scores: Dict[str, float]


class EvalRequest(BaseModel):
    dataset_path: str = "data/eval_dataset.json"


class EvalResult(BaseModel):
    question: str
    predicted_answer: str
    expected_answer: str
    retrieval_confidence: float
    groundedness: float
    exact_match: float


class EvalResponse(BaseModel):
    summary: Dict[str, float]
    results: List[EvalResult]
