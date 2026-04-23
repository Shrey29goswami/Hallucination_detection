from __future__ import annotations

from pathlib import Path
from typing import List
import shutil

from fastapi import FastAPI, File, UploadFile

from app.core.config import settings
from app.models.schemas import AskRequest, AskResponse, EvalRequest, EvalResponse, HealthResponse, IngestResponse
from app.services.evaluator import Evaluator
from app.services.ingestion import IngestionService
from app.services.pipeline import HallucinationDetectionPipeline

app = FastAPI(title=settings.app_name, version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)) -> IngestResponse:
    saved_paths: List[Path] = []
    for file in files:
        target = settings.document_dir / file.filename
        with target.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(target)
    doc_ids, chunk_count = IngestionService().ingest_paths(saved_paths)
    return IngestResponse(files_ingested=len(saved_paths), chunks_indexed=chunk_count, document_ids=doc_ids)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    result = HallucinationDetectionPipeline().answer(
        question=payload.question,
        top_k=payload.top_k,
        use_graph=payload.use_graph,
    )
    return AskResponse(**result)


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(payload: EvalRequest) -> EvalResponse:
    result = Evaluator().run(payload.dataset_path)
    return EvalResponse(**result)
