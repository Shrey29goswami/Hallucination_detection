from pathlib import Path

from app.core.config import settings
from app.services.ingestion import IngestionService


if __name__ == "__main__":
    paths = sorted((Path("data/sample_docs")).glob("*.txt"))
    doc_ids, chunk_count = IngestionService().ingest_paths(paths)
    print({"document_ids": doc_ids, "chunks_indexed": chunk_count, "saved_to": str(settings.base_dir)})
