from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
import hashlib

from app.services.chunker import Chunk, TextChunker
from app.services.embeddings import EmbeddingService
from app.services.graph_store import GraphStore
from app.services.index_store import IndexStore


class IngestionService:
    def __init__(self):
        self.chunker = TextChunker()
        self.embeddings = EmbeddingService()
        self.index = IndexStore()
        self.graph = GraphStore()

    def ingest_paths(self, paths: Iterable[Path]) -> Tuple[List[str], int]:
        all_chunks: List[Chunk] = []
        doc_ids: List[str] = []

        for path in paths:
            text = self.chunker.read_text_file(path)
            document_id = hashlib.sha1(path.name.encode()).hexdigest()[:10]
            doc_ids.append(document_id)
            chunks = self.chunker.split(text=text, document_id=document_id, title=path.name)
            all_chunks.extend(chunks)

        if not all_chunks:
            return [], 0

        vectors = self.embeddings.embed([chunk.text for chunk in all_chunks])
        self.index.build(all_chunks, vectors)
        self.graph.build([chunk.__dict__ for chunk in all_chunks])
        return doc_ids, len(all_chunks)
