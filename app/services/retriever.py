from __future__ import annotations

from typing import Dict, List

from rapidfuzz import fuzz

from app.core.config import settings
from app.services.embeddings import EmbeddingService
from app.services.graph_store import GraphStore
from app.services.index_store import IndexStore


class HybridRetriever:
    def __init__(self):
        self.embeddings = EmbeddingService()
        self.index = IndexStore()
        self.graph = GraphStore()

    def retrieve(self, question: str, top_k: int | None = None, use_graph: bool = True) -> List[Dict]:
        top_k = top_k or settings.top_k
        query_vector = self.embeddings.embed([question])[0]
        dense = self.index.query_dense(query_vector, top_k=top_k)
        sparse = self.index.query_bm25(question, top_k=top_k)

        merged: Dict[str, Dict] = {}
        max_sparse = max([x["score"] for x in sparse], default=1.0) or 1.0

        for item in dense:
            merged[item["chunk_id"]] = {**item, "hybrid_score": item["score"]}
        for item in sparse:
            normalized = item["score"] / max_sparse
            if item["chunk_id"] not in merged:
                merged[item["chunk_id"]] = {**item, "hybrid_score": 0.0}
            merged[item["chunk_id"]]["hybrid_score"] += 0.65 * normalized
            merged[item["chunk_id"]]["retrieval_type"] = "hybrid"

        if use_graph:
            seeds = [k for k, _ in sorted(merged.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)[:2]]
            expanded = self.graph.expand(seeds, hops=settings.graph_hops)
            for chunk_id in expanded:
                if chunk_id in merged:
                    merged[chunk_id]["hybrid_score"] += 0.10
                    merged[chunk_id]["retrieval_type"] = "graph+hybrid"
                else:
                    for meta in self.index.meta:
                        if meta["chunk_id"] == chunk_id:
                            merged[chunk_id] = {**meta, "score": 0.0, "hybrid_score": 0.10, "retrieval_type": "graph"}
                            break

        for item in merged.values():
            lexical = fuzz.partial_ratio(question.lower(), item["text"].lower()) / 100
            item["hybrid_score"] += 0.15 * lexical

        return sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
