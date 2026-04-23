from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import hnswlib
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.services.chunker import Chunk


class IndexStore:
    def __init__(self):
        self.index_path = settings.index_dir / "chunks_hnsw.bin"
        self.meta_path = settings.index_dir / "chunks_meta.json"
        self._index = None
        self._meta: List[Dict] = []
        self._bm25 = None

    @property
    def meta(self) -> List[Dict]:
        return self._meta

    def build(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if not chunks:
            return
        dim = len(vectors[0])
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
        index.set_ef(50)
        index.add_items(np.array(vectors, dtype=np.float32), np.arange(len(vectors)))
        index.save_index(str(self.index_path))

        self._meta = [chunk.__dict__ for chunk in chunks]
        self.meta_path.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
        self._index = index
        self._rebuild_bm25()

    def load(self) -> bool:
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if not self._meta:
            return False
        dim = 384  # all-MiniLM-L6-v2 output size
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.load_index(str(self.index_path))
        self._index.set_ef(50)
        self._rebuild_bm25()
        return True

    def _rebuild_bm25(self) -> None:
        corpus = [m["text"].lower().split() for m in self._meta]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def query_dense(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        if self._index is None:
            if not self.load():
                return []
        labels, distances = self._index.knn_query(np.array([vector], dtype=np.float32), k=min(top_k, len(self._meta)))
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            item = dict(self._meta[int(idx)])
            item["score"] = float(1 - dist)
            item["retrieval_type"] = "dense"
            results.append(item)
        return results

    def query_bm25(self, query: str, top_k: int = 5) -> List[Dict]:
        if self._bm25 is None and not self.load():
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_ids = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        results = []
        for idx in top_ids:
            item = dict(self._meta[int(idx)])
            item["score"] = float(scores[idx])
            item["retrieval_type"] = "sparse"
            results.append(item)
        return results
