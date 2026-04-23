from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import networkx as nx

from app.core.config import settings


class GraphStore:
    def __init__(self):
        self.graph_path = settings.graph_dir / "knowledge_graph.json"
        self.graph = nx.Graph()

    def _extract_entities(self, text: str) -> List[str]:
        candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
        candidates.extend(re.findall(r"\b([A-Z]{2,}[A-Z0-9-]*)\b", text))
        cleaned = [c.strip() for c in candidates if len(c.strip()) > 2]
        counts = Counter(cleaned)
        return [c for c, n in counts.items() if n >= 1][:12]

    def build(self, chunks: List[Dict]) -> None:
        self.graph = nx.Graph()
        for chunk in chunks:
            chunk_node = chunk["chunk_id"]
            self.graph.add_node(chunk_node, node_type="chunk", text=chunk["text"], title=chunk["title"])
            entities = self._extract_entities(chunk["text"])
            for entity in entities:
                self.graph.add_node(entity, node_type="entity")
                self.graph.add_edge(chunk_node, entity)
            for i, left in enumerate(entities):
                for right in entities[i + 1 :]:
                    self.graph.add_edge(left, right)
        payload = nx.node_link_data(self.graph)
        self.graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> bool:
        if not self.graph_path.exists():
            return False
        payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
        self.graph = nx.node_link_graph(payload)
        return True

    def expand(self, seeds: List[str], hops: int = 2) -> Set[str]:
        if len(self.graph.nodes) == 0 and not self.load():
            return set()
        frontier = set(seeds)
        seen = set(seeds)
        for _ in range(hops):
            nxt = set()
            for node in frontier:
                if node not in self.graph:
                    continue
                nxt.update(self.graph.neighbors(node))
            nxt -= seen
            seen.update(nxt)
            frontier = nxt
        return {n for n in seen if str(n).endswith("-chunk-0") or "-chunk-" in str(n)}
