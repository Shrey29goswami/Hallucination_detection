from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import re

from app.core.config import settings


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str


class TextChunker:
    def __init__(self, max_chunk_size: int | None = None, overlap: int | None = None):
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def read_text_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def normalize_text(self, text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def split(self, text: str, document_id: str, title: str) -> List[Chunk]:
        text = self.normalize_text(text)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        chunks: List[Chunk] = []
        current = ""
        idx = 0

        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= self.max_chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(Chunk(chunk_id=f"{document_id}-chunk-{idx}", document_id=document_id, title=title, text=current))
                idx += 1
                overlap_text = current[-self.overlap :] if self.overlap > 0 else ""
                current = f"{overlap_text}\n\n{para}".strip()
            else:
                for start in range(0, len(para), self.max_chunk_size - self.overlap):
                    part = para[start : start + self.max_chunk_size]
                    chunks.append(Chunk(chunk_id=f"{document_id}-chunk-{idx}", document_id=document_id, title=title, text=part))
                    idx += 1
                current = ""

        if current:
            chunks.append(Chunk(chunk_id=f"{document_id}-chunk-{idx}", document_id=document_id, title=title, text=current))
        return chunks
