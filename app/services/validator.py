from __future__ import annotations

from collections import defaultdict
from typing import Dict, List
import re


class ValidationService:
    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def validate(self, answer: str, evidence: List[Dict]) -> Dict:
        answer_sents = self._split_sentences(answer)
        if not answer_sents:
            return {
                "supported_sentences": 0,
                "unsupported_sentences": 0,
                "citation_coverage": 0.0,
                "cross_source_consistency": 0.0,
                "unsupported_claims": [],
            }

        evidence_by_doc = defaultdict(list)
        for item in evidence:
            evidence_by_doc[item["document_id"]].append(item["text"].lower())

        supported = 0
        unsupported_claims = []
        for sent in answer_sents:
            sent_words = set(re.findall(r"\w+", sent.lower()))
            hit_docs = 0
            for doc_id, chunks in evidence_by_doc.items():
                doc_supported = any(len(sent_words.intersection(set(re.findall(r"\w+", chunk)))) >= 5 for chunk in chunks)
                if doc_supported:
                    hit_docs += 1
            if hit_docs >= 1:
                supported += 1
            else:
                unsupported_claims.append(sent)

        citation_coverage = supported / len(answer_sents)
        cross_source = min(1.0, (sum(1 for sent in answer_sents if self._is_multi_source_supported(sent, evidence_by_doc)) / len(answer_sents)))
        return {
            "supported_sentences": supported,
            "unsupported_sentences": len(answer_sents) - supported,
            "citation_coverage": round(citation_coverage, 3),
            "cross_source_consistency": round(cross_source, 3),
            "unsupported_claims": unsupported_claims,
        }

    def _is_multi_source_supported(self, sentence: str, evidence_by_doc: Dict[str, List[str]]) -> bool:
        sent_words = set(re.findall(r"\w+", sentence.lower()))
        hit_docs = 0
        for chunks in evidence_by_doc.values():
            if any(len(sent_words.intersection(set(re.findall(r"\w+", chunk)))) >= 4 for chunk in chunks):
                hit_docs += 1
        return hit_docs >= 2
