from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from rapidfuzz import fuzz

from app.services.pipeline import HallucinationDetectionPipeline


class Evaluator:
    def __init__(self):
        self.pipeline = HallucinationDetectionPipeline()

    def run(self, dataset_path: str) -> Dict:
        data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        results: List[Dict] = []
        for item in data:
            output = self.pipeline.answer(item["question"])
            predicted = output["answer"]
            expected = item["expected_answer"]
            exact_match = fuzz.token_set_ratio(predicted.lower(), expected.lower()) / 100
            results.append(
                {
                    "question": item["question"],
                    "predicted_answer": predicted,
                    "expected_answer": expected,
                    "retrieval_confidence": output["scores"]["retrieval_confidence"],
                    "groundedness": output["scores"]["groundedness"],
                    "exact_match": round(exact_match, 3),
                }
            )
        summary = {
            "avg_retrieval_confidence": round(sum(r["retrieval_confidence"] for r in results) / len(results), 3),
            "avg_groundedness": round(sum(r["groundedness"] for r in results) / len(results), 3),
            "avg_exact_match": round(sum(r["exact_match"] for r in results) / len(results), 3),
        }
        return {"summary": summary, "results": results}
