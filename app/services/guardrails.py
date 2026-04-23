from __future__ import annotations

from typing import Dict, List
import re

from app.core.config import settings


TOXIC_PATTERNS = [
    r"\bidiot\b", r"\bkill\b", r"\bhate\b", r"\bstupid\b", r"\bterrorist\b", r"\bnazi\b",
]
BIAS_PATTERNS = [
    r"all\s+(women|men|immigrants|muslims|hindus|christians)",
    r"(women|men|immigrants|muslims|hindus|christians)\s+are\s+always",
]
PROMPT_ATTACK_PATTERNS = [
    r"ignore previous instructions",
    r"reveal system prompt",
    r"jailbreak",
    r"bypass guardrails",
]


class GuardrailService:
    def assess(self, question: str, answer: str) -> Dict:
        text = f"{question}\n{answer}".lower()
        toxicity_hits = [p for p in TOXIC_PATTERNS if re.search(p, text)]
        bias_hits = [p for p in BIAS_PATTERNS if re.search(p, text)]
        attack_hits = [p for p in PROMPT_ATTACK_PATTERNS if re.search(p, text)]

        toxicity_score = min(1.0, len(toxicity_hits) * 0.35)
        bias_score = min(1.0, len(bias_hits) * 0.4)
        attack_score = min(1.0, len(attack_hits) * 0.5)
        overall_risk = round((toxicity_score + bias_score + attack_score) / 3, 3)

        blocked = overall_risk >= settings.toxicity_threshold
        safe_answer = answer
        if blocked:
            safe_answer = (
                "The request or generated response triggered the alignment guardrails. "
                "Please rephrase the question or constrain it to factual, non-harmful content."
            )

        return {
            "toxicity_score": toxicity_score,
            "bias_score": bias_score,
            "prompt_attack_score": attack_score,
            "overall_risk": overall_risk,
            "blocked": blocked,
            "safe_answer": safe_answer,
            "flags": {
                "toxicity_hits": toxicity_hits,
                "bias_hits": bias_hits,
                "prompt_attack_hits": attack_hits,
            },
        }
