from __future__ import annotations

from typing import Dict, List

from app.services.guardrails import GuardrailService
from app.services.llm import LLMService
from app.services.retriever import HybridRetriever
from app.services.validator import ValidationService


SYSTEM_PROMPT = """
You are a hallucination-aware research assistant.
Rules:
1. Answer only from evidence.
2. If evidence is insufficient, say so explicitly.
3. Prefer concise, factual language.
4. Mention uncertainty where the sources disagree.
5. Do not invent citations.
""".strip()

REFLECTION_PROMPT = """
You are a self-reflection verifier.
Review the answer against the evidence.
If any claim is weakly supported, rewrite the answer to be narrower and more evidence-grounded.
""".strip()


class HallucinationDetectionPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = LLMService()
        self.guardrails = GuardrailService()
        self.validator = ValidationService()

    def answer(self, question: str, top_k: int | None = None, use_graph: bool = True) -> Dict:
        evidence = self.retriever.retrieve(question=question, top_k=top_k, use_graph=use_graph)
        reasoning_trace = [
            "Retrieved dense + sparse evidence.",
            "Expanded evidence through GraphRAG neighborhood search." if use_graph else "Graph expansion skipped.",
            "Generated answer with grounding instructions.",
            "Ran self-reflection pass and alignment guardrails.",
        ]

        evidence_block = "\n\n".join(
            [f"Chunk: {e['chunk_id']}\nSource: {e['title']}\n{e['text']}" for e in evidence]
        )
        user_prompt = f"Question: {question}\n\n[EVIDENCE]\n{evidence_block}\n\nWrite an evidence-grounded answer with explicit uncertainty where needed."
        draft = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        reflection_prompt = (
            f"Question: {question}\n\nEvidence:\n{evidence_block}\n\nInitial answer:\n{draft}\n\n"
            "Rewrite only if the answer goes beyond the evidence or lacks caution."
        )
        final_answer = self.llm.generate(REFLECTION_PROMPT, reflection_prompt)

        validation = self.validator.validate(final_answer, evidence)
        guardrails = self.guardrails.assess(question, final_answer)
        if guardrails["blocked"]:
            final_answer = guardrails["safe_answer"]

        retrieval_confidence = round(sum([e["hybrid_score"] for e in evidence]) / max(len(evidence), 1), 3)
        groundedness = round((validation["citation_coverage"] + validation["cross_source_consistency"]) / 2, 3)
        risk_adjusted_trust = round(max(0.0, groundedness * (1 - guardrails["overall_risk"])), 3)

        return {
            "answer": final_answer,
            "citations": evidence,
            "reasoning_trace": reasoning_trace,
            "validation": validation,
            "guardrails": guardrails,
            "scores": {
                "retrieval_confidence": retrieval_confidence,
                "groundedness": groundedness,
                "risk_adjusted_trust": risk_adjusted_trust,
            },
        }
