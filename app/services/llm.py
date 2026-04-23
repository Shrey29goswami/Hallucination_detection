from __future__ import annotations

from typing import List
import httpx

from app.core.config import settings


class LLMService:
    def __init__(self):
        self.provider = settings.generator_provider.lower()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openai":
            return self._openai_chat(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        if self.provider == "ollama":
            return self._openai_chat(
                base_url=settings.ollama_base_url,
                api_key="ollama",
                model=settings.ollama_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        return self._stub_answer(user_prompt)

    def _openai_chat(self, base_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=90) as client:
            response = client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _stub_answer(self, prompt: str) -> str:
        evidence_lines: List[str] = []
        for line in prompt.splitlines():
            if line.startswith("[EVIDENCE]") or line.startswith("Question:"):
                continue
            if "Source:" in line or "Chunk:" in line:
                continue
            if line.strip():
                evidence_lines.append(line.strip())
        summary = " ".join(evidence_lines[:6])[:1400]
        return (
            "Evidence-grounded answer (stub mode): "
            + summary
            + "\n\nCitations: Use the cited chunks shown in the UI/API response."
        )
