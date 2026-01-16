from __future__ import annotations

import json
from typing import Dict, List, Optional

import yaml

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

from src.core.chunking import Chunk


def load_prompts(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def calculate_cost(model: str, total_tokens: int) -> float:
    pricing = {
        "gpt-3.5-turbo": 0.0015,  # placeholder per 1K tokens
    }
    return (total_tokens / 1000.0) * pricing.get(model, 0.0)


class MedicalRAGGenerator:
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo") -> None:
        if OpenAI is None:
            raise ImportError("openai is required for MedicalRAGGenerator")
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        prompts = load_prompts("config/prompts.yaml")
        self.system_prompt = prompts.get("system", "")
        self.rag_prompt = prompts.get("rag_prompt", "{context}\n\n{query}")
        self.summarization_prompt = prompts.get("summarization", "{conversation}")

    def _format_context(self, retrieved_chunks: List[Chunk]) -> str:
        return "\n\n".join(
            f"[Nguon {i + 1}: {chunk.metadata.get('title', '')}]\n{chunk.original_content}"
            for i, chunk in enumerate(retrieved_chunks)
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Chunk],
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict:
        context = self._format_context(retrieved_chunks)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append(
            {
                "role": "user",
                "content": self.rag_prompt.format(context=context, query=query),
            }
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        usage = response.usage
        total_tokens = getattr(usage, "total_tokens", 0)

        return {
            "answer": answer,
            "sources": [chunk.metadata for chunk in retrieved_chunks],
            "tokens_used": total_tokens,
            "cost": calculate_cost(self.model, total_tokens),
        }

    def summarize_conversation(self, conversation_history: List[Dict]) -> str:
        if not conversation_history:
            return ""
        conversation = json.dumps(conversation_history, ensure_ascii=False)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.summarization_prompt.format(conversation=conversation),
                },
            ],
            temperature=0.2,
            max_tokens=256,
        )
        return response.choices[0].message.content
