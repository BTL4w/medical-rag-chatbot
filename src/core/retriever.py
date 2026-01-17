from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
import pickle

import numpy as np
import sys
import os

sys.path.append(os.path.abspath('../src'))

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency
    BM25Okapi = None

from core.chunking import Chunk
from core.embedding import ONNXEmbedding
from core.reranker import ONNXReranker
from db.vector_store import VectorStore


class BM25Retriever:
    def __init__(self, chunks: List[Chunk]) -> None:
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is required for BM25Retriever")
        self.corpus = [chunk.enriched_content for chunk in chunks]
        self.tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunks = chunks

    def search(self, query: str, top_k: int = 20) -> List[Chunk]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    def score(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(query.split())

    def save(self, path: str) -> None:
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        with open(path, "rb") as f:
            try:
                payload = pickle.load(f)
            except ModuleNotFoundError as exc:
                raise ValueError(
                    "BM25 index was saved with an older pickle format. "
                    "Please rebuild the index using BM25Retriever.save()."
                ) from exc
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict) and "chunks" in payload:
            chunks = [
                Chunk(
                    chunk_id=item.get("chunk_id"),
                    enriched_content=item.get("enriched_content", ""),
                    original_content=item.get("original_content", ""),
                    metadata=item.get("metadata", {}),
                    section=item.get("section"),
                    subsection=item.get("subsection"),
                )
                for item in payload["chunks"]
            ]
            return cls(chunks)
        raise ValueError("Unsupported BM25 index format. Please rebuild the index.")


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        embedding_model: ONNXEmbedding,
        reranker: Optional[ONNXReranker] = None,
        candidates: int = 20,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.candidates = candidates

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        filter: Optional[Dict] = None,
    ) -> List[Chunk]:
        query_embedding = self.embedding_model.embed([query])[0]
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.candidates,
            filter=filter,
        )
        vector_scores = {res["chunk"].chunk_id: res["score"] for res in vector_results}

        bm25_scores_raw = self.bm25_retriever.score(query)
        bm25_indices = np.argsort(bm25_scores_raw)[::-1][: self.candidates]
        bm25_scores = {
            self.bm25_retriever.chunks[i].chunk_id: float(bm25_scores_raw[i])
            for i in bm25_indices
        }

        vector_scores = self._normalize(vector_scores)
        bm25_scores = self._normalize(bm25_scores)

        combined: Dict[str, Dict[str, object]] = {}
        for res in vector_results:
            combined.setdefault(res["chunk"].chunk_id, {"chunk": res["chunk"], "score": 0.0})
            combined[res["chunk"].chunk_id]["score"] += vector_weight * vector_scores.get(
                res["chunk"].chunk_id, 0.0
            )
        for chunk_id, score in bm25_scores.items():
            chunk = next(
                (c for c in self.bm25_retriever.chunks if c.chunk_id == chunk_id),
                None,
            )
            if chunk is None:
                continue
            combined.setdefault(chunk_id, {"chunk": chunk, "score": 0.0})
            combined[chunk_id]["score"] += bm25_weight * score

        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        candidates = [item["chunk"] for item in ranked[: self.candidates]]
        chunks = [item["chunk"] for item in ranked[: self.candidates]]

        if self.reranker is None:
            return chunks[:top_k]

        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return [chunks[idx] for idx, _ in reranked]
