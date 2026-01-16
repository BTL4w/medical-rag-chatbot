from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from src.core.chunking import Chunk


class VectorStore(ABC):
    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, object]]:
        raise NotImplementedError


class QdrantStore(VectorStore):
    def __init__(self, url: str = "http://localhost:6333") -> None:
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("qdrant-client is required for QdrantStore") from exc
        self.client = QdrantClient(url=url)

    def create_collection(self, name: str = "youmed_articles", dimension: int = 1024) -> None:
        from qdrant_client.http import models as qmodels

        self.client.recreate_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dimension, distance=qmodels.Distance.COSINE),
        )

    def upsert(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        from qdrant_client.http import models as qmodels

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                # "original_content": chunk.original_content,
                "enriched_content": chunk.enriched_content,
                "metadata": chunk.metadata,
                "section": chunk.section,
                "subsection": chunk.subsection,
            }
            points.append(
                qmodels.PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )
        self.client.upsert(collection_name="youmed_articles", points=points)

    def _chunk_from_payload(self, payload: Dict, chunk_id: str) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            enriched_content=payload.get("enriched_content", ""),
            # original_content=payload.get("original_content", ""),
            metadata=payload.get("metadata", {}),
            section=payload.get("section"),
            subsection=payload.get("subsection"),
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, object]]:
        search_kwargs = {}
        if filter:
            search_kwargs["query_filter"] = filter
        results = self.client.search(
            collection_name="youmed_articles",
            query_vector=query_embedding.tolist(),
            limit=top_k,
            **search_kwargs,
        )
        output = []
        for hit in results:
            chunk = self._chunk_from_payload(hit.payload, hit.id)
            output.append({"chunk": chunk, "score": float(hit.score)})
        return output


class PineconeStore(VectorStore):
    def __init__(self, api_key: str, index_name: str = "youmed-articles") -> None:
        try:
            from pinecone import Pinecone
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("pinecone-client is required for PineconeStore") from exc
        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(index_name)
        self.index_name = index_name

    def create_collection(self, name: str, dimension: int) -> None:
        if name not in self.client.list_indexes().names():
            from pinecone import ServerlessSpec

            self.client.create_index(
                name=name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

    def upsert(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                (
                    chunk.chunk_id,
                    embedding.tolist(),
                    {
                        # "original_content": chunk.original_content,
                        "enriched_content": chunk.enriched_content,
                        "metadata": chunk.metadata,
                        "section": chunk.section,
                        "subsection": chunk.subsection,
                    },
                )
            )
        self.index.upsert(vectors=vectors)

    def _chunk_from_metadata(self, metadata: Dict, chunk_id: str) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            enriched_content=metadata.get("enriched_content", ""),
            # original_content=metadata.get("original_content", ""),
            metadata=metadata.get("metadata", {}),
            section=metadata.get("section"),
            subsection=metadata.get("subsection"),
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, object]]:
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )
        output = []
        for match in results.get("matches", []):
            chunk = self._chunk_from_metadata(match.get("metadata", {}), match.get("id"))
            output.append({"chunk": chunk, "score": float(match.get("score", 0.0))})
        return output


class VectorStoreFactory:
    @staticmethod
    def create(store_type: str = "qdrant", **kwargs) -> VectorStore:
        if store_type == "qdrant":
            return QdrantStore(**kwargs)
        if store_type == "pinecone":
            return PineconeStore(**kwargs)
        raise ValueError(f"Unsupported vector store type: {store_type}")
