from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.abspath('../src'))
try:
    from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - optional dependency
    MarkdownHeaderTextSplitter = None
    RecursiveCharacterTextSplitter = None


@dataclass
class Chunk:
    chunk_id: str
    enriched_content: str
    original_content: Optional[str] = None
    metadata: Optional[Dict] = None
    section: Optional[str] = None
    subsection: Optional[str] = None


class MarkdownChunker:
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 50,
        headers_to_split: Optional[List[tuple]] = None,
    ) -> None:
        self.headers_to_split = headers_to_split or [("##", "section"), ("###", "subsection")]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._init_splitters()

    def _init_splitters(self) -> None:
        self._header_splitter = None
        self._recursive_splitter = None
        if MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter:
            self._header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split
            )
            self._recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def _split_by_headers_fallback(self, content: str) -> List[Dict[str, Optional[str]]]:
        lines = content.splitlines()
        section = None
        subsection = None
        buffer: List[str] = []
        segments: List[Dict[str, Optional[str]]] = []

        def flush() -> None:
            if buffer:
                segments.append(
                    {
                        "section": section,
                        "subsection": subsection,
                        "text": "\n".join(buffer).strip(),
                    }
                )
                buffer.clear()

        for line in lines:
            if line.startswith("### "):
                flush()
                subsection = line.replace("###", "", 1).strip() or None
                continue
            if line.startswith("## "):
                flush()
                section = line.replace("##", "", 1).strip() or None
                subsection = None
                continue
            buffer.append(line)

        flush()
        return [seg for seg in segments if seg.get("text")]

    def _split_text_fallback(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(text_length, start + self.chunk_size)
            chunks.append(text[start:end])
            if end == text_length:
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def _enrich_text(
        self,
        chunk_text: str,
        metadata: Dict,
        section: Optional[str],
        subsection: Optional[str],
    ) -> str:
        parts = [
            (metadata or {}).get("keyword"),
            section,
            subsection,
            chunk_text,
        ]
        parts = [part.strip() for part in parts if part]
        return " | ".join(parts)

    def chunk_document(self, content: str, metadata: Dict) -> List[Chunk]:
        if not content:
            return []

        chunks: List[Chunk] = []
        if self._header_splitter and self._recursive_splitter:
            header_docs = self._header_splitter.split_text(content)
            for doc in header_docs:
                section = doc.metadata.get("section")
                subsection = doc.metadata.get("subsection")
                for sub_chunk in self._recursive_splitter.split_text(doc.page_content):
                    chunk_id = str(uuid.uuid4())
                    enriched = self._enrich_text(sub_chunk, metadata, section, subsection)
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            enriched_content=enriched,
                            original_content=sub_chunk,
                            metadata=metadata,
                            section=section,
                            subsection=subsection,
                        )
                    )
            return chunks

        for segment in self._split_by_headers_fallback(content):
            for sub_chunk in self._split_text_fallback(segment["text"]):
                chunk_id = str(uuid.uuid4())
                enriched = self._enrich_text(
                    sub_chunk,
                    metadata,
                    segment.get("section"),
                    segment.get("subsection"),
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        enriched_content=enriched,
                        original_content=sub_chunk,
                        metadata=metadata,
                        section=segment.get("section"),
                        subsection=segment.get("subsection"),
                    )
                )
        return chunks

    def process_jsonl(self, input_path: str) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                chunks = self.chunk_document(doc.get("content", ""), doc.get("metadata", {}))
                all_chunks.extend(chunks)
        return all_chunks
