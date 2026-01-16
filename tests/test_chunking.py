import json
import sys
import os

# Add src to path (when running from testing folder or project root)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('src')

from core.chunking import MarkdownChunker


def test_chunking():
    chunker = MarkdownChunker()
    with open("data/processed/youmed_articles_test.jsonl", "r", encoding="utf-8") as f:
        doc = json.loads(f.readline())
    chunks = chunker.chunk_document(doc["content"], doc["metadata"])
    print(chunks)
    assert len(chunks) > 0
    assert all(chunk.metadata.get("keyword") in chunk.enriched_content for chunk in chunks)
    assert all(chunk.section or chunk.subsection for chunk in chunks)
#test_chunking()
print("Test chunking passed")