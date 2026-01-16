from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    ort = None
    AutoTokenizer = None


class ONNXReranker:
    def __init__(self, model_path: str = "models/bge-reranker-v2-m3-onnx") -> None:
        if AutoTokenizer is None or ort is None:
            raise ImportError("onnxruntime and transformers are required for ONNXReranker")
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        self.session = ort.InferenceSession(model_path)
        self.output_name = self.session.get_outputs()[0].name

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        ort_inputs = {k: v for k, v in inputs.items()}
        outputs = self.session.run([self.output_name], ort_inputs)[0]
        scores = outputs.squeeze().astype(np.float32)
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
