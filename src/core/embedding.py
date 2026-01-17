from __future__ import annotations

from typing import Iterable, List

import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../src'))
try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    ort = None
    AutoTokenizer = None


class ONNXEmbedding:
    def __init__(self, model_path: str = "models/bge-m3-onnx") -> None:
        if AutoTokenizer is None or ort is None:
            raise ImportError("onnxruntime and transformers are required for ONNXEmbedding")

        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

        model_file = os.path.join(model_path, "model.onnx")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"model.onnx not found in {model_path}")

        self.session = ort.InferenceSession(
            model_file,
            providers=["CPUExecutionProvider"]
        )

        self.output_name = self.session.get_outputs()[0].name

    def _mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask[..., None].astype(np.float32)
        masked_output = model_output * mask
        sum_embeddings = masked_output.sum(axis=1)
        sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        outputs = self.session.run([self.output_name], ort_inputs)[0]
        pooled = self._mean_pooling(outputs, inputs["attention_mask"])
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.clip(norms, a_min=1e-9, a_max=None)
        return normalized.astype(np.float32)

    def embed_batch(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batches.append(self.embed(batch))
        return np.vstack(batches)
