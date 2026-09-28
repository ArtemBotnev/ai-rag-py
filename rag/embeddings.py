from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from typing import Iterable, Protocol

from rag.tokenizer import tokenize


class EmbeddingClient(Protocol):
    def embed_all(self, texts: Iterable[str]) -> list[list[float]]:
        raise NotImplementedError


class OllamaEmbeddingClient:
    def __init__(self, base_url: str, model: str, batch_size: int = 16, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

    def embed_all(self, texts: Iterable[str]) -> list[list[float]]:
        inputs = list(texts)
        embeddings: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            embeddings.extend(self._embed_batch(inputs[start : start + self.batch_size]))
        return [min_max_normalize(embedding) for embedding in embeddings]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return self._embed_batch_legacy(batch)
            raise self._ollama_error() from exc
        except urllib.error.URLError as exc:
            raise self._ollama_error() from exc

        if "embeddings" in response_payload:
            return response_payload["embeddings"]
        if "embedding" in response_payload:
            return [response_payload["embedding"]]
        raise RuntimeError(f"Ollama response does not contain embeddings: {response_payload}")

    def _embed_batch_legacy(self, batch: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in batch:
            payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                raise self._ollama_error(prefix="Ollama legacy embeddings request failed") from exc
            if "embedding" not in response_payload:
                raise RuntimeError(f"Ollama response does not contain embedding: {response_payload}")
            embeddings.append(response_payload["embedding"])
        return embeddings

    def _ollama_error(self, prefix: str = "Ollama embeddings request failed") -> RuntimeError:
        return RuntimeError(
            f"{prefix}. Start Ollama and make sure the embedding model is available: "
            f"`ollama pull {self.model}`."
        )


class HashingEmbeddingClient:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_all(self, texts: Iterable[str]) -> list[list[float]]:
        return [embed_text(text, self.dimensions) for text in texts]


def embed_text(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text.lower()):
        digest = hashlib.blake2b(token.text.encode("utf-8"), digest_size=16).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def min_max_normalize(vector: list[float]) -> list[float]:
    if not vector:
        return []
    min_value = min(vector)
    max_value = max(vector)
    if max_value == min_value:
        return [0.0] * len(vector)
    return [(value - min_value) / (max_value - min_value) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))
