from __future__ import annotations

import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from rag.inference.embeddings import EmbeddingClient, cosine_similarity


@dataclass(frozen=True)
class IndexedChunk:
    strategy: str
    chunk_id: str
    source: str
    title: str
    file_path: str
    section: str
    content_type: str
    token_count: int
    char_count: int
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    file_path: str
    section: str
    title: str
    score: float
    text: str
    preview: str
    token_count: int


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    duration_ms: int


class SqliteRagRetriever:
    def __init__(self, sqlite_path: Path, strategy: str, embedding_client: EmbeddingClient) -> None:
        self.sqlite_path = sqlite_path
        self.strategy = strategy
        self.embedding_client = embedding_client
        self._chunks = load_chunks(sqlite_path, strategy)
        if not self._chunks:
            raise RuntimeError(f"No indexed chunks found for strategy {strategy}. Run scripts/index_documents.py first.")

    def search(self, question: str, limit: int = 4) -> RetrievalResult:
        started = time.perf_counter()
        query_embeddings = self.embedding_client.embed_all([question])
        if not query_embeddings:
            raise RuntimeError("Embedding client returned no embedding for the question.")
        query_embedding = query_embeddings[0]
        hits = sorted(
            (
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file_path,
                    section=chunk.section,
                    title=chunk.title,
                    score=cosine_similarity(query_embedding, chunk.embedding),
                    text=chunk.text,
                    preview=preview(chunk.text),
                    token_count=chunk.token_count,
                )
                for chunk in self._chunks
            ),
            key=lambda hit: hit.score,
            reverse=True,
        )[:limit]
        return RetrievalResult(chunks=hits, duration_ms=elapsed_ms(started))


def load_chunks(sqlite_path: Path, strategy: str) -> list[IndexedChunk]:
    if not sqlite_path.exists():
        raise RuntimeError(f"SQLite index not found: {sqlite_path}")
    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(
            """
            select strategy, chunk_id, source, title, file_path, section, content_type,
                   token_count, char_count, embedding_blob, text_content
            from chunks
            where strategy = ?
            """,
            (strategy,),
        ).fetchall()
    return [
        IndexedChunk(
            strategy=row[0],
            chunk_id=row[1],
            source=row[2],
            title=row[3],
            file_path=row[4],
            section=row[5],
            content_type=row[6],
            token_count=row[7],
            char_count=row[8],
            embedding=decode_embedding(row[9]),
            text=row[10],
        )
        for row in rows
    ]


def decode_embedding(blob: bytes) -> list[float]:
    if len(blob) % 4 != 0:
        raise RuntimeError(f"Invalid embedding blob size: {len(blob)}")
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def preview(text: str, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
