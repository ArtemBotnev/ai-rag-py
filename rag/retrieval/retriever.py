from __future__ import annotations

import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rag.inference.embeddings import EmbeddingClient, cosine_similarity


class RetrievalMode(str, Enum):
    BASELINE = "BASELINE"
    IMPROVED = "IMPROVED"


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode = RetrievalMode.BASELINE
    top_k_before: int = 8
    top_k_after: int = 4
    relevance_threshold: float = 0.45
    enable_query_rewrite: bool = True


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
    similarity_score: float
    text: str
    preview: str
    token_count: int
    matched_terms: list[str]
    source_queries: list[str]


@dataclass(frozen=True)
class RetrievalResult:
    mode: RetrievalMode
    question: str
    chunks: list[RetrievedChunk]
    candidate_chunks: list[RetrievedChunk]
    discarded_chunks: list[RetrievedChunk]
    duration_ms: int
    top_k_before: int
    top_k_after: int
    relevance_threshold: float | None
    rewritten_query: str | None = None


@dataclass(frozen=True)
class ScoredChunk:
    chunk: IndexedChunk
    similarity_score: float
    rank: int


@dataclass(frozen=True)
class RankedCandidate:
    chunk: IndexedChunk
    similarity_score: float
    rank: int
    query_label: str


@dataclass(frozen=True)
class MergedCandidate:
    chunk: IndexedChunk
    similarity_score: float
    best_rank: int
    source_queries: list[str]


class SqliteRagRetriever:
    def __init__(self, sqlite_path: Path, strategy: str, embedding_client: EmbeddingClient) -> None:
        self.sqlite_path = sqlite_path
        self.strategy = strategy
        self.embedding_client = embedding_client
        self._chunks = load_chunks(sqlite_path, strategy)
        if not self._chunks:
            raise RuntimeError(f"No indexed chunks found for strategy {strategy}. Run scripts/index_documents.py first.")

    def search(
        self,
        question: str,
        limit: int = 4,
        config: RetrievalConfig | None = None,
        rewritten_query: str | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        sanitized_question = question.strip()
        if not sanitized_question:
            raise RuntimeError("Question must not be blank.")

        normalized_config = normalize_config(config or RetrievalConfig(top_k_before=limit, top_k_after=limit))
        active_queries = [sanitized_question]
        candidate_rewrite = (rewritten_query or "").strip()
        if (
            normalized_config.mode == RetrievalMode.IMPROVED
            and normalized_config.enable_query_rewrite
            and candidate_rewrite
            and candidate_rewrite.lower() != sanitized_question.lower()
        ):
            active_queries.append(candidate_rewrite)

        embeddings = self.embedding_client.embed_all(active_queries)
        if len(embeddings) != len(active_queries):
            raise RuntimeError("Embedding client returned an unexpected number of embeddings.")

        if normalized_config.mode == RetrievalMode.BASELINE:
            chunks = [
                scored_to_retrieved(hit, score=hit.similarity_score)
                for hit in self._search_by_embedding(embeddings[0], normalized_config.top_k_after)
            ]
            return RetrievalResult(
                mode=RetrievalMode.BASELINE,
                question=sanitized_question,
                rewritten_query=None,
                chunks=chunks,
                candidate_chunks=chunks,
                discarded_chunks=[],
                duration_ms=elapsed_ms(started),
                top_k_before=normalized_config.top_k_after,
                top_k_after=normalized_config.top_k_after,
                relevance_threshold=None,
            )

        result = self._search_improved(
            question=sanitized_question,
            active_queries=active_queries,
            embeddings=embeddings,
            config=normalized_config,
            started=started,
        )
        return result

    def _search_by_embedding(self, query_embedding: list[float], limit: int) -> list[ScoredChunk]:
        return [
            ScoredChunk(chunk=chunk, similarity_score=score, rank=index)
            for index, (chunk, score) in enumerate(
                sorted(
                    ((chunk, cosine_similarity(query_embedding, chunk.embedding)) for chunk in self._chunks),
                    key=lambda item: item[1],
                    reverse=True,
                )[:limit]
            )
        ]

    def _search_improved(
        self,
        question: str,
        active_queries: list[str],
        embeddings: list[list[float]],
        config: RetrievalConfig,
        started: float,
    ) -> RetrievalResult:
        ranked_candidates: list[RankedCandidate] = []
        for query_index, embedding in enumerate(embeddings):
            query_label = "original" if query_index == 0 else "rewritten"
            ranked_candidates.extend(
                RankedCandidate(
                    chunk=hit.chunk,
                    similarity_score=hit.similarity_score,
                    rank=hit.rank,
                    query_label=query_label,
                )
                for hit in self._search_by_embedding(embedding, config.top_k_before)
            )

        merged_candidates = merge_candidates(ranked_candidates)
        query_terms = extract_query_terms(active_queries)
        reranked = rerank_candidates(merged_candidates, query_terms)
        candidate_chunks = reranked[: config.top_k_before]
        chunks = [
            candidate
            for candidate in candidate_chunks
            if candidate.score >= config.relevance_threshold
        ][: config.top_k_after]
        kept_ids = {chunk.chunk_id for chunk in chunks}
        discarded_chunks = [candidate for candidate in candidate_chunks if candidate.chunk_id not in kept_ids]

        return RetrievalResult(
            mode=RetrievalMode.IMPROVED,
            question=question,
            rewritten_query=active_queries[1] if len(active_queries) > 1 else None,
            chunks=chunks,
            candidate_chunks=candidate_chunks,
            discarded_chunks=discarded_chunks,
            duration_ms=elapsed_ms(started),
            top_k_before=config.top_k_before,
            top_k_after=config.top_k_after,
            relevance_threshold=config.relevance_threshold,
        )


def normalize_config(config: RetrievalConfig) -> RetrievalConfig:
    top_k_after = max(1, config.top_k_after)
    top_k_before = max(top_k_after, config.top_k_before)
    return RetrievalConfig(
        mode=coerce_retrieval_mode(config.mode),
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        relevance_threshold=min(1.0, max(0.0, config.relevance_threshold)),
        enable_query_rewrite=config.enable_query_rewrite,
    )


def coerce_retrieval_mode(mode: RetrievalMode | str) -> RetrievalMode:
    if isinstance(mode, RetrievalMode):
        return mode
    return RetrievalMode(str(mode).upper())


def merge_candidates(candidates: list[RankedCandidate]) -> list[MergedCandidate]:
    grouped: dict[str, list[RankedCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.chunk.chunk_id, []).append(candidate)

    merged: list[MergedCandidate] = []
    for duplicates in grouped.values():
        representative = max(duplicates, key=lambda item: item.similarity_score)
        source_queries = []
        for duplicate in duplicates:
            if duplicate.query_label not in source_queries:
                source_queries.append(duplicate.query_label)
        merged.append(
            MergedCandidate(
                chunk=representative.chunk,
                similarity_score=representative.similarity_score,
                best_rank=min(duplicate.rank for duplicate in duplicates),
                source_queries=source_queries,
            )
        )
    return merged


def rerank_candidates(candidates: list[MergedCandidate], query_terms: set[str]) -> list[RetrievedChunk]:
    if not candidates:
        return []

    min_similarity = min(candidate.similarity_score for candidate in candidates)
    max_similarity = max(candidate.similarity_score for candidate in candidates)
    max_rank = max(1, max(candidate.best_rank for candidate in candidates))

    reranked: list[RetrievedChunk] = []
    for candidate in candidates:
        path_tokens = tokenize_for_search(candidate.chunk.file_path)
        title_tokens = tokenize_for_search(candidate.chunk.title)
        section_tokens = tokenize_for_search(candidate.chunk.section)
        text_tokens = tokenize_for_search(candidate.chunk.text)
        matched_terms = sorted(
            term
            for term in query_terms
            if term in path_tokens or term in title_tokens or term in section_tokens or term in text_tokens
        )
        path_overlap = overlap_ratio(query_terms, path_tokens)
        title_overlap = overlap_ratio(query_terms, title_tokens)
        section_overlap = overlap_ratio(query_terms, section_tokens)
        text_overlap = overlap_ratio(query_terms, text_tokens)
        normalized_similarity = normalize_value(candidate.similarity_score, min_similarity, max_similarity)
        rank_signal = 1.0 - (candidate.best_rank / max_rank)
        relevance_score = (
            normalized_similarity * 0.40
            + rank_signal * 0.15
            + path_overlap * 0.20
            + title_overlap * 0.10
            + section_overlap * 0.05
            + text_overlap * 0.10
            + max(path_overlap, title_overlap, section_overlap, text_overlap) * 0.10
        )
        if not matched_terms:
            relevance_score -= 0.15
        if path_overlap > 0.0 or title_overlap > 0.0 or section_overlap > 0.0:
            relevance_score += 0.05

        reranked.append(
            indexed_to_retrieved(
                chunk=candidate.chunk,
                score=min(1.0, max(0.0, relevance_score)),
                similarity_score=candidate.similarity_score,
                matched_terms=matched_terms,
                source_queries=candidate.source_queries,
            )
        )

    return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)


def scored_to_retrieved(hit: ScoredChunk, score: float) -> RetrievedChunk:
    return indexed_to_retrieved(
        chunk=hit.chunk,
        score=score,
        similarity_score=hit.similarity_score,
        matched_terms=[],
        source_queries=["original"],
    )


def indexed_to_retrieved(
    chunk: IndexedChunk,
    score: float,
    similarity_score: float,
    matched_terms: list[str],
    source_queries: list[str],
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        file_path=chunk.file_path,
        section=chunk.section,
        title=chunk.title,
        score=score,
        similarity_score=similarity_score,
        text=chunk.text,
        preview=preview(chunk.text),
        token_count=chunk.token_count,
        matched_terms=matched_terms,
        source_queries=source_queries,
    )


def extract_query_terms(queries: list[str]) -> set[str]:
    terms = {
        token
        for query in queries
        for token in tokenize_for_search(query)
        if token not in STOP_WORDS
    }
    if terms:
        return terms
    return {token for query in queries for token in tokenize_for_search(query, min_length=1)}


def tokenize_for_search(value: str, min_length: int = 2) -> set[str]:
    prepared = CAMEL_CASE_BOUNDARY.sub(r"\1 \2", value).lower()
    return {match.group(0) for match in TOKEN_REGEX.finditer(prepared) if len(match.group(0)) >= min_length}


def overlap_ratio(query_terms: set[str], candidate_terms: set[str]) -> float:
    if not query_terms or not candidate_terms:
        return 0.0
    return sum(1 for term in query_terms if term in candidate_terms) / len(query_terms)


def normalize_value(value: float, minimum: float, maximum: float) -> float:
    if maximum - minimum < 1e-9:
        return 1.0
    return (value - minimum) / (maximum - minimum)


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


CAMEL_CASE_BOUNDARY = re.compile(r"([a-z])([A-Z])")
TOKEN_REGEX = re.compile(r"\w+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}
