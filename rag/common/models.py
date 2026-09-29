from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenSlice:
    text: str
    start_offset: int
    end_offset: int
    token_count: int


@dataclass(frozen=True)
class SourceDocument:
    source: str
    file_path: str
    title: str
    content_type: str
    text: str


@dataclass(frozen=True)
class SectionUnit:
    section: str
    text: str
    start_offset: int
    end_offset: int
    token_count: int


@dataclass(frozen=True)
class ChunkingConfig:
    min_tokens: int = 500
    target_tokens: int = 800
    max_tokens: int = 1000
    overlap_tokens: int = 80


@dataclass(frozen=True)
class Chunk:
    strategy: str
    chunk_id: str
    source: str
    title: str
    file_path: str
    section: str
    content_type: str
    start_offset: int
    end_offset: int
    token_count: int
    char_count: int
    text: str
    embedding: list[float]
