from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rag.models import Chunk


def write_sqlite(sqlite_path: Path, chunks_by_strategy: dict[str, list[Chunk]]) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            create table if not exists chunks (
                id integer primary key autoincrement,
                strategy text not null,
                chunk_id text not null,
                source text not null,
                title text not null,
                file_path text not null,
                section text not null,
                content_type text not null,
                start_offset integer not null,
                end_offset integer not null,
                token_count integer not null,
                char_count integer not null,
                embedding_blob blob not null,
                text_content text not null,
                created_at text not null
            )
            """
        )
        connection.execute("create index if not exists idx_chunks_strategy on chunks(strategy)")
        connection.execute("create index if not exists idx_chunks_file_path on chunks(file_path)")
        for strategy, chunks in chunks_by_strategy.items():
            connection.execute("delete from chunks where strategy = ?", (strategy,))
            connection.executemany(
                """
                insert into chunks (
                    strategy, chunk_id, source, title, file_path, section, content_type,
                    start_offset, end_offset, token_count, char_count, embedding_blob,
                    text_content, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.strategy,
                        chunk.chunk_id,
                        chunk.source,
                        chunk.title,
                        chunk.file_path,
                        chunk.section,
                        chunk.content_type,
                        chunk.start_offset,
                        chunk.end_offset,
                        chunk.token_count,
                        chunk.char_count,
                        encode_embedding(chunk.embedding),
                        chunk.text,
                        datetime.now(timezone.utc).isoformat(),
                    )
                    for chunk in chunks
                ],
            )


def encode_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"<{len(embedding)}f", *embedding)


def write_jsonl(path: Path, chunks_by_strategy: dict[str, list[Chunk]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for chunks in chunks_by_strategy.values():
            for chunk in chunks:
                row = asdict(chunk)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
