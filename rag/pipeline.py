from __future__ import annotations

import argparse
import time
from dataclasses import replace

from rag.chunkers import FixedSizeChunker, ParagraphAwareChunker, StructureAwareChunker
from rag.document_loader import load_documents
from rag.embeddings import EmbeddingClient
from rag.models import Chunk, ChunkingConfig
from rag.reporter import build_query_comparisons, summarize, write_reports
from rag.storage import write_jsonl, write_sqlite


def run_pipeline(args: argparse.Namespace, embedding_client: EmbeddingClient) -> dict[str, object]:
    total_started = time.perf_counter()
    source_root = args.source.resolve()
    artifacts_dir = args.output.resolve()
    sqlite_path = artifacts_dir / "rag-index.sqlite"
    jsonl_path = artifacts_dir / "rag-chunks.jsonl"
    report_json_path = artifacts_dir / "chunking-comparison.json"
    report_markdown_path = artifacts_dir / "chunking-comparison.md"
    config = ChunkingConfig(args.min_tokens, args.target_tokens, args.max_tokens, args.overlap_tokens)

    started = time.perf_counter()
    documents = load_documents(source_root)
    document_loading_ms = elapsed_ms(started)
    if not documents:
        raise RuntimeError(f"No supported documents found under {source_root}")

    chunkers = [FixedSizeChunker(config), StructureAwareChunker(config), ParagraphAwareChunker(config)]
    chunks_by_strategy: dict[str, list[Chunk]] = {}
    timings: dict[str, int] = {"Document loading": document_loading_ms}

    for chunker in chunkers:
        started = time.perf_counter()
        chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
        chunking_ms = elapsed_ms(started)
        timings[f"{chunker.strategy} chunking"] = chunking_ms
        print(
            f"{chunker.strategy}: prepared {len(chunks)} chunks in {chunking_ms} ms "
            f"{chunk_range_summary(chunks)}"
        )

        started = time.perf_counter()
        embeddings = embedding_client.embed_all(chunk.text for chunk in chunks)
        chunks_by_strategy[chunker.strategy] = [
            replace(chunk, embedding=embedding) for chunk, embedding in zip(chunks, embeddings)
        ]
        embedding_ms = elapsed_ms(started)
        timings[f"{chunker.strategy} embeddings"] = embedding_ms
        print(f"{chunker.strategy}: received {len(embeddings)} embeddings in {embedding_ms} ms")

    started = time.perf_counter()
    write_sqlite(sqlite_path, chunks_by_strategy)
    timings["SQLite write"] = elapsed_ms(started)

    started = time.perf_counter()
    write_jsonl(jsonl_path, chunks_by_strategy)
    timings["JSONL write"] = elapsed_ms(started)

    started = time.perf_counter()
    queries = load_sample_queries(args.queries)
    query_comparisons = build_query_comparisons(chunks_by_strategy, queries, embedding_client)
    timings["Query embeddings"] = elapsed_ms(started)

    report = {
        "sourceRoot": str(source_root),
        "sqlitePath": str(sqlite_path),
        "jsonlPath": str(jsonl_path),
        "queriesPath": str(args.queries.resolve()),
        "embeddingProvider": args.embedding_provider,
        "embeddingModel": args.ollama_model if args.embedding_provider == "ollama" else "hashing",
        "strategySummaries": [
            summarize(strategy, documents, chunks) for strategy, chunks in chunks_by_strategy.items()
        ],
        "queryComparisons": query_comparisons,
        "timings": timings,
    }

    started = time.perf_counter()
    write_reports(report, report_json_path, report_markdown_path)
    timings["Report write"] = elapsed_ms(started)
    timings["Total"] = elapsed_ms(total_started)
    write_reports(report, report_json_path, report_markdown_path)
    return report


def load_sample_queries(path) -> list[str]:
    queries_path = path.resolve()
    if not queries_path.exists():
        raise RuntimeError(f"Sample queries file not found: {queries_path}")
    queries = [
        line.strip()
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not queries:
        raise RuntimeError(f"Sample queries file is empty: {queries_path}")
    return queries


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def chunk_range_summary(chunks: list[Chunk]) -> str:
    return (
        f"[<500={sum(1 for chunk in chunks if chunk.token_count < 500)}, "
        f"500..1000={sum(1 for chunk in chunks if 500 <= chunk.token_count <= 1000)}, "
        f">1000={sum(1 for chunk in chunks if chunk.token_count > 1000)}]"
    )
