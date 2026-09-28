from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median

from rag.embeddings import EmbeddingClient, cosine_similarity
from rag.models import Chunk, SourceDocument


def summarize(strategy: str, documents: list[SourceDocument], chunks: list[Chunk]) -> dict[str, object]:
    token_counts = sorted(chunk.token_count for chunk in chunks)
    char_counts = [chunk.char_count for chunk in chunks]
    unique_files = len({chunk.file_path for chunk in chunks})
    return {
        "strategy": strategy,
        "documentCount": len(documents),
        "chunkCount": len(chunks),
        "averageTokens": sum(token_counts) / len(token_counts),
        "medianTokens": int(median(token_counts)),
        "minTokens": token_counts[0],
        "maxTokens": token_counts[-1],
        "averageChars": sum(char_counts) / len(char_counts),
        "chunksUnder500Tokens": sum(1 for value in token_counts if value < 500),
        "chunksBetween500And1000Tokens": sum(1 for value in token_counts if 500 <= value <= 1000),
        "chunksOver1000Tokens": sum(1 for value in token_counts if value > 1000),
        "uniqueFiles": unique_files,
        "averageChunksPerFile": len(chunks) / max(1, unique_files),
    }


def build_query_comparisons(
    chunks_by_strategy: dict[str, list[Chunk]],
    queries: list[str],
    embedding_client: EmbeddingClient,
) -> list[dict[str, object]]:
    query_embeddings = embedding_client.embed_all(queries)
    comparisons: list[dict[str, object]] = []
    for query, query_embedding in zip(queries, query_embeddings):
        strategy_hits = []
        for strategy, chunks in chunks_by_strategy.items():
            hits = sorted(
                (
                    {
                        "chunkId": chunk.chunk_id,
                        "filePath": chunk.file_path,
                        "section": chunk.section,
                        "score": cosine_similarity(query_embedding, chunk.embedding),
                        "preview": re.sub(r"\s+", " ", chunk.text).strip()[:180],
                    }
                    for chunk in chunks
                ),
                key=lambda hit: hit["score"],
                reverse=True,
            )[:3]
            strategy_hits.append({"strategy": strategy, "hits": hits})
        comparisons.append({"query": query, "strategyHits": strategy_hits})
    return comparisons


def write_reports(report: dict[str, object], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")


def report_to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Chunking Comparison",
        "",
        f"- Source root: `{report['sourceRoot']}`",
        f"- SQLite index: `{report['sqlitePath']}`",
        f"- JSONL chunks: `{report['jsonlPath']}`",
        f"- Sample queries: `{report['queriesPath']}`",
        f"- Embedding provider: `{report['embeddingProvider']}`",
        f"- Embedding model: `{report['embeddingModel']}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Documents | Files | Chunks | Avg tokens | Median tokens | Min | Max | <500 | 500-1000 | >1000 | Avg chars | Chunks/file |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in report["strategySummaries"]:
        lines.append(
            f"| {summary['strategy']} | {summary['documentCount']} | {summary['uniqueFiles']} | "
            f"{summary['chunkCount']} | {summary['averageTokens']:.1f} | {summary['medianTokens']} | "
            f"{summary['minTokens']} | {summary['maxTokens']} | {summary['chunksUnder500Tokens']} | "
            f"{summary['chunksBetween500And1000Tokens']} | {summary['chunksOver1000Tokens']} | "
            f"{summary['averageChars']:.1f} | {summary['averageChunksPerFile']:.2f} |"
        )
    lines.extend(["", "## Comparison Notes", ""])
    summaries = {summary["strategy"]: summary for summary in report["strategySummaries"]}
    if "FIXED_SIZE" in summaries:
        fixed = summaries["FIXED_SIZE"]
        lines.append(
            f"- `FIXED_SIZE` is the baseline with predictable windows: "
            f"{fixed['chunksBetween500And1000Tokens']} chunks are in the 500-1000 token range."
        )
    if "STRUCTURE_AWARE" in summaries:
        structure = summaries["STRUCTURE_AWARE"]
        lines.append(
            f"- `STRUCTURE_AWARE` keeps Markdown sections and Kotlin declarations in `section`, "
            f"but produces {structure['chunksUnder500Tokens']} short chunks on this corpus."
        )
    if "PARAGRAPH_AWARE" in summaries:
        paragraph = summaries["PARAGRAPH_AWARE"]
        lines.append(
            f"- `PARAGRAPH_AWARE` preserves natural text flow and stays close to the target size "
            f"with average {paragraph['averageTokens']:.1f} tokens per chunk."
        )
    lines.extend(
        [
            "",
            "## Timings",
            "",
            "| Stage | Duration ms |",
            "| --- | ---: |",
        ]
    )
    for stage, duration in report["timings"].items():
        lines.append(f"| {stage} | {duration} |")
    lines.extend(["", "## Retrieval Samples", ""])
    for comparison in report["queryComparisons"]:
        lines.extend([f"### {comparison['query']}", ""])
        for strategy_hits in comparison["strategyHits"]:
            lines.extend([f"**{strategy_hits['strategy'].lower().replace('_', '-')}**", ""])
            for hit in strategy_hits["hits"]:
                lines.append(f"- `{hit['filePath']}` / `{hit['section']}` / score={hit['score']:.4f}")
                lines.append(f"  - {hit['preview']}")
            lines.append("")
    return "\n".join(lines)
