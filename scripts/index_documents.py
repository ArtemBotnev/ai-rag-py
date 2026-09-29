#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.cli import build_embedding_client, parse_args
from rag.indexing.pipeline import run_pipeline


def main() -> None:
    args = parse_args()
    embedding_client = build_embedding_client(args)
    try:
        report = run_pipeline(args, embedding_client)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"SQLite index: {report['sqlitePath']}")
    print(f"JSONL chunks: {report['jsonlPath']}")
    for summary in report["strategySummaries"]:
        print(
            f"{summary['strategy']}: chunks={summary['chunkCount']}, "
            f"avgTokens={summary['averageTokens']:.1f}, files={summary['uniqueFiles']}, "
            f"<500={summary['chunksUnder500Tokens']}, "
            f"500..1000={summary['chunksBetween500And1000Tokens']}, "
            f">1000={summary['chunksOver1000Tokens']}"
        )


if __name__ == "__main__":
    main()
