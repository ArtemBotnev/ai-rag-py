from __future__ import annotations

import argparse
from pathlib import Path

from rag.embeddings import HashingEmbeddingClient, OllamaEmbeddingClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index source documents with three chunking strategies.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "source", help="Directory with source documents.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts", help="Directory for index and reports.")
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "config" / "sample_queries.txt", help="Sample queries file for retrieval comparison.")
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--target-tokens", type=int, default=800)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--overlap-tokens", type=int, default=80)
    parser.add_argument("--embedding-provider", choices=("ollama", "hashing"), default="ollama")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="nomic-embed-text:latest")
    parser.add_argument("--ollama-batch-size", type=int, default=16)
    return parser.parse_args()


def build_embedding_client(args: argparse.Namespace) -> OllamaEmbeddingClient | HashingEmbeddingClient:
    if args.embedding_provider == "hashing":
        return HashingEmbeddingClient()
    return OllamaEmbeddingClient(
        base_url=args.ollama_url,
        model=args.ollama_model,
        batch_size=args.ollama_batch_size,
    )
