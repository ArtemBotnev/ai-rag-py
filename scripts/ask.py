#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.qa.agent import AgentAnswer, AnswerMode, RagAgent
from rag.evaluation.control_questions import load_control_questions
from rag.inference.embeddings import HashingEmbeddingClient, OllamaEmbeddingClient
from rag.evaluation.evaluation import (
    RagQualityEvaluator,
    format_duration,
    write_evaluation_report,
    write_improved_evaluation_report,
)
from rag.inference.llm import OllamaChatClient
from rag.retrieval.retriever import RetrievalConfig, RetrievalMode, SqliteRagRetriever


def main() -> None:
    args = parse_args()
    try:
        if args.control:
            run_control(args)
        elif args.control_improved:
            run_control_improved(args)
        elif args.compare:
            run_compare(args)
        else:
            run_ask(args)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions with or without RAG over the local SQLite index.")
    parser.add_argument("question", nargs="?", help="Question for single ask or compare mode.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rag", action="store_true", help="Answer using retrieved chunks.")
    mode.add_argument("--no-rag", action="store_true", help="Answer without retrieval.")
    mode.add_argument("--compare", action="store_true", help="Run no-RAG, baseline RAG and improved RAG for one question.")
    mode.add_argument("--control", action="store_true", help="Run no-RAG, baseline RAG and improved RAG for all control questions.")
    mode.add_argument("--control-improved", action="store_true", help="Run improved RAG only for all control questions.")
    parser.add_argument("--sqlite-path", type=Path, default=PROJECT_ROOT / "artifacts" / "rag-index.sqlite")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--control-questions", type=Path, default=PROJECT_ROOT / "CONTROL_QUESTIONS.md")
    parser.add_argument("--strategy", default="PARAGRAPH_AWARE", choices=("FIXED_SIZE", "STRUCTURE_AWARE", "PARAGRAPH_AWARE"))
    parser.add_argument("--retrieval-mode", default="IMPROVED", choices=("BASELINE", "IMPROVED"))
    parser.add_argument("--top-k", type=int, default=None, help="Alias for --top-k-after.")
    parser.add_argument("--top-k-before", type=int, default=None, help="Candidate count before filtering in improved mode.")
    parser.add_argument("--top-k-after", type=int, default=None, help="Final context chunk count after filtering.")
    parser.add_argument("--relevance-threshold", type=float, default=0.45)
    parser.add_argument("--query-rewrite", default="on", choices=("on", "off", "true", "false", "yes", "no", "1", "0"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="nomic-embed-text:latest", help="Embedding model for retrieval.")
    parser.add_argument("--ollama-batch-size", type=int, default=16)
    parser.add_argument("--embedding-provider", choices=("ollama", "hashing"), default="ollama")
    parser.add_argument("--llm-model", default=None, help="Ollama chat model. If omitted, the first non-embedding model is used.")
    parser.add_argument("--chat-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if not args.control and not args.control_improved and not args.question:
        parser.error("question is required unless --control or --control-improved is used")
    if not args.rag and not args.no_rag and not args.compare and not args.control and not args.control_improved:
        args.compare = True
    args.top_k_after = args.top_k_after if args.top_k_after is not None else (args.top_k if args.top_k is not None else 4)
    args.top_k_before = args.top_k_before if args.top_k_before is not None else max(4, args.top_k_after) * 2
    args.query_rewrite_enabled = parse_bool(args.query_rewrite)
    if args.top_k_after < 1:
        parser.error("--top-k-after/--top-k must be positive")
    if args.top_k_before < 1:
        parser.error("--top-k-before must be positive")
    if not 0.0 <= args.relevance_threshold <= 1.0:
        parser.error("--relevance-threshold must be between 0 and 1")
    return args


def run_ask(args: argparse.Namespace) -> None:
    use_rag = args.rag or not args.no_rag
    agent = build_agent(args, require_retriever=use_rag)
    answer = (
        agent.answer_with_rag_config(args.question, active_retrieval_config(args))
        if use_rag
        else agent.answer_without_rag(args.question)
    )
    print_answer(answer)


def run_compare(args: argparse.Namespace) -> None:
    agent = build_agent(args, require_retriever=True)
    without_rag = agent.answer_without_rag(args.question)
    baseline_rag = agent.answer_with_rag_config(args.question, baseline_retrieval_config(args))
    improved_rag = agent.answer_with_rag_config(args.question, improved_retrieval_config(args))
    print("=== WITHOUT RAG ===")
    print_answer(without_rag)
    print()
    print("=== BASELINE RAG ===")
    print_answer(baseline_rag)
    print()
    print("=== IMPROVED RAG ===")
    print_answer(improved_rag)


def run_control(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_control_questions(args.control_questions)
    agent = build_agent(args, require_retriever=True)
    report = RagQualityEvaluator(
        agent=agent,
        sqlite_path=args.sqlite_path,
        retrieval_strategy=args.strategy,
        baseline_config=baseline_retrieval_config(args),
        improved_config=improved_retrieval_config(args),
    ).evaluate(questions)
    json_path = args.output_dir / "rag-quality-comparison.json"
    markdown_path = args.output_dir / "rag-quality-comparison.md"
    write_evaluation_report(report, json_path, markdown_path)
    print(f"Evaluation report (md): {markdown_path.resolve()}")
    print(f"Evaluation report (json): {json_path.resolve()}")
    print(f"Baseline RAG wins: {report.summary.baseline_rag_wins}")
    print(f"Improved RAG wins: {report.summary.improved_rag_wins}")
    print(f"Without RAG wins: {report.summary.without_rag_wins}")
    print(f"Ties: {report.summary.ties}")


def run_control_improved(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_control_questions(args.control_questions)
    agent = build_agent(args, require_retriever=True)
    report = RagQualityEvaluator(
        agent=agent,
        sqlite_path=args.sqlite_path,
        retrieval_strategy=args.strategy,
        baseline_config=baseline_retrieval_config(args),
        improved_config=improved_retrieval_config(args),
    ).evaluate_improved_only(questions)
    json_path = args.output_dir / "rag-quality-improved.json"
    markdown_path = args.output_dir / "rag-quality-improved.md"
    write_improved_evaluation_report(report, json_path, markdown_path)
    print(f"Improved RAG report (md): {markdown_path.resolve()}")
    print(f"Improved RAG report (json): {json_path.resolve()}")
    print(f"Average coverage: {report.summary.average_improved_rag_coverage:.2f}")
    print(f"Average source coverage: {report.summary.average_improved_rag_source_coverage:.2f}")
    print(f"Average time: {format_duration(report.summary.average_improved_rag_time_ms)}")
    print(f"Average tokens: {report.summary.average_improved_rag_tokens:.1f}")


def build_agent(args: argparse.Namespace, require_retriever: bool) -> RagAgent:
    chat_client = OllamaChatClient(
        base_url=args.ollama_url,
        model=args.llm_model,
        timeout_seconds=args.chat_timeout_seconds,
    )
    retriever = None
    if require_retriever:
        retriever = SqliteRagRetriever(
            sqlite_path=args.sqlite_path,
            strategy=args.strategy,
            embedding_client=build_embedding_client(args),
        )
    return RagAgent(chat_client=chat_client, retriever=retriever)


def build_embedding_client(args: argparse.Namespace) -> OllamaEmbeddingClient | HashingEmbeddingClient:
    if args.embedding_provider == "hashing":
        return HashingEmbeddingClient()
    return OllamaEmbeddingClient(
        base_url=args.ollama_url,
        model=args.ollama_model,
        batch_size=args.ollama_batch_size,
    )


def active_retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    mode = RetrievalMode(args.retrieval_mode)
    if mode == RetrievalMode.BASELINE:
        return baseline_retrieval_config(args)
    return improved_retrieval_config(args)


def baseline_retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        mode=RetrievalMode.BASELINE,
        top_k_before=args.top_k_after,
        top_k_after=args.top_k_after,
        relevance_threshold=0.0,
        enable_query_rewrite=False,
    )


def improved_retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        mode=RetrievalMode.IMPROVED,
        top_k_before=args.top_k_before,
        top_k_after=args.top_k_after,
        relevance_threshold=args.relevance_threshold,
        enable_query_rewrite=args.query_rewrite_enabled,
    )


def parse_bool(value: str) -> bool:
    return value.lower() in {"on", "true", "yes", "1"}


def print_answer(answer: AgentAnswer) -> None:
    print(f"Mode: {answer.mode.value}")
    print(f"Question: {answer.question}")
    print(f"Model: {answer.model}")
    print(
        "Metrics: "
        f"total={format_duration(answer.total_duration_ms)}, "
        f"generation={format_duration(answer.generation_wall_time_ms)}, "
        f"retrieval={format_duration(answer.retrieval_duration_ms)}, "
        f"tokens={answer.total_tokens} "
        f"(prompt={answer.prompt_tokens}, completion={answer.completion_tokens})"
    )
    if answer.mode == AnswerMode.WITH_RAG:
        if answer.retrieval:
            print(f"Retrieval mode: {answer.retrieval.mode.value}")
            if answer.retrieval.rewritten_query:
                print(f"Rewritten query: {answer.retrieval.rewritten_query}")
            print(
                "Retrieval config: "
                f"top_k_before={answer.retrieval.top_k_before}, "
                f"top_k_after={answer.retrieval.top_k_after}, "
                f"threshold={answer.retrieval.relevance_threshold}"
            )
        print("Retrieved chunks:")
        for index, chunk in enumerate(answer.retrieved_chunks, start=1):
            print(
                f"{index}. source={chunk.source} / file={chunk.file_path} / "
                f"section={chunk.section} / chunk_id={chunk.chunk_id} / "
                f"score={chunk.score:.4f} / similarity={chunk.similarity_score:.4f}"
            )
    print()
    print(answer.answer)


if __name__ == "__main__":
    main()
