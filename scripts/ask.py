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
from rag.evaluation.evaluation import RagQualityEvaluator, format_duration, write_evaluation_report
from rag.inference.llm import OllamaChatClient
from rag.retrieval.retriever import SqliteRagRetriever


def main() -> None:
    args = parse_args()
    try:
        if args.control:
            run_control(args)
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
    mode.add_argument("--compare", action="store_true", help="Run both no-RAG and RAG modes for one question.")
    mode.add_argument("--control", action="store_true", help="Run both modes for all control questions.")
    parser.add_argument("--sqlite-path", type=Path, default=PROJECT_ROOT / "artifacts" / "rag-index.sqlite")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--control-questions", type=Path, default=PROJECT_ROOT / "CONTROL_QUESTIONS.md")
    parser.add_argument("--strategy", default="PARAGRAPH_AWARE", choices=("FIXED_SIZE", "STRUCTURE_AWARE", "PARAGRAPH_AWARE"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="nomic-embed-text:latest", help="Embedding model for retrieval.")
    parser.add_argument("--ollama-batch-size", type=int, default=16)
    parser.add_argument("--embedding-provider", choices=("ollama", "hashing"), default="ollama")
    parser.add_argument("--llm-model", default=None, help="Ollama chat model. If omitted, the first non-embedding model is used.")
    parser.add_argument("--chat-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if not args.control and not args.question:
        parser.error("question is required unless --control is used")
    if not args.rag and not args.no_rag and not args.compare and not args.control:
        args.compare = True
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    return args


def run_ask(args: argparse.Namespace) -> None:
    use_rag = args.rag or not args.no_rag
    agent = build_agent(args, require_retriever=use_rag)
    answer = (
        agent.answer_with_rag(args.question, top_k=args.top_k)
        if use_rag
        else agent.answer_without_rag(args.question)
    )
    print_answer(answer)


def run_compare(args: argparse.Namespace) -> None:
    agent = build_agent(args, require_retriever=True)
    without_rag = agent.answer_without_rag(args.question)
    with_rag = agent.answer_with_rag(args.question, top_k=args.top_k)
    print("=== WITHOUT RAG ===")
    print_answer(without_rag)
    print()
    print("=== WITH RAG ===")
    print_answer(with_rag)


def run_control(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_control_questions(args.control_questions)
    agent = build_agent(args, require_retriever=True)
    report = RagQualityEvaluator(
        agent=agent,
        sqlite_path=args.sqlite_path,
        retrieval_strategy=args.strategy,
        top_k=args.top_k,
    ).evaluate(questions)
    json_path = args.output_dir / "rag-quality-comparison.json"
    markdown_path = args.output_dir / "rag-quality-comparison.md"
    write_evaluation_report(report, json_path, markdown_path)
    print(f"Evaluation report (md): {markdown_path.resolve()}")
    print(f"Evaluation report (json): {json_path.resolve()}")
    print(f"With RAG wins: {report.summary.with_rag_wins}")
    print(f"Without RAG wins: {report.summary.without_rag_wins}")
    print(f"Ties: {report.summary.ties}")


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


def print_answer(answer: AgentAnswer) -> None:
    print(f"Mode: {answer.mode.value}")
    print(f"Question: {answer.question}")
    print(f"Model: {answer.model}")
    print(
        "Metrics: "
        f"total={format_duration(answer.total_duration_ms)}, "
        f"generation={format_duration(answer.completion.wall_time_ms)}, "
        f"retrieval={format_duration(answer.retrieval_duration_ms)}, "
        f"tokens={answer.completion.usage.total_tokens} "
        f"(prompt={answer.completion.usage.prompt_eval_count}, completion={answer.completion.usage.eval_count})"
    )
    if answer.mode == AnswerMode.WITH_RAG:
        print("Retrieved chunks:")
        for index, chunk in enumerate(answer.retrieved_chunks, start=1):
            print(f"{index}. {chunk.file_path} / {chunk.section} / score={chunk.score:.4f}")
    print()
    print(answer.answer)


if __name__ == "__main__":
    main()
