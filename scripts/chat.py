#!/usr/bin/env python3
from __future__ import annotations

import argparse
import locale
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.inference.embeddings import HashingEmbeddingClient, OllamaEmbeddingClient
from rag.inference.llm import OllamaChatClient
from rag.qa.agent import RagAgent
from rag.qa.chat import (
    ChatSession,
    ChatTurnResult,
    extract_scenario_messages,
    format_task_state,
    load_chat_session,
    save_chat_session,
)
from rag.retrieval.retriever import RetrievalConfig, RetrievalMode, SqliteRagRetriever


def main() -> None:
    args = parse_args()
    try:
        session = build_session(args)
        if args.scenario:
            run_scenario(args, session)
        else:
            run_interactive(args, session)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mini chat with RAG context, dialogue history and task memory.")
    parser.add_argument("--scenario", type=Path, help="Markdown scenario file. Each bullet/numbered item is one user turn.")
    parser.add_argument("--history-file", type=Path, help="Optional JSON file for loading and saving chat history.")
    parser.add_argument("--history-limit", type=int, default=8, help="Recent messages passed to the chat prompt.")
    parser.add_argument("--sqlite-path", type=Path, default=PROJECT_ROOT / "artifacts" / "rag-index.sqlite")
    parser.add_argument("--strategy", default="PARAGRAPH_AWARE", choices=("FIXED_SIZE", "STRUCTURE_AWARE", "PARAGRAPH_AWARE"))
    parser.add_argument("--top-k-before", type=int, default=8)
    parser.add_argument("--top-k-after", type=int, default=4)
    parser.add_argument("--relevance-threshold", type=float, default=0.45)
    parser.add_argument("--query-rewrite", default="on", choices=("on", "off", "true", "false", "yes", "no", "1", "0"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="nomic-embed-text:latest", help="Embedding model for retrieval.")
    parser.add_argument("--ollama-batch-size", type=int, default=16)
    parser.add_argument("--embedding-provider", choices=("ollama", "hashing"), default="ollama")
    parser.add_argument("--llm-model", default=None, help="Ollama chat model. If omitted, the first non-embedding model is used.")
    parser.add_argument("--chat-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.history_limit < 2:
        parser.error("--history-limit must be at least 2")
    if args.top_k_before < 1:
        parser.error("--top-k-before must be positive")
    if args.top_k_after < 1:
        parser.error("--top-k-after must be positive")
    if not 0.0 <= args.relevance_threshold <= 1.0:
        parser.error("--relevance-threshold must be between 0 and 1")
    args.query_rewrite_enabled = parse_bool(args.query_rewrite)
    return args


def build_session(args: argparse.Namespace) -> ChatSession:
    chat_client = OllamaChatClient(
        base_url=args.ollama_url,
        model=args.llm_model,
        timeout_seconds=args.chat_timeout_seconds,
    )
    retriever = SqliteRagRetriever(
        sqlite_path=args.sqlite_path,
        strategy=args.strategy,
        embedding_client=build_embedding_client(args),
    )
    agent = RagAgent(chat_client=chat_client, retriever=retriever)
    retrieval_config = RetrievalConfig(
        mode=RetrievalMode.IMPROVED,
        top_k_before=args.top_k_before,
        top_k_after=args.top_k_after,
        relevance_threshold=args.relevance_threshold,
        enable_query_rewrite=args.query_rewrite_enabled,
    )
    if args.history_file:
        return load_chat_session(
            path=args.history_file,
            agent=agent,
            retrieval_config=retrieval_config,
            history_limit=args.history_limit,
        )
    return ChatSession(agent=agent, retrieval_config=retrieval_config, history_limit=args.history_limit)


def build_embedding_client(args: argparse.Namespace) -> OllamaEmbeddingClient | HashingEmbeddingClient:
    if args.embedding_provider == "hashing":
        return HashingEmbeddingClient()
    return OllamaEmbeddingClient(
        base_url=args.ollama_url,
        model=args.ollama_model,
        batch_size=args.ollama_batch_size,
    )


def run_interactive(args: argparse.Namespace, session: ChatSession) -> None:
    print("Mini RAG chat. Type /exit to stop.")
    while True:
        message = read_console_line("> ")
        if message is None:
            print()
            break
        message = message.strip()
        if message in {"/exit", "/quit"}:
            break
        if not message:
            continue
        result = session.ask(message)
        persist_if_needed(args, session)
        print_turn(result)


def run_scenario(args: argparse.Namespace, session: ChatSession) -> None:
    messages = extract_scenario_messages(args.scenario)
    if not messages:
        raise RuntimeError(f"No scenario messages found in {args.scenario}")
    print(f"Scenario: {args.scenario}")
    print(f"Turns: {len(messages)}")
    for index, message in enumerate(messages, start=1):
        print()
        print(f"=== Turn {index} ===")
        print(f"User: {message}")
        sys.stdout.flush()
        result = session.ask(message)
        persist_if_needed(args, session)
        print_turn(result, show_user=False)


def persist_if_needed(args: argparse.Namespace, session: ChatSession) -> None:
    if args.history_file:
        save_chat_session(session, args.history_file)


def print_turn(result: ChatTurnResult, show_user: bool = True) -> None:
    if show_user:
        print(f"User: {result.user_message}")
    print(format_task_state(result.task_state))
    print("Retrieval query:")
    print(result.retrieval_question)
    if result.answer.retrieval:
        print("Retrieved sources:")
        for index, chunk in enumerate(result.answer.retrieved_chunks, start=1):
            print(
                f"{index}. source={chunk.source}; section={chunk.section}; "
                f"chunk_id={chunk.chunk_id}; score={chunk.score:.4f}"
            )
    print()
    print(result.answer.answer)


def parse_bool(value: str) -> bool:
    return value.lower() in {"on", "true", "yes", "1"}


def read_console_line(prompt: str) -> str | None:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    raw = sys.stdin.buffer.readline()
    if raw == b"":
        return None
    for encoding in input_encodings():
        try:
            return repair_mojibake(raw.decode(encoding)).rstrip("\r\n")
        except UnicodeDecodeError:
            continue
    return repair_mojibake(raw.decode("utf-8", errors="replace")).rstrip("\r\n")


def input_encodings() -> list[str]:
    candidates = [
        "utf-8",
        sys.stdin.encoding,
        locale.getpreferredencoding(False),
        "cp1251",
    ]
    result: list[str] = []
    for encoding in candidates:
        if encoding and encoding not in result:
            result.append(encoding)
    return result


def repair_mojibake(value: str) -> str:
    if not looks_like_cyrillic_mojibake(value):
        return value
    try:
        repaired = value.encode("cp1251").decode("utf-8")
    except UnicodeError:
        return value
    if mojibake_score(repaired) < mojibake_score(value):
        return repaired
    return value


def looks_like_cyrillic_mojibake(value: str) -> bool:
    markers = ("Р", "СЃ", "С‚", "СЂ", "Рё", "Р°", "РЅ", "Рє")
    return sum(value.count(marker) for marker in markers) >= 3


def mojibake_score(value: str) -> int:
    markers = ("Р", "СЃ", "С‚", "СЂ", "Рё", "Р°", "РЅ", "Рє", "Рџ", "Рќ")
    return sum(value.count(marker) for marker in markers)


if __name__ == "__main__":
    main()
