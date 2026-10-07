#!/usr/bin/env python3
from __future__ import annotations

import argparse
import locale
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.inference.embeddings import HashingEmbeddingClient, OllamaEmbeddingClient
from rag.inference.llm import ChatCompletion, OllamaChatClient, OpenAIResponsesChatClient
from rag.qa.chat import extract_scenario_messages
from rag.retrieval.retriever import RetrievalConfig, RetrievalMode, RetrievalResult, SqliteRagRetriever


class ChatClient(Protocol):
    def chat(self, system_prompt: str | None, user_prompt: str) -> ChatCompletion:
        ...


@dataclass(frozen=True)
class ComparedAnswer:
    label: str
    model: str
    answer: str
    duration_ms: int
    error: str | None = None


def main() -> None:
    args = parse_args()
    try:
        retriever = build_retriever(args)
        openai_client = OpenAIResponsesChatClient(
            model=args.openai_model,
            base_url=args.openai_url,
            timeout_seconds=args.chat_timeout_seconds,
        )
        ollama_client = OllamaChatClient(
            base_url=args.ollama_url,
            model=args.llm_model,
            timeout_seconds=args.chat_timeout_seconds,
        )
        if args.scenario:
            run_scenario(args, retriever, openai_client, ollama_client)
        elif args.question:
            run_turn(args.question, args, retriever, openai_client, ollama_client)
        else:
            run_interactive(args, retriever, openai_client, ollama_client)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OpenAI and local Ollama answers over the same RAG context.")
    parser.add_argument("question", nargs="?", help="Optional one-shot question. If omitted, starts interactive chat.")
    parser.add_argument("--scenario", type=Path, help="Markdown scenario file. Each bullet/numbered item is one user turn.")
    parser.add_argument("--sqlite-path", type=Path, default=PROJECT_ROOT / "artifacts" / "rag-index.sqlite")
    parser.add_argument("--strategy", default="PARAGRAPH_AWARE", choices=("FIXED_SIZE", "STRUCTURE_AWARE", "PARAGRAPH_AWARE"))
    parser.add_argument("--top-k-before", type=int, default=8)
    parser.add_argument("--top-k-after", type=int, default=4)
    parser.add_argument("--relevance-threshold", type=float, default=0.45)
    parser.add_argument("--retrieval-mode", default="IMPROVED", choices=("BASELINE", "IMPROVED"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="nomic-embed-text:latest", help="Embedding model for retrieval.")
    parser.add_argument("--ollama-batch-size", type=int, default=16)
    parser.add_argument("--embedding-provider", choices=("ollama", "hashing"), default="ollama")
    parser.add_argument("--llm-model", default=None, help="Local Ollama chat model. If omitted, first non-embedding model is used.")
    parser.add_argument("--openai-url", default="https://api.openai.com/v1")
    parser.add_argument("--openai-model", default=None, help="OpenAI model. Defaults to OPENAI_MODEL or gpt-5-mini.")
    parser.add_argument("--chat-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.top_k_before < 1:
        parser.error("--top-k-before must be positive")
    if args.top_k_after < 1:
        parser.error("--top-k-after must be positive")
    if not 0.0 <= args.relevance_threshold <= 1.0:
        parser.error("--relevance-threshold must be between 0 and 1")
    if args.scenario and args.question:
        parser.error("question cannot be used together with --scenario")
    return args


def build_retriever(args: argparse.Namespace) -> SqliteRagRetriever:
    return SqliteRagRetriever(
        sqlite_path=args.sqlite_path,
        strategy=args.strategy,
        embedding_client=build_embedding_client(args),
    )


def build_embedding_client(args: argparse.Namespace) -> OllamaEmbeddingClient | HashingEmbeddingClient:
    if args.embedding_provider == "hashing":
        return HashingEmbeddingClient()
    return OllamaEmbeddingClient(
        base_url=args.ollama_url,
        model=args.ollama_model,
        batch_size=args.ollama_batch_size,
    )


def run_interactive(
    args: argparse.Namespace,
    retriever: SqliteRagRetriever,
    openai_client: ChatClient,
    ollama_client: ChatClient,
) -> None:
    print("LLM compare RAG chat. Type /exit to stop.")
    while True:
        message = read_console_line("\n> ")
        if message is None:
            print()
            break
        message = message.strip()
        if message in {"/exit", "/quit"}:
            break
        if not message:
            continue
        print()
        run_turn(message, args, retriever, openai_client, ollama_client)


def run_scenario(
    args: argparse.Namespace,
    retriever: SqliteRagRetriever,
    openai_client: ChatClient,
    ollama_client: ChatClient,
) -> None:
    messages = extract_scenario_messages(args.scenario)
    if not messages:
        raise RuntimeError(f"No scenario messages found in {args.scenario}")
    for message in messages:
        print()
        print(f"> {message}")
        print()
        run_turn(message, args, retriever, openai_client, ollama_client)


def run_turn(
    question: str,
    args: argparse.Namespace,
    retriever: SqliteRagRetriever,
    openai_client: ChatClient,
    ollama_client: ChatClient,
) -> None:
    question = question.strip()
    if not question:
        raise RuntimeError("Question must not be blank.")
    retrieval = retriever.search(question, config=retrieval_config(args))
    prompt = build_prompt(question, retrieval)
    answers = [
        ask_model("OpenAI", openai_client, prompt),
        ask_model("Ollama", ollama_client, prompt),
    ]
    for index, answer in enumerate(answers):
        if index:
            print()
        print_compared_answer(answer)


def retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    mode = RetrievalMode(args.retrieval_mode)
    if mode == RetrievalMode.BASELINE:
        return RetrievalConfig(
            mode=RetrievalMode.BASELINE,
            top_k_before=args.top_k_after,
            top_k_after=args.top_k_after,
            relevance_threshold=0.0,
            enable_query_rewrite=False,
        )
    return RetrievalConfig(
        mode=RetrievalMode.IMPROVED,
        top_k_before=args.top_k_before,
        top_k_after=args.top_k_after,
        relevance_threshold=args.relevance_threshold,
        enable_query_rewrite=False,
    )


def ask_model(label: str, client: ChatClient, prompt: str) -> ComparedAnswer:
    started = time.perf_counter()
    try:
        completion = client.chat(COMPARE_SYSTEM_PROMPT, prompt)
    except RuntimeError as exc:
        return ComparedAnswer(
            label=label,
            model="unknown",
            answer="",
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=str(exc),
        )
    return ComparedAnswer(
        label=label,
        model=completion.model,
        answer=completion.content,
        duration_ms=completion.wall_time_ms or int((time.perf_counter() - started) * 1000),
    )


def build_prompt(question: str, retrieval: RetrievalResult) -> str:
    lines = [
        "Question:",
        question.strip(),
        "",
    ]
    lines.append("Retrieved context:")
    if not retrieval.chunks:
        lines.append("No context was retrieved above the relevance threshold.")
    for index, chunk in enumerate(retrieval.chunks, start=1):
        lines.extend(
            [
                f"[{index}] source={chunk.source}",
                f"section={chunk.section}",
                "content:",
                chunk.text.strip(),
                "",
            ]
        )
    lines.extend(
        [
            "",
            "Answer using only the retrieved context. If the context is insufficient, say \"не знаю\".",
        ]
    )
    return "\n".join(lines)


def print_compared_answer(answer: ComparedAnswer) -> None:
    print(f"{answer.label} ({answer.model})")
    print(f"Time: {answer.duration_ms} ms")
    if answer.error:
        print(f"Error: {answer.error}")
    else:
        print(answer.answer)


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


COMPARE_SYSTEM_PROMPT = """
You answer questions about the indexed repository.
Use only the retrieved context for repository-specific claims.
Do not print sources, citations, retrieval diagnostics, token counts, or implementation details.
If the context does not support the answer, say "не знаю".
Answer in the user's language.
""".strip()


if __name__ == "__main__":
    main()
