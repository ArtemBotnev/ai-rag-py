from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rag.inference.llm import ChatCompletion, OllamaChatClient
from rag.qa.query_rewriter import QueryRewriter
from rag.retrieval.retriever import (
    RetrievalConfig,
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
    SqliteRagRetriever,
)


class AnswerMode(str, Enum):
    WITH_RAG = "WITH_RAG"
    WITHOUT_RAG = "WITHOUT_RAG"


@dataclass(frozen=True)
class AgentAnswer:
    mode: AnswerMode
    question: str
    answer: str
    model: str
    completion: ChatCompletion
    retrieved_chunks: list[RetrievedChunk]
    retrieval_duration_ms: int = 0
    retrieval: RetrievalResult | None = None
    extra_completions: list[ChatCompletion] | None = None

    @property
    def total_duration_ms(self) -> int:
        return self.retrieval_duration_ms + self.generation_wall_time_ms

    @property
    def generation_wall_time_ms(self) -> int:
        return self.completion.wall_time_ms + sum(
            completion.wall_time_ms for completion in self.extra_completions or []
        )

    @property
    def ollama_total_duration_ms(self) -> int:
        return self.completion.usage.total_duration_ms + sum(
            completion.usage.total_duration_ms for completion in self.extra_completions or []
        )

    @property
    def prompt_tokens(self) -> int:
        return self.completion.usage.prompt_eval_count + sum(
            completion.usage.prompt_eval_count for completion in self.extra_completions or []
        )

    @property
    def completion_tokens(self) -> int:
        return self.completion.usage.eval_count + sum(
            completion.usage.eval_count for completion in self.extra_completions or []
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RagAgent:
    def __init__(self, chat_client: OllamaChatClient, retriever: SqliteRagRetriever | None = None) -> None:
        self.chat_client = chat_client
        self.retriever = retriever
        self.query_rewriter = QueryRewriter(chat_client)

    def answer_without_rag(self, question: str) -> AgentAnswer:
        completion = self.chat_client.chat(NO_RAG_SYSTEM_PROMPT, question.strip())
        return AgentAnswer(
            mode=AnswerMode.WITHOUT_RAG,
            question=question,
            answer=completion.content,
            model=completion.model,
            completion=completion,
            retrieved_chunks=[],
        )

    def answer_with_rag(self, question: str, top_k: int = 4) -> AgentAnswer:
        return self.answer_with_rag_config(
            question,
            RetrievalConfig(
                mode=RetrievalMode.BASELINE,
                top_k_before=top_k,
                top_k_after=top_k,
                enable_query_rewrite=False,
            ),
        )

    def answer_with_rag_config(self, question: str, retrieval_config: RetrievalConfig) -> AgentAnswer:
        if self.retriever is None:
            raise RuntimeError("RAG mode requires a retriever.")
        rewrite_result = None
        rewritten_query = None
        if retrieval_config.mode == RetrievalMode.IMPROVED and retrieval_config.enable_query_rewrite:
            rewrite_result = self.query_rewriter.rewrite(question)
            rewritten_query = rewrite_result.query if rewrite_result else None
        retrieval = self.retriever.search(question, config=retrieval_config, rewritten_query=rewritten_query)
        completion = self.chat_client.chat(
            WITH_RAG_SYSTEM_PROMPT,
            build_rag_prompt(question, retrieval.chunks, rewritten_query=retrieval.rewritten_query),
        )
        return AgentAnswer(
            mode=AnswerMode.WITH_RAG,
            question=question,
            answer=completion.content,
            model=completion.model,
            completion=completion,
            retrieved_chunks=retrieval.chunks,
            retrieval_duration_ms=retrieval.duration_ms,
            retrieval=retrieval,
            extra_completions=[rewrite_result.completion] if rewrite_result else [],
        )


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    rewritten_query: str | None = None,
) -> str:
    lines = [
        "Question:",
        question.strip(),
        "",
    ]
    if rewritten_query:
        lines.extend(["Retrieval rewrite:", rewritten_query, ""])
    lines.extend(["Retrieved context:", ""])
    if not retrieved_chunks:
        lines.append("No context was retrieved.")
    for index, chunk in enumerate(retrieved_chunks, start=1):
        lines.extend(
            [
                f"[{index}] file={chunk.file_path}",
                f"section={chunk.section}",
                f"score={chunk.score:.4f}",
                f"similarity={chunk.similarity_score:.4f}",
                "content:",
                chunk.text.strip(),
                "",
            ]
        )
    lines.extend(
        [
            "Answer:",
            "Use the retrieved context for repository-specific facts. If the context is insufficient, say so.",
        ]
    )
    return "\n".join(lines)


NO_RAG_SYSTEM_PROMPT = """
You answer questions clearly and concisely.
If you are unsure, say what is uncertain instead of inventing details.
""".strip()

WITH_RAG_SYSTEM_PROMPT = """
You answer questions about the indexed repository.
Use only the retrieved context for repository-specific claims.
If the context does not support part of the answer, state that directly.
Mention the source file paths you relied on in the answer when helpful.
""".strip()
