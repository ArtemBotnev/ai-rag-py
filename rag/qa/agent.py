from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rag.inference.llm import ChatCompletion, ChatUsage, OllamaChatClient
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

    def answer_with_rag_config(
        self,
        question: str,
        retrieval_config: RetrievalConfig,
        retrieval_question: str | None = None,
        prompt_context: str | None = None,
    ) -> AgentAnswer:
        if self.retriever is None:
            raise RuntimeError("RAG mode requires a retriever.")
        search_question = (retrieval_question or question).strip()
        rewrite_result = None
        rewritten_query = None
        if retrieval_config.mode == RetrievalMode.IMPROVED and retrieval_config.enable_query_rewrite:
            rewrite_result = self.query_rewriter.rewrite(search_question)
            rewritten_query = rewrite_result.query if rewrite_result else None
        retrieval = self.retriever.search(search_question, config=retrieval_config, rewritten_query=rewritten_query)
        if should_decline_for_low_relevance(retrieval):
            completion = build_low_relevance_completion(self.chat_client, retrieval)
        else:
            completion = self.chat_client.chat(
                WITH_RAG_SYSTEM_PROMPT,
                build_rag_prompt(question, retrieval, prompt_context=prompt_context),
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
    retrieval: RetrievalResult,
    prompt_context: str | None = None,
) -> str:
    lines = [
        "Question:",
        question.strip(),
        "",
    ]
    if prompt_context:
        lines.extend(["Conversation and task state:", prompt_context.strip(), ""])
    if retrieval.rewritten_query:
        lines.extend(["Retrieval rewrite:", retrieval.rewritten_query, ""])
    if retrieval.relevance_threshold is not None:
        lines.extend(["Relevance threshold:", f"{retrieval.relevance_threshold:.4f}", ""])
    lines.extend(["Retrieved context:", ""])
    if not retrieval.chunks:
        lines.append("No context was retrieved.")
    for index, chunk in enumerate(retrieval.chunks, start=1):
        lines.extend(
            [
                f"[{index}] source={chunk.source}",
                f"file={chunk.file_path}",
                f"section={chunk.section}",
                f"chunk_id={chunk.chunk_id}",
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
            "Use the retrieved context for repository-specific facts.",
            "Return exactly these three sections:",
            "Ответ:",
            "<direct answer in the user's language>",
            "",
            "Источники:",
            "- source=<source>; section=<section>; chunk_id=<chunk_id>",
            "",
            "Цитаты:",
            "- [<chunk_id>] \"<short exact fragment from the retrieved chunk>\"",
            "",
            "Every source must include source, section and chunk_id.",
            "Every citation must be a short exact fragment copied from a retrieved chunk.",
            "If the retrieved context does not support the answer, say \"не знаю\" and ask the user to clarify.",
            "If no retrieved chunk is above the relevance threshold, say \"не знаю\" and ask the user to clarify.",
        ]
    )
    return "\n".join(lines)


def should_decline_for_low_relevance(retrieval: RetrievalResult) -> bool:
    return retrieval.relevance_threshold is not None and not retrieval.chunks


def build_low_relevance_completion(chat_client: OllamaChatClient, retrieval: RetrievalResult) -> ChatCompletion:
    threshold = retrieval.relevance_threshold if retrieval.relevance_threshold is not None else 0.0
    best_score = max((chunk.score for chunk in retrieval.candidate_chunks), default=0.0)
    content = "\n".join(
        [
            "Ответ:",
            (
                "не знаю. В найденных фрагментах нет источника с релевантностью выше порога "
                f"{threshold:.2f} (лучший score={best_score:.4f}). Уточните вопрос или добавьте больше деталей."
            ),
            "",
            "Источники:",
            "- нет релевантных источников",
            "",
            "Цитаты:",
            "- нет релевантных цитат",
        ]
    )
    return ChatCompletion(
        model=synthetic_model_name(chat_client),
        content=content,
        usage=ChatUsage(
            total_duration_ms=0,
            load_duration_ms=0,
            prompt_eval_count=0,
            prompt_eval_duration_ms=0,
            eval_count=0,
            eval_duration_ms=0,
        ),
        wall_time_ms=0,
    )


def synthetic_model_name(chat_client: OllamaChatClient) -> str:
    return (
        getattr(chat_client, "configured_model", None)
        or getattr(chat_client, "_model", None)
        or "rag-low-relevance-gate"
    )


NO_RAG_SYSTEM_PROMPT = """
You answer questions clearly and concisely.
If you are unsure, say what is uncertain instead of inventing details.
""".strip()

WITH_RAG_SYSTEM_PROMPT = """
You answer questions about the indexed repository.
Use only the retrieved context for repository-specific claims.
The answer must always contain exactly these sections: Ответ, Источники, Цитаты.
In Источники list source, section and chunk_id for every chunk used.
In Цитаты include short exact fragments copied from the retrieved chunks.
If the context does not support the answer, say "не знаю" and ask the user to clarify.
""".strip()
