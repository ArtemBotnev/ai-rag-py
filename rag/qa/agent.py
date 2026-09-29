from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rag.inference.llm import ChatCompletion, OllamaChatClient
from rag.retrieval.retriever import RetrievedChunk, SqliteRagRetriever


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

    @property
    def total_duration_ms(self) -> int:
        return self.retrieval_duration_ms + self.completion.wall_time_ms


class RagAgent:
    def __init__(self, chat_client: OllamaChatClient, retriever: SqliteRagRetriever | None = None) -> None:
        self.chat_client = chat_client
        self.retriever = retriever

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
        if self.retriever is None:
            raise RuntimeError("RAG mode requires a retriever.")
        retrieval = self.retriever.search(question, limit=top_k)
        completion = self.chat_client.chat(WITH_RAG_SYSTEM_PROMPT, build_rag_prompt(question, retrieval.chunks))
        return AgentAnswer(
            mode=AnswerMode.WITH_RAG,
            question=question,
            answer=completion.content,
            model=completion.model,
            completion=completion,
            retrieved_chunks=retrieval.chunks,
            retrieval_duration_ms=retrieval.duration_ms,
        )


def build_rag_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    lines = [
        "Question:",
        question.strip(),
        "",
        "Retrieved context:",
        "",
    ]
    if not retrieved_chunks:
        lines.append("No context was retrieved.")
    for index, chunk in enumerate(retrieved_chunks, start=1):
        lines.extend(
            [
                f"[{index}] file={chunk.file_path}",
                f"section={chunk.section}",
                f"score={chunk.score:.4f}",
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
