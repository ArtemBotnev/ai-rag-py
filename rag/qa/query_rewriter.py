from __future__ import annotations

from dataclasses import dataclass

from rag.inference.llm import ChatCompletion, OllamaChatClient


@dataclass(frozen=True)
class QueryRewriteResult:
    query: str
    completion: ChatCompletion


class QueryRewriter:
    def __init__(self, chat_client: OllamaChatClient) -> None:
        self.chat_client = chat_client

    def rewrite(self, question: str) -> QueryRewriteResult | None:
        sanitized_question = question.strip()
        if not sanitized_question:
            return None

        completion = self.chat_client.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_prompt(sanitized_question),
        )
        rewritten_query = first_rewritten_line(completion.content)
        if not rewritten_query or rewritten_query.lower() == sanitized_question.lower():
            return None
        return QueryRewriteResult(query=rewritten_query, completion=completion)


def build_prompt(question: str) -> str:
    return "\n".join(
        [
            "Rewrite the user question into a concise retrieval query for repository search.",
            "Keep class names, file names, module names, APIs and domain terms.",
            "Prefer precise technical keywords and short noun phrases over full sentences.",
            "If the repository content is likely English, the rewrite may use English technical terms.",
            "Return only the rewritten query, with no explanation.",
            "",
            "Question:",
            question,
        ]
    )


def first_rewritten_line(content: str) -> str:
    for line in content.splitlines():
        candidate = line.strip()
        for prefix in ("Query:", "query:"):
            candidate = candidate.removeprefix(prefix).strip()
        candidate = candidate.strip("\"'`")
        if candidate:
            return candidate
    return ""


SYSTEM_PROMPT = """
You rewrite software questions into compact repository-search queries.
Return one line only. Do not explain the rewrite.
""".strip()
