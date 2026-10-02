from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from rag.qa.chat_utils import compact
from rag.qa.message_router import MessageType, RoutedMessage, route_message


class DialogueAction(str, Enum):
    LOCAL_REPLY = "LOCAL_REPLY"
    UPDATE_TASK_STATE = "UPDATE_TASK_STATE"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    RAG_ANSWER = "RAG_ANSWER"
    SYNTHESIZE_WITH_RAG = "SYNTHESIZE_WITH_RAG"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True)
class DialogueDecision:
    action: DialogueAction
    routed: RoutedMessage
    retrieval_query: str | None = None
    prompt_instruction: str | None = None


class DialogueAgent:
    def decide(self, message: str, task_state: Any, history: list[Any]) -> DialogueDecision:
        routed = route_message(message)
        if routed.type == MessageType.SMALL_TALK:
            return DialogueDecision(DialogueAction.LOCAL_REPLY, routed)
        if routed.type in {MessageType.TASK_SETUP, MessageType.CONSTRAINT_UPDATE, MessageType.CLARIFICATION}:
            return DialogueDecision(DialogueAction.UPDATE_TASK_STATE, routed)
        if routed.type == MessageType.OUT_OF_SCOPE:
            return DialogueDecision(DialogueAction.OUT_OF_SCOPE, routed)
        if is_synthesis_request(message):
            return DialogueDecision(
                action=DialogueAction.SYNTHESIZE_WITH_RAG,
                routed=routed,
                retrieval_query=build_synthesis_retrieval_query(message, task_state, history),
                prompt_instruction=(
                    "This is a synthesis request. Use task memory and recent conversation to preserve the user's goal, "
                    "then use retrieved context for repository facts. Produce the requested plan/summary, not a generic answer."
                ),
            )
        return DialogueDecision(DialogueAction.RAG_ANSWER, routed)


def is_synthesis_request(message: str) -> bool:
    normalized = compact(message).lower()
    return any(marker in normalized for marker in SYNTHESIS_MARKERS)


def build_synthesis_retrieval_query(message: str, task_state: Any, history: list[Any]) -> str:
    lines = [
        "Синтез ответа по истории диалога и источникам.",
        f"Запрос пользователя: {compact(message)}",
    ]
    goal = getattr(task_state, "goal", "")
    clarifications = list(getattr(task_state, "clarifications", []))
    constraints = list(getattr(task_state, "constraints", []))
    terms = list(getattr(task_state, "terms", []))
    if goal:
        lines.append(f"Цель диалога: {goal}")
    if clarifications:
        lines.append("Уточнения: " + "; ".join(clarifications[-6:]))
    if constraints:
        lines.append("Ограничения: " + "; ".join(constraints[-6:]))
    if terms:
        lines.append("Термины: " + "; ".join(terms[-8:]))
    user_turns = [
        compact(getattr(item, "content", ""))
        for item in history
        if getattr(item, "role", "") == "user"
    ]
    if user_turns:
        lines.append("Последние вопросы пользователя: " + "; ".join(user_turns[-6:]))
    lines.append("Нужно найти факты для пошагового плана, итоговой структуры или резюме по цели диалога.")
    return "\n".join(lines)


SYNTHESIS_MARKERS = (
    "вернись к цели",
    "цель всё ещё",
    "цель все еще",
    "собери итог",
    "подведи итог",
    "итог",
    "резюмируй",
    "пошаговый план",
    "план реализации",
    "структуру доклада",
    "короткую структуру",
)
