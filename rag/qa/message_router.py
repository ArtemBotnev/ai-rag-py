from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from rag.qa.chat_utils import compact


class MessageType(str, Enum):
    SMALL_TALK = "SMALL_TALK"
    TASK_SETUP = "TASK_SETUP"
    CONSTRAINT_UPDATE = "CONSTRAINT_UPDATE"
    CLARIFICATION = "CLARIFICATION"
    RAG_QUESTION = "RAG_QUESTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True)
class RoutedMessage:
    type: MessageType
    reason: str


def route_message(message: str) -> RoutedMessage:
    normalized = compact(message).lower().strip("!., ")
    if has_marker(normalized, OUT_OF_SCOPE_MARKERS):
        return RoutedMessage(MessageType.OUT_OF_SCOPE, "явно вне индексированного проекта")
    if is_goal_setup(normalized):
        return RoutedMessage(MessageType.TASK_SETUP, "явная постановка цели диалога")
    if has_explicit_question_mark(normalized):
        return RoutedMessage(MessageType.RAG_QUESTION, "явный вопрос к индексированным источникам")
    if has_marker(normalized, CONSTRAINT_MARKERS):
        return RoutedMessage(MessageType.CONSTRAINT_UPDATE, "обновление ограничений диалога")
    if is_answer_command(normalized):
        return RoutedMessage(MessageType.RAG_QUESTION, "команда ответа с учётом индексированных источников")
    if has_marker(normalized, CLARIFICATION_MARKERS):
        return RoutedMessage(MessageType.CLARIFICATION, "уточнение цели, терминов или контекста")
    if is_question(normalized):
        return RoutedMessage(MessageType.RAG_QUESTION, "вопрос к индексированным источникам")
    if has_marker(normalized, TASK_INTENT_MARKERS):
        return RoutedMessage(MessageType.CLARIFICATION, "неявное уточнение намерения без смены цели")
    if is_small_talk(normalized):
        return RoutedMessage(MessageType.SMALL_TALK, "служебная реплика")
    return RoutedMessage(MessageType.RAG_QUESTION, "сообщение интерпретировано как вопрос к RAG")


def is_small_talk(normalized: str) -> bool:
    if normalized in CONVERSATIONAL_MESSAGES:
        return True
    tokens = normalized.split()
    return len(tokens) <= 4 and bool(tokens) and tokens[0] in GREETING_WORDS


def is_question(normalized: str) -> bool:
    if has_explicit_question_mark(normalized):
        return True
    return any(normalized.startswith(prefix) for prefix in QUESTION_PREFIXES)


def has_explicit_question_mark(normalized: str) -> bool:
    return "?" in normalized


def is_answer_command(normalized: str) -> bool:
    return any(normalized.startswith(prefix) for prefix in ANSWER_COMMAND_PREFIXES)


def is_goal_setup(normalized: str) -> bool:
    return any(
        normalized == marker or normalized.startswith(f"{marker}:") or normalized.startswith(f"{marker} ")
        for marker in TASK_GOAL_PREFIXES
    )


def has_marker(normalized: str, markers: tuple[str, ...]) -> bool:
    return any(matches_marker(normalized, marker) for marker in markers)


def matches_marker(normalized: str, marker: str) -> bool:
    if marker in STEM_MARKERS:
        return marker in normalized
    phrase = marker.strip()
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized) is not None


GREETING_WORDS = {"привет", "здравствуй", "здравствуйте", "добрый", "доброе", "hello", "hi"}
THANKS_MESSAGES = {"спасибо", "благодарю", "thanks", "thank you"}
ACK_MESSAGES = {"ок", "окей", "хорошо", "понял", "поняла", "принял", "ясно"}
CONVERSATIONAL_MESSAGES = GREETING_WORDS | THANKS_MESSAGES | ACK_MESSAGES | {
    "приветствую",
    "добрый день",
    "доброе утро",
    "добрый вечер",
}
TASK_GOAL_PREFIXES = (
    "цель",
    "цель диалога",
    "моя цель",
    "наша цель",
    "задача",
    "задача диалога",
    "цель задачи",
)
TASK_INTENT_MARKERS = (
    "хочу",
    "нужно",
    "надо",
    "помоги",
    "объясни",
    "разобраться",
    "узнать",
    "выяснить",
    "должны",
    "собираюсь",
)
CONSTRAINT_MARKERS = (
    "отвечай",
    "ответь",
    "кратко",
    "лаконично",
    "подробно",
    "только",
    "без ",
    "не надо",
    "не нужно",
    "не уходи",
    "не использ",
    "огранич",
    "важно",
    "считай",
    "используй",
    "учитывай",
)
CLARIFICATION_MARKERS = (
    "уточ",
    "добав",
    "ещё",
    "еще",
    "подробнее",
    "имею в виду",
    "в контексте",
    "под этим",
    "термин",
    "назовем",
    "назовём",
    "будем называть",
)
QUESTION_PREFIXES = (
    "что ",
    "как ",
    "где ",
    "какие ",
    "какой ",
    "какая ",
    "каким ",
    "почему ",
    "зачем ",
    "когда ",
    "чем ",
    "покажи ",
    "расскажи ",
    "объясни ",
    "сравни ",
    "перечисли ",
)
ANSWER_COMMAND_PREFIXES = (
    "собери ",
    "дай ",
    "подготовь ",
    "сформируй ",
    "сделай ",
    "резюмируй ",
    "подведи ",
    "начни ",
)
OUT_OF_SCOPE_MARKERS = (
    "в другом проекте",
    "последняя версия",
    "новости",
    "android 16",
    "курс валют",
)
STEM_MARKERS = {
    "огранич",
    "не использ",
    "уточ",
    "добав",
}
