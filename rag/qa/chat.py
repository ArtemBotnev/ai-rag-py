from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.inference.llm import ChatCompletion, ChatUsage
from rag.qa.agent import AgentAnswer, AnswerMode, RagAgent
from rag.qa.dialogue_agent import DialogueAction, DialogueAgent
from rag.qa.message_router import MessageType, RoutedMessage
from rag.retrieval.retriever import RetrievalConfig


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatMessage":
        return cls(role=str(payload.get("role", "")), content=str(payload.get("content", "")))


@dataclass
class TaskState:
    goal: str = ""
    clarifications: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "clarifications": self.clarifications,
            "constraints": self.constraints,
            "terms": self.terms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskState":
        return cls(
            goal=str(payload.get("goal", "")),
            clarifications=coerce_string_list(payload.get("clarifications")),
            constraints=coerce_string_list(payload.get("constraints")),
            terms=coerce_string_list(payload.get("terms")),
        )


@dataclass(frozen=True)
class ChatTurnResult:
    user_message: str
    answer: AgentAnswer
    task_state: TaskState
    retrieval_question: str


class ChatSession:
    def __init__(
        self,
        agent: RagAgent,
        retrieval_config: RetrievalConfig,
        history: list[ChatMessage] | None = None,
        task_state: TaskState | None = None,
        history_limit: int = 8,
    ) -> None:
        self.agent = agent
        self.retrieval_config = retrieval_config
        self.history = history or []
        self.task_state = task_state or TaskState()
        self.history_limit = max(2, history_limit)
        self.dialogue_agent = DialogueAgent()

    def ask(self, message: str) -> ChatTurnResult:
        user_message = message.strip()
        if not user_message:
            raise RuntimeError("Message must not be blank.")

        decision = self.dialogue_agent.decide(user_message, self.task_state, self.recent_history())
        if decision.action != DialogueAction.LOCAL_REPLY:
            update_task_state(self.task_state, user_message, decision.routed.type)

        if decision.action in {
            DialogueAction.LOCAL_REPLY,
            DialogueAction.UPDATE_TASK_STATE,
            DialogueAction.ASK_CLARIFICATION,
            DialogueAction.OUT_OF_SCOPE,
        }:
            answer = build_local_answer(user_message, decision.routed, self.task_state)
            self.history.append(ChatMessage(role="user", content=user_message))
            self.history.append(ChatMessage(role="assistant", content=answer.answer))
            return ChatTurnResult(
                user_message=user_message,
                answer=answer,
                task_state=self.task_state,
                retrieval_question=f"{decision.action.value}: RAG-поиск не выполнялся",
            )

        retrieval_question = decision.retrieval_query or build_retrieval_question(user_message, self.task_state)
        prompt_context = build_prompt_context(
            self.task_state,
            self.recent_history(),
            prompt_instruction=decision.prompt_instruction,
        )
        self.history.append(ChatMessage(role="user", content=user_message))
        answer = self.agent.answer_with_rag_config(
            question=user_message,
            retrieval_config=self.retrieval_config,
            retrieval_question=retrieval_question,
            prompt_context=prompt_context,
        )
        self.history.append(ChatMessage(role="assistant", content=answer.answer))
        return ChatTurnResult(
            user_message=user_message,
            answer=answer,
            task_state=self.task_state,
            retrieval_question=retrieval_question,
        )

    def recent_history(self) -> list[ChatMessage]:
        return self.history[-self.history_limit :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "task_state": self.task_state.to_dict(),
            "history": [message.to_dict() for message in self.history],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        agent: RagAgent,
        retrieval_config: RetrievalConfig,
        history_limit: int = 8,
    ) -> "ChatSession":
        history = [
            ChatMessage.from_dict(item)
            for item in payload.get("history", [])
            if isinstance(item, dict)
        ]
        state_payload = payload.get("task_state", {})
        task_state = TaskState.from_dict(state_payload if isinstance(state_payload, dict) else {})
        return cls(
            agent=agent,
            retrieval_config=retrieval_config,
            history=history,
            task_state=task_state,
            history_limit=history_limit,
        )


def load_chat_session(
    path: Path,
    agent: RagAgent,
    retrieval_config: RetrievalConfig,
    history_limit: int = 8,
) -> ChatSession:
    if not path.exists():
        return ChatSession(agent=agent, retrieval_config=retrieval_config, history_limit=history_limit)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid chat history JSON: {path}")
    return ChatSession.from_dict(payload, agent=agent, retrieval_config=retrieval_config, history_limit=history_limit)


def save_chat_session(session: ChatSession, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_task_state(state: TaskState, message: str, message_type: MessageType = MessageType.RAG_QUESTION) -> None:
    normalized = normalize_task_message(message, message_type)
    if message_type == MessageType.TASK_SETUP:
        state.goal = trim_sentence(normalized, 220)

    if message_type == MessageType.CONSTRAINT_UPDATE:
        append_unique(state.constraints, trim_sentence(normalized, 220), limit=12)
    elif message_type == MessageType.CLARIFICATION:
        append_unique(state.clarifications, trim_sentence(normalized, 220), limit=12)

    if message_type in {MessageType.TASK_SETUP, MessageType.CLARIFICATION}:
        append_unique(state.clarifications, trim_sentence(normalized, 220), limit=12)

    for term in extract_terms(normalized):
        append_unique(state.terms, term, limit=16)


def build_retrieval_question(message: str, state: TaskState) -> str:
    lines = [message.strip()]
    if state.goal:
        lines.append(f"Цель диалога: {state.goal}")
    if state.constraints:
        lines.append("Ограничения и термины: " + "; ".join(state.constraints[-5:]))
    if state.terms:
        lines.append("Зафиксированные термины: " + "; ".join(state.terms[-8:]))
    return "\n".join(lines)


def build_prompt_context(
    state: TaskState,
    history: list[ChatMessage],
    prompt_instruction: str | None = None,
) -> str:
    lines = [
        "Task memory:",
        f"- goal: {state.goal or 'not fixed yet'}",
        f"- clarifications: {format_items(state.clarifications)}",
        f"- constraints: {format_items(state.constraints)}",
        f"- terms: {format_items(state.terms)}",
        "",
        "Recent conversation:",
    ]
    if not history:
        lines.append("- none")
    for message in history:
        lines.append(f"- {message.role}: {trim_sentence(message.content, 500)}")
    lines.extend(
        [
            "",
            "Use the task memory to preserve the dialogue goal and fixed constraints.",
            "Do not treat task memory as a source for repository facts; repository facts must come from retrieved context.",
        ]
    )
    if prompt_instruction:
        lines.extend(["", "Dialogue agent instruction:", prompt_instruction])
    return "\n".join(lines)


def format_task_state(state: TaskState) -> str:
    return "\n".join(
        [
            "Память задачи:",
            f"- цель: {state.goal or 'не зафиксирована'}",
            f"- уточнения: {format_items(state.clarifications)}",
            f"- ограничения/термины: {format_items(state.constraints + state.terms)}",
        ]
    )


def extract_scenario_messages(path: Path) -> list[str]:
    messages: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            messages.append(stripped[2:].strip())
        elif stripped[0].isdigit() and ". " in stripped[:5]:
            messages.append(stripped.split(". ", 1)[1].strip())
    return [message for message in messages if message]


def extract_terms(message: str) -> list[str]:
    terms: list[str] = []
    quoted_terms = re.findall(r"[`\"«](.+?)[`\"»]", message)
    terms.extend(trim_sentence(term, 80) for term in quoted_terms)
    for pattern in TERM_PATTERNS:
        terms.extend(trim_sentence(match, 80) for match in re.findall(pattern, message, flags=re.IGNORECASE))
    return [term for term in terms if term]


def normalize_task_message(message: str, message_type: MessageType) -> str:
    normalized = compact(message)
    if message_type != MessageType.TASK_SETUP:
        return normalized
    without_greeting = re.sub(
        r"^(привет|здравствуйте|здравствуй|добрый день|доброе утро|добрый вечер)[!,. ]+",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip() or normalized
    return re.sub(
        r"^цель диалога:\s*",
        "",
        without_greeting,
        flags=re.IGNORECASE,
    ).strip() or without_greeting


def build_local_answer(message: str, routed: RoutedMessage, state: TaskState) -> AgentAnswer:
    normalized = compact(message).lower().strip("!.?, ")
    if routed.type == MessageType.SMALL_TALK and normalized in THANKS_MESSAGES:
        response = "Пожалуйста. Задайте следующий вопрос по индексированным исходникам."
        source = "служебная реплика; RAG-поиск не выполнялся"
    elif routed.type == MessageType.SMALL_TALK and normalized in ACK_MESSAGES:
        response = "Принял. Продолжайте, когда будете готовы."
        source = "служебная реплика; RAG-поиск не выполнялся"
    elif routed.type == MessageType.SMALL_TALK:
        response = "Привет! Задайте вопрос по индексированным исходникам, и я отвечу с источниками."
        source = "служебная реплика; RAG-поиск не выполнялся"
    elif routed.type == MessageType.TASK_SETUP:
        response = (
            f"Зафиксировал цель: {state.goal}. "
            "Что разобрать сначала: архитектурные слои, модули, data layer, синхронизацию или UI?"
        )
        source = "постановка задачи; RAG-поиск не выполнялся"
    elif routed.type == MessageType.CONSTRAINT_UPDATE:
        response = f"Принял ограничение: {trim_sentence(message, 180)}."
        source = "настройка диалога; RAG-поиск не выполнялся"
    elif routed.type == MessageType.CLARIFICATION:
        response = "Принял уточнение и сохранил его в памяти задачи."
        source = "уточнение диалога; RAG-поиск не выполнялся"
    elif routed.type == MessageType.OUT_OF_SCOPE:
        response = "Это похоже на вопрос вне текущего индекса. Задайте вопрос по проиндексированным исходникам Now in Android."
        source = "вне области индекса; RAG-поиск не выполнялся"
    else:
        response = "Принял. Задайте вопрос по индексированным исходникам."
        source = "локальная маршрутизация; RAG-поиск не выполнялся"
    content = "\n".join(
        [
            "Ответ:",
            response,
            "",
            "Источники:",
            f"- {source}",
            "",
            "Цитаты:",
            "- нет цитат",
        ]
    )
    completion = ChatCompletion(
        model="chat-local-conversation",
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
    return AgentAnswer(
        mode=AnswerMode.WITH_RAG,
        question=message,
        answer=content,
        model=completion.model,
        completion=completion,
        retrieved_chunks=[],
    )


def coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def append_unique(items: list[str], value: str, limit: int) -> None:
    normalized = value.strip()
    if not normalized:
        return
    existing = {item.lower() for item in items}
    if normalized.lower() not in existing:
        items.append(normalized)
    del items[:-limit]


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def trim_sentence(value: str, limit: int) -> str:
    prepared = compact(value)
    if len(prepared) <= limit:
        return prepared
    return prepared[: limit - 1].rstrip() + "..."


def format_items(items: list[str]) -> str:
    return "; ".join(items) if items else "нет"


TERM_PATTERNS = (
    r"(?:термин|термином)\s+([\w .:/-]{2,80})",
    r"(?:назов[её]м|будем называть)\s+([\w .:/-]{2,80})",
)
THANKS_MESSAGES = {"спасибо", "благодарю", "thanks", "thank you"}
ACK_MESSAGES = {"ок", "окей", "хорошо", "понял", "поняла", "принял", "ясно"}
