from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatUsage:
    total_duration_ms: int
    load_duration_ms: int
    prompt_eval_count: int
    prompt_eval_duration_ms: int
    eval_count: int
    eval_duration_ms: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_eval_count + self.eval_count


@dataclass(frozen=True)
class ChatCompletion:
    model: str
    content: str
    usage: ChatUsage
    wall_time_ms: int


class OllamaChatClient:
    def __init__(self, base_url: str, model: str | None = None, timeout_seconds: int = 900) -> None:
        self.base_url = base_url.rstrip("/")
        self.configured_model = model or os.getenv("OLLAMA_LLM_MODEL")
        self.timeout_seconds = timeout_seconds
        self._model: str | None = None

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self.configured_model or self._discover_model()
        return self._model

    def chat(self, system_prompt: str | None, user_prompt: str) -> ChatCompletion:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = json.dumps({"model": self.model, "messages": messages, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise self._ollama_error(f"Ollama chat request failed for model `{self.model}`") from exc

        message = response_payload.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Ollama response does not contain message content: {response_payload}")
        return ChatCompletion(
            model=response_payload.get("model", self.model),
            content=content.strip(),
            usage=ChatUsage(
                total_duration_ms=nanoseconds_to_ms(response_payload.get("total_duration", 0)),
                load_duration_ms=nanoseconds_to_ms(response_payload.get("load_duration", 0)),
                prompt_eval_count=int(response_payload.get("prompt_eval_count", 0) or 0),
                prompt_eval_duration_ms=nanoseconds_to_ms(response_payload.get("prompt_eval_duration", 0)),
                eval_count=int(response_payload.get("eval_count", 0) or 0),
                eval_duration_ms=nanoseconds_to_ms(response_payload.get("eval_duration", 0)),
            ),
            wall_time_ms=int((time.perf_counter() - started) * 1000),
        )

    def _discover_model(self) -> str:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise self._ollama_error("Ollama model discovery failed") from exc

        for model in response_payload.get("models", []):
            name = model.get("name", "")
            if name and "embed" not in name.lower():
                return name
        raise RuntimeError(
            "No chat-capable Ollama model found. Pull a text model, pass --llm-model, "
            "or run `OLLAMA_LLM_MODEL=llama3.1:8b scripts/start_ollama.sh`."
        )

    def _ollama_error(self, prefix: str) -> RuntimeError:
        return RuntimeError(f"{prefix}. Start Ollama and make sure the model is available at {self.base_url}.")


def nanoseconds_to_ms(value: object) -> int:
    try:
        return int(value) // 1_000_000
    except (TypeError, ValueError):
        return 0
