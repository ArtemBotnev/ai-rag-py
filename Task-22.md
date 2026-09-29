# Task 22

## Что сделано

Реализован первый RAG-запрос поверх локального индекса:

```text
вопрос -> поиск релевантных чанков -> объединение контекста с вопросом -> запрос к LLM
```

Добавлен агент с двумя режимами:

- `WITHOUT_RAG` - вопрос отправляется напрямую в LLM;
- `WITH_RAG` - вопрос сначала используется для retrieval по SQLite-индексу, затем найденные чанки добавляются в prompt.

## Контрольные вопросы

Добавлены два файла с 10 контрольными вопросами по базе `source/nowinandroid-main`:

- `CONTROL_QUESTIONS.md` - английский вариант по умолчанию;
- `CONTROL_QUESTIONS.ru.md` - русский вариант.

Для каждого вопроса зафиксированы:

- сам вопрос;
- ожидание от ответа;
- ожидаемые источники;
- check-фразы для автоматической оценки покрытия ответа.

## CLI для вопросов

Добавлен скрипт `scripts/ask.py`.

Поддерживаемые режимы:

```bash
.venv/bin/python scripts/ask.py --rag "Как в проекте реализована синхронизация данных?"
.venv/bin/python scripts/ask.py --no-rag "Как устроена модульная структура проекта?"
.venv/bin/python scripts/ask.py --compare "Какие архитектурные слои использует Now in Android?"
.venv/bin/python scripts/ask.py --control
.venv/bin/python scripts/ask.py --control --control-questions CONTROL_QUESTIONS.ru.md
```

В режиме `--control` скрипт проходит по всем контрольным вопросам, вызывает оба режима ответа и сохраняет отчёты.

## Метрики и отчёты

Для ответов фиксируются:

- время retrieval;
- время генерации;
- общее время;
- prompt tokens;
- completion tokens;
- total tokens;
- найденные источники;
- покрытие ожидаемых источников;
- покрытие check-фраз;
- итоговый verdict: `WITH_RAG`, `WITHOUT_RAG` или `TIE`.

Результаты сохраняются в:

```text
artifacts/rag-quality-comparison.json
artifacts/rag-quality-comparison.md
```

## Новые модули

Код был разнесён по архитектурным слоям:

```text
rag/common/
  models.py
  tokenizer.py

rag/indexing/
  chunkers.py
  document_loader.py
  pipeline.py
  reporter.py
  storage.py

rag/inference/
  embeddings.py
  llm.py

rag/retrieval/
  retriever.py

rag/qa/
  agent.py

rag/evaluation/
  control_questions.py
  evaluation.py
```

Основные новые части:

- `rag/retrieval/retriever.py` - загружает чанки из SQLite, декодирует embeddings и ищет top-k чанков по cosine similarity.
- `rag/inference/llm.py` - клиент Ollama `/api/chat`, возвращает ответ, токены и тайминги.
- `rag/qa/agent.py` - orchestration режимов `WITH_RAG` и `WITHOUT_RAG`.
- `rag/evaluation/control_questions.py` - парсер `CONTROL_QUESTIONS.md` и `CONTROL_QUESTIONS.ru.md`.
- `rag/evaluation/evaluation.py` - оценка качества и генерация JSON/Markdown отчётов.

## Ollama

Скрипты запуска Ollama разделены по назначению:

```bash
scripts/start_ollama.sh
scripts/start_ollama_llm.sh
```

`scripts/start_ollama.sh` готовит embedding-модель:

```bash
OLLAMA_MODEL=nomic-embed-text:latest scripts/start_ollama.sh
```

`scripts/start_ollama_llm.sh` готовит LLM-модель:

```bash
OLLAMA_LLM_MODEL=llama3.1:8b scripts/start_ollama_llm.sh
```

## Документация

Обновлён `README.md`:

- добавлены команды для RAG/no-RAG/compare/control;
- описаны новые отчёты качества;
- обновлена структура проекта;
- описаны отдельные скрипты для embedding- и LLM-моделей.

## Проверки

Были выполнены проверки без полного запуска LLM:

```bash
.venv/bin/python -m py_compile $(find rag scripts -name '*.py' | sort)
.venv/bin/python scripts/ask.py --help
bash -n scripts/start_ollama.sh
bash -n scripts/start_ollama_llm.sh
```

Также проверены:

- парсинг 10 контрольных вопросов;
- загрузка чанков из `artifacts/rag-index.sqlite`;
- retrieval по SQLite через hashing fallback;
- генерация тестового JSON/Markdown отчёта.
