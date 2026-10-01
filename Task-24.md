# Task 24

## Что сделано

Доработан RAG-ответ: агент теперь обязан возвращать структурированный результат с ответом, источниками и цитатами, а при недостаточной релевантности не генерирует неподтверждённый ответ.

Обязательный формат RAG-ответа:

```text
Ответ:
...

Источники:
- source=<source>; section=<section>; chunk_id=<chunk_id>

Цитаты:
- [<chunk_id>] "<фрагмент из найденного чанка>"
```

## Источники и цитаты

Обновлён:

```text
rag/qa/agent.py
```

Prompt для RAG теперь требует:

- блок `Ответ` с прямым ответом на языке пользователя;
- блок `Источники`;
- блок `Цитаты`;
- для каждого источника указывать `source`, `section` и `chunk_id`;
- для каждой цитаты использовать короткий точный фрагмент из найденного чанка.

В retrieved context теперь передаются:

- `source`;
- `file_path`;
- `section`;
- `chunk_id`;
- `score`;
- `similarity`;
- полный текст чанка.

Системный prompt для RAG также обновлён: модель должна использовать только retrieved context, всегда возвращать секции `Ответ`, `Источники`, `Цитаты`, а если контекст не подтверждает ответ - сказать `не знаю` и попросить уточнение.

## Low relevance gate

Добавлено правило на уровне агента:

```text
если после retrieval/filter нет чанков выше relevance_threshold,
агент возвращает "не знаю" без вызова LLM для генерации ответа
```

Это сделано детерминированно, чтобы поведение не зависело от того, выполнит ли LLM инструкцию в prompt.

Fallback-ответ содержит:

- `Ответ` с фразой `не знаю`;
- объяснение, что нет фрагментов выше порога релевантности;
- лучший найденный `score`;
- просьбу уточнить вопрос;
- пустой список релевантных источников;
- пустой список релевантных цитат.

## Retrieval models

Обновлён:

```text
rag/retrieval/retriever.py
```

В `RetrievedChunk` добавлено поле:

```text
source
```

Теперь source сохраняется вместе с `chunk_id`, `file_path`, `section`, score и остальными retrieval metadata.

## CLI

Обновлён:

```text
scripts/ask.py
```

Добавлен режим:

```bash
.venv/bin/python scripts/ask.py --control-improved
```

Он запускает контрольные вопросы только через `IMPROVED_RAG`, без дополнительных прогонов `WITHOUT_RAG` и `BASELINE_RAG`.

В консоль для каждого вопроса теперь выводятся:

- метрики improved RAG;
- полный структурированный ответ модели;
- retrieved evidence с `source`, `section`, `chunk_id`, `score`, `similarity`;
- короткий preview чанка как цитатный фрагмент для быстрой проверки.

Для русского набора вопросов:

```bash
.venv/bin/python scripts/ask.py --control-improved --control-questions CONTROL_QUESTIONS.ru.md
```

В диагностическом выводе retrieved chunks теперь печатаются:

```text
source=<source> / file=<file_path> / section=<section> / chunk_id=<chunk_id>
```

Это дополняет сам ответ LLM и помогает быстро проверить, какие источники реально были переданы в prompt.

## Evaluation и отчёты

Обновлён:

```text
rag/evaluation/evaluation.py
```

Добавлен improved-only evaluation:

- `ImprovedControlQuestionResult`;
- `ImprovedEvaluationSummary`;
- `ImprovedRagEvaluationReport`;
- `RagQualityEvaluator.evaluate_improved_only(...)`;
- `write_improved_evaluation_report(...)`.

Новые файлы отчёта:

```text
artifacts/rag-quality-improved.json
artifacts/rag-quality-improved.md
```

В `RetrievedEvidence` добавлено поле:

```text
source
```

Retrieval diagnostics в JSON/Markdown отчётах теперь включают:

- `source`;
- `file_path`;
- `section`;
- `chunk_id`;
- `score`;
- `similarity_score`;
- `matched_terms`;
- `source_queries`.

## Документация

Обновлён:

```text
README.md
```

Добавлено описание обязательного формата RAG-ответа и правила low relevance:

- RAG-ответ должен содержать `Ответ`, `Источники`, `Цитаты`;
- источники должны включать `source`, `section`, `chunk_id`;
- цитаты должны быть фрагментами из retrieved chunks;
- если нет чанков выше `--relevance-threshold`, агент возвращает `не знаю` и просит уточнить вопрос.

## Проверки

Выполнены:

```bash
find rag scripts -name '*.py' -print0 | xargs -0 .venv/bin/python -m py_compile
.venv/bin/python scripts/ask.py --help
.venv/bin/python scripts/ask.py --rag --embedding-provider hashing --query-rewrite off --relevance-threshold 1.0 "zzzz qqqq unrelated nonsense"
```

Low relevance smoke test вернул структурированный ответ:

```text
Ответ:
не знаю...

Источники:
- нет релевантных источников

Цитаты:
- нет релевантных цитат
```
