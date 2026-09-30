# Task 23

## Что сделано

Доработан RAG-пайплайн: после первичного vector search добавлен improved retrieval с query rewrite, rerank/filter и отдельными настройками `top-K` до и после фильтрации.

Теперь сравнение качества выполняется в трёх режимах:

- `WITHOUT_RAG` - вопрос отправляется напрямую в LLM;
- `BASELINE_RAG` - старый режим `question -> vector top-K -> LLM`;
- `IMPROVED_RAG` - новый режим `question -> query rewrite -> top-K candidates -> rerank/filter -> filtered context -> LLM`.

## Improved retrieval

Добавлены модели retrieval:

```text
RetrievalMode
RetrievalConfig
RetrievalResult
RetrievedChunk
```

Файл:

```text
rag/retrieval/retriever.py
```

Новый режим `IMPROVED` делает:

1. берёт исходный вопрос;
2. при включённом rewrite получает короткий retrieval query;
3. ищет кандидатов по исходному вопросу и rewritten query;
4. объединяет дубликаты по `chunk_id`;
5. считает эвристический relevance score;
6. отбрасывает кандидатов ниже `relevance_threshold`;
7. оставляет финальные `top_k_after` чанков для prompt.

Rerank/filter использует:

- исходный cosine similarity;
- rank кандидата в первичной выдаче;
- пересечение query terms с `file_path`;
- пересечение с `title`;
- пересечение с `section`;
- пересечение с текстом чанка;
- штраф для кандидатов без совпавших терминов.

В `RetrievedChunk` теперь отдельно хранятся:

- `score` - итоговый relevance score;
- `similarity_score` - исходный cosine similarity;
- `matched_terms` - совпавшие query terms;
- `source_queries` - откуда пришёл кандидат: `original`, `rewritten` или оба.

## Query rewrite

Добавлен отдельный компонент:

```text
rag/qa/query_rewriter.py
```

Он использует тот же chat client, что и агент, и просит LLM переписать вопрос в короткий поисковый запрос для retrieval.

Rewrite включается только в `IMPROVED` режиме и не заменяет исходный вопрос полностью: retriever ищет и по original question, и по rewritten query.

## Агент

Обновлён:

```text
rag/qa/agent.py
```

Добавлен метод:

```text
answer_with_rag_config(question, retrieval_config)
```

Агент теперь:

- поддерживает `BASELINE` и `IMPROVED` retrieval;
- передаёт rewritten query в retriever;
- добавляет rewritten query в prompt;
- возвращает полный retrieval trace;
- учитывает время и токены query rewrite в общих метриках improved ответа.

Старый метод `answer_with_rag(question, top_k)` сохранён как baseline-совместимый wrapper.

## CLI

Обновлён:

```text
scripts/ask.py
```

Добавлены параметры:

```bash
--retrieval-mode BASELINE|IMPROVED
--top-k <n>
--top-k-before <n>
--top-k-after <n>
--relevance-threshold <0..1>
--query-rewrite on|off
```

`--top-k` оставлен как совместимый alias для `--top-k-after`.

По умолчанию `--rag` использует:

```text
retrieval-mode = IMPROVED
top-k-before = 8
top-k-after = 4
relevance-threshold = 0.45
query-rewrite = on
```

Пример improved RAG:

```bash
.venv/bin/python scripts/ask.py --rag "Как в проекте реализована синхронизация данных?"
```

Пример baseline RAG:

```bash
.venv/bin/python scripts/ask.py --rag \
  --retrieval-mode BASELINE \
  --top-k 4 \
  "Как в проекте реализована синхронизация данных?"
```

`--compare` теперь запускает:

```text
WITHOUT_RAG
BASELINE_RAG
IMPROVED_RAG
```

`--control` также сравнивает все три режима.

## Evaluation и отчёты

Обновлён:

```text
rag/evaluation/evaluation.py
```

Отчёт теперь хранит отдельно:

- `without_rag`;
- `baseline_rag`;
- `improved_rag`;
- verdict: `WITHOUT_RAG`, `BASELINE_RAG`, `IMPROVED_RAG` или `TIE`.

В summary добавлены:

- среднее покрытие `BASELINE_RAG`;
- среднее покрытие `IMPROVED_RAG`;
- среднее покрытие ожидаемых источников для baseline/improved;
- среднее время и токены для baseline/improved;
- количество побед baseline/improved/without/tie.

В retrieval diagnostics пишутся:

- original query;
- rewritten query;
- `top_k_before`;
- `top_k_after`;
- `relevance_threshold`;
- candidates before filter;
- discarded by filter;
- итоговые retrieved evidence;
- `score`;
- `similarity_score`;
- `matched_terms`;
- `source_queries`.

Результаты сохраняются в прежние файлы:

```text
artifacts/rag-quality-comparison.json
artifacts/rag-quality-comparison.md
```

## Документация

Обновлён:

```text
README.md
```

Добавлены:

- описание `BASELINE_RAG` и `IMPROVED_RAG`;
- improved retrieval chain;
- новые CLI-параметры;
- примеры запуска baseline/improved;
- описание retrieval diagnostics в отчётах.

## Проверки

Были выполнены быстрые проверки:

```bash
.venv/bin/python -m py_compile $(find rag scripts -name '*.py' | sort)
.venv/bin/python scripts/ask.py --help
```

Также проверены без реального LLM-прогона:

- baseline retrieval через `HashingEmbeddingClient`;
- improved retrieval через `HashingEmbeddingClient`;
- формирование candidates/discarded chunks;
- генерация evaluation report с fake chat client;
- корректный учёт rewrite-токенов и rewrite-времени в improved метриках.

Полный `scripts/ask.py --control` с реальной Ollama не запускался, потому что теперь evaluation выполняет три режима и дополнительный rewrite-вызов для improved RAG.
