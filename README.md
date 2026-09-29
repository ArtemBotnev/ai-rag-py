# AI RAG Python Indexer

Локальный пайплайн индексации документов из `source/`.

## Что делает

- читает Markdown, Kotlin/KTS и PDF из `source/`;
- строит чанки тремя стратегиями:
  - `FIXED_SIZE`;
  - `STRUCTURE_AWARE`;
  - `PARAGRAPH_AWARE`;
- генерирует эмбеддинги через Ollama;
- сохраняет индекс в SQLite и JSONL;
- пишет отчёт сравнения стратегий chunking.

## Подготовка Ollama

По умолчанию используются две разные модели:

- embedding-модель `nomic-embed-text:latest`;
- LLM-модель `llama3.1:8b` для ответов на вопросы.

Установите Ollama системно:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Для индексации запустите Ollama и скачайте embedding-модель:

```bash
scripts/start_ollama.sh
```

Для ответов на вопросы запустите Ollama и скачайте LLM-модель:

```bash
scripts/start_ollama_llm.sh
```

Оба скрипта проверяют, что Ollama доступна, запускают `ollama serve` при необходимости и скачивают только свою модель.

Можно переопределить модели:

```bash
OLLAMA_MODEL=nomic-embed-text:latest scripts/start_ollama.sh
OLLAMA_LLM_MODEL=llama3.1:8b scripts/start_ollama_llm.sh
```

## Запуск

```bash
.venv/bin/python scripts/index_documents.py
```

Полезные параметры:

```bash
.venv/bin/python scripts/index_documents.py \
  --ollama-url http://localhost:11434 \
  --ollama-model nomic-embed-text:latest \
  --queries config/sample_queries.txt \
  --ollama-batch-size 16
```

Для запуска без Ollama есть явный fallback:

```bash
.venv/bin/python scripts/index_documents.py --embedding-provider hashing
```

## Первый RAG-запрос

После индексации можно задать вопрос к базе документов двумя способами.

Сравнить ответ модели без RAG и с RAG:

```bash
.venv/bin/python scripts/ask.py --compare "Какие архитектурные слои использует Now in Android?"
```

Только RAG-режим:

```bash
.venv/bin/python scripts/ask.py --rag "Как в проекте реализована синхронизация данных?"
```

Только режим без retrieval:

```bash
.venv/bin/python scripts/ask.py --no-rag "Как устроена модульная структура проекта?"
```

Полный прогон по 10 контрольным вопросам из `CONTROL_QUESTIONS.md`:

```bash
.venv/bin/python scripts/ask.py --control
```

Русский вариант вопросов лежит в `CONTROL_QUESTIONS.ru.md`:

```bash
.venv/bin/python scripts/ask.py --control --control-questions CONTROL_QUESTIONS.ru.md
```

Полезные параметры:

```bash
.venv/bin/python scripts/ask.py --control \
  --strategy PARAGRAPH_AWARE \
  --top-k 4 \
  --ollama-url http://localhost:11434 \
  --ollama-model nomic-embed-text:latest \
  --llm-model llama3.1:8b
```

`scripts/ask.py --control` для каждого вопроса запускает оба режима, считает покрытие ожидаемых фактов, попадание ожидаемых источников, время ответа и токены Ollama (`prompt_eval_count`, `eval_count`).

## Результаты

- `artifacts/rag-index.sqlite` - локальный SQLite-индекс;
- `artifacts/rag-chunks.jsonl` - чанки с текстом, метаданными и эмбеддингами;
- `artifacts/chunking-comparison.json` - машинно-читаемый отчёт;
- `artifacts/chunking-comparison.md` - отчёт сравнения стратегий.
- `artifacts/rag-quality-comparison.json` - машинно-читаемый отчёт сравнения RAG/no-RAG;
- `artifacts/rag-quality-comparison.md` - отчёт по 10 контрольным вопросам.

## Структура проекта

```text
scripts/index_documents.py  # исполняемая точка входа, аналог Kotlin Main.kt
scripts/ask.py              # вопросы к LLM в режимах с RAG / без RAG / control evaluation
scripts/start_ollama.sh     # запуск Ollama и загрузка embedding-модели
scripts/start_ollama_llm.sh # запуск Ollama и загрузка LLM-модели
CONTROL_QUESTIONS.md        # 10 контрольных вопросов по source/nowinandroid-main
CONTROL_QUESTIONS.ru.md     # русский вариант контрольных вопросов
config/sample_queries.txt   # вопросы для Retrieval Samples в отчёте
rag/cli.py                  # CLI-аргументы и выбор embedding-клиента
rag/common/models.py        # dataclass-модели
rag/common/tokenizer.py     # токенизация и token slices
rag/indexing/pipeline.py    # orchestration индексации, аналог IndexingPipeline
rag/indexing/document_loader.py  # загрузка Markdown, Kotlin/KTS и PDF
rag/indexing/chunkers.py    # FIXED_SIZE, STRUCTURE_AWARE, PARAGRAPH_AWARE
rag/indexing/storage.py     # SQLite и JSONL запись
rag/indexing/reporter.py    # chunking comparison report
rag/inference/embeddings.py # Ollama embeddings и hashing fallback
rag/inference/llm.py        # Ollama chat client с токенами и таймингами
rag/retrieval/retriever.py  # поиск релевантных чанков в SQLite
rag/qa/agent.py             # режимы WITH_RAG / WITHOUT_RAG
rag/evaluation/control_questions.py  # парсер контрольных вопросов
rag/evaluation/evaluation.py         # оценка качества и отчёт RAG/no-RAG
```

## Стратегии chunking

В проекте используются три стратегии разбиения документов на чанки для последующей генерации эмбеддингов и записи в локальный индекс SQLite.

## 1. Fixed-size chunking

### Идея

Документ рассматривается как единый поток токенов и режется на окна примерно одинакового размера.

### Как работает

- текст документа токенизируется;
- формируются окна фиксированного размера;
- соседние чанки частично перекрываются через `overlap`;
- если в конце остаётся небольшой хвост, он может быть присоединён к предыдущему чанку.

### Плюсы

- предсказуемый размер чанков;
- равномерное покрытие документа;
- простая и стабильная стратегия для индексации.

### Минусы

- границы чанков не совпадают со смысловыми границами;
- чанк может начинаться или заканчиваться посередине раздела, абзаца, функции или класса;
- retrieved-фрагменты могут содержать лишний контекст.

### Когда полезна

- для базового эталонного сравнения;
- для длинных документов, где важнее равномерность, чем структура;
- как простой baseline.

## 2. Structure-aware chunking

### Идея

Документ сначала разбивается на структурные единицы, а затем эти единицы агрегируются в чанки целевого размера.

### Как работает

- для `Markdown` используются заголовки и разделы;
- для `Kotlin` и `KTS` используются top-level declarations: `class`, `object`, `interface`, `fun`;
- для `PDF` используются абзацы;
- затем соседние структурные единицы объединяются в чанки, пока их размер не приблизится к целевому диапазону.

### Плюсы

- чанки ближе к смысловым блокам;
- поле `section` становится содержательным;
- результаты retrieval легче интерпретировать;
- лучше подходит для вопросов про конкретные классы, функции и секции документации.

### Минусы

- размер чанков менее равномерен;
- в корпусе с мелкой структурой появляются короткие чанки;
- логика сложнее, чем у fixed-size подхода.

### Когда полезна

- для документации с явной иерархией разделов;
- для кода, где важно сохранять границы классов и функций;
- для explainable retrieval.

## 3. Paragraph-aware chunking

### Идея

Базовой единицей выступает абзац, а не фиксированное окно и не формальная структура файла.

### Как работает

- текст разбивается на абзацы;
- абзацы последовательно объединяются в чанки;
- сборка продолжается, пока размер чанка не достигнет целевого диапазона;
- слишком большие абзацы могут быть дополнительно разрезаны.

### Плюсы

- лучше сохраняет естественный поток текста;
- хорошо подходит для README, статей, описаний архитектуры, PDF;
- менее жесткая стратегия, чем structure-aware.

### Минусы

- для кода обычно менее точна, чем structure-aware;
- на коротких абзацах могут появляться мелкие чанки;
- не использует формальные границы классов, функций и разделов.

### Когда полезна

- для текстовых документов и документации;
- как промежуточный вариант между fixed-size и structure-aware;
- для сравнения, как влияет уважение к естественным абзацам текста.

## Краткое сравнение

| Стратегия | Главная идея | Сильная сторона | Слабая сторона |
| --- | --- | --- | --- |
| Fixed-size | Ровные окна по токенам | Предсказуемость и стабильность | Слабая связь со смысловой структурой |
| Structure-aware | Разбиение по структуре документа и кода | Хорошая интерпретируемость и точность по секциям | Нерегулярный размер чанков |
| Paragraph-aware | Разбиение по абзацам | Сохранение естественного текстового потока | Хуже подходит для кода |

## Практический смысл сравнения

Сравнение трёх стратегий позволяет увидеть:

- насколько сильно структура документа влияет на качество retrieval;
- насколько полезна равномерность размеров чанков;
- как меняется распределение чанков по размерам;
- какая стратегия лучше работает на смешанном корпусе из `README`, `Markdown`, `Kotlin`, `KTS` и `PDF`.
