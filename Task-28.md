# Task 28

## Что сделано

Добавлен отдельный CLI-чат для сравнения облачной OpenAI LLM и локальной Ollama LLM поверх одного и того же RAG-контекста.

Новый чат:

- выполняет retrieval по SQLite-индексу один раз на пользовательский вопрос;
- передаёт одинаковый найденный контекст сначала в OpenAI, затем в локальную Ollama LLM;
- выводит только ответ каждой модели и время ответа;
- не печатает retrieval query, retrieved chunks, источники, цитаты, токены и debug-информацию;
- обрабатывает каждое сообщение как самостоятельный RAG-вопрос без памяти диалога;
- поддерживает прогон сценариев через `--scenario`;
- устойчиво читает русский ввод из терминала: UTF-8, fallback-кодировки вроде CP1251 и типичный mojibake.

Все сообщения запускают retrieval по текущему вопросу, а обе LLM получают одинаковый prompt/context без истории и task state.

## OpenAI client

Обновлён:

```text
rag/inference/llm.py
```

Добавлен:

```text
OpenAIResponsesChatClient
```

Клиент использует OpenAI Responses API через стандартный HTTP-клиент Python, без новой зависимости на SDK.

Поведение:

- API key берётся только из окружения `OPENAI_API_KEY`;
- модель берётся из `OPENAI_MODEL` или значения по умолчанию `gpt-5-mini`;
- системная инструкция передаётся через `instructions`;
- пользовательский prompt передаётся через `input`;
- `store` устанавливается в `false`;
- ответ приводится к общему формату `ChatCompletion`, совместимому с текущим `OllamaChatClient`.

CLI-аргумента для передачи OpenAI key нет, чтобы ключ не попадал в историю shell-команд.

## Compare chat

Добавлен:

```text
scripts/chat_llm_compare.py
```

Интерактивный запуск:

```bash
.venv/bin/python scripts/chat_llm_compare.py
```

Одиночный вопрос:

```bash
.venv/bin/python scripts/chat_llm_compare.py "Какие архитектурные слои используются?"
```

Прогон сценария:

```bash
.venv/bin/python scripts/chat_llm_compare.py --scenario scenarios/chat_feature.md
```

На каждый RAG-вопрос вывод имеет компактный вид:

```text
OpenAI (<model>)
Time: <ms> ms
<answer>

Ollama (<model>)
Time: <ms> ms
<answer>
```

Перед пользовательским prompt и после вопроса добавлены пустые строки, чтобы интерактивный вывод легче читался.

## Stateless compare chat

Compare chat больше не использует память обычного чата:

```text
rag/qa/chat.py
```

Используются:

- `extract_scenario_messages(...)` для markdown-сценариев.

Retrieval query равен текущему пользовательскому сообщению. Prompt содержит только вопрос и найденный RAG-контекст.

## Bash-запуск

Добавлен:

```text
scripts/run_llm_compare_chat.sh
```

Скрипт:

- находит Python из `.venv/bin/python` или переменной `PYTHON`;
- проверяет, что `OPENAI_API_KEY` уже задан в окружении;
- предупреждает, если Ollama не отвечает по `OLLAMA_HOST`;
- запускает `scripts/chat_llm_compare.py`;
- пробрасывает пользовательские аргументы дальше в Python CLI.

Интерактивный запуск:

```bash
scripts/run_llm_compare_chat.sh
```

Одиночный вопрос:

```bash
scripts/run_llm_compare_chat.sh "Какие архитектурные слои используются?"
```

Прогон сценария:

```bash
scripts/run_llm_compare_chat.sh --scenario scenarios/chat_feature.md
```

## Документация

Обновлён:

```text
README.md
```

Добавлен раздел про сравнение OpenAI и локальной LLM поверх RAG:

- подготовка локальной LLM через `scripts/start_ollama_llm.sh`;
- требование, что `OPENAI_API_KEY` должен быть уже задан в окружении;
- запуск `scripts/run_llm_compare_chat.sh`;
- одиночный вопрос;
- `--scenario`;
- переопределение моделей через `OPENAI_MODEL`, `OLLAMA_LLM_MODEL` и CLI-параметры.

## Проверки

Выполнены:

```bash
find rag scripts -name '*.py' -print0 | xargs -0 .venv/bin/python -m py_compile
.venv/bin/python scripts/chat_llm_compare.py --help
bash -n scripts/run_llm_compare_chat.sh
```

Проверен интерактивный ввод русских сообщений в UTF-8 и CP1251.

Полный реальный LLM-прогон требует:

- заданный `OPENAI_API_KEY` в окружении;
- доступ к OpenAI API;
- запущенный Ollama chat model.
