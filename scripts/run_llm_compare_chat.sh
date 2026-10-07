#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
BASE_URL="http://${HOST}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Error: Python executable not found: ${PYTHON}"
  echo "Create the virtual environment first or pass PYTHON=/path/to/python."
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Error: OPENAI_API_KEY is not set."
  echo "Export it before running this script:"
  echo "export OPENAI_API_KEY=\"...\""
  exit 1
fi

if ! curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1; then
  echo "Warning: Ollama is not responding at ${BASE_URL}."
  echo "Start it in another terminal with:"
  echo "scripts/start_ollama_llm.sh"
  echo
fi

exec "${PYTHON}" "${PROJECT_ROOT}/scripts/chat_llm_compare.py" \
  --ollama-url "${BASE_URL}" \
  "$@"
