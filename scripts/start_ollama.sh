#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-nomic-embed-text:latest}"
HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
BASE_URL="http://${HOST}"
export OLLAMA_HOST="${HOST}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: ollama command not found."
  echo
  echo "Install Ollama system-wide on Linux:"
  echo "curl -fsSL https://ollama.com/install.sh | sh"
  echo
  echo "Then restart the terminal or make sure ollama is available in PATH."
  exit 1
fi

echo "Using Ollama model: ${MODEL}"

if curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already running at ${BASE_URL}"
else
  echo "Starting Ollama server at ${BASE_URL}..."
  ollama serve &
  OLLAMA_PID=$!

  for _ in $(seq 1 30); do
    if curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1; then
      echo "Ollama server is ready"
      break
    fi
    sleep 1
  done

  if ! curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1; then
    echo "Error: Ollama server did not become ready in time"
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
    exit 1
  fi
fi

if ollama list | awk '{print $1}' | grep -Fxq "${MODEL}"; then
  echo "Model is already available: ${MODEL}"
else
  echo "Pulling model: ${MODEL}"
  ollama pull "${MODEL}"
fi

echo
echo "Ollama is ready. Run the indexer in another terminal:"
echo ".venv/bin/python scripts/index_documents.py"

if [[ -n "${OLLAMA_PID:-}" ]]; then
  echo
  echo "Keeping Ollama server attached to this terminal. Press Ctrl+C to stop it."
  wait "${OLLAMA_PID}"
fi
