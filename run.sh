#!/usr/bin/env bash
# CodeAtlas launcher: FastAPI backend (port 8000) + Streamlit frontend (port 8501).
set -e
cd "$(dirname "$0")"

# free the ports if an old run is still holding them
lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
lsof -ti tcp:8501 | xargs kill -9 2>/dev/null || true

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

# Ollama: start the server if needed, then make sure the Llama + Qwen models are installed
if command -v ollama >/dev/null 2>&1; then
  if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    (ollama serve >/dev/null 2>&1 &)
    sleep 3
  fi
  for m in llama3.2:3b qwen2.5:3b; do
    ollama list | grep -q "^$m" || ollama pull "$m"
  done
else
  echo "Ollama not found. Install with: brew install ollama   (or choose the Hugging Face engine in the sidebar)"
fi

(cd backend && uvicorn main:app --port 8000) &
BACK=$!
trap "kill $BACK 2>/dev/null" EXIT
sleep 2
cd frontend && streamlit run app.py --server.port 8501
