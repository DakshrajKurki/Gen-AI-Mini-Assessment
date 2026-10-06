# CodeAtlas — Map any repository in plain English

Paste a GitHub URL **or upload any project (zip / tar.gz / loose files — code, notebooks, docs, data, configs)** → a **locally running LLM** (Llama 3.2 by default) reads the code **live on screen** and explains it in simple language.

```
GitHub / Uploaded Repository → Code Processing → Local LLM → Backend → Frontend → Explanation
```

## Tech stack (as per assessment)
| Layer | Tools |
|---|---|
| Local GenAI | Python, Ollama **or** Hugging Face Transformers · Llama 3.2, Qwen 2.5, Phi-3, Gemma 2 |
| Backend | FastAPI, Pydantic, Uvicorn |
| Repo processing | GitPython, Python `ast`, basic file handling |
| Frontend | Streamlit |

## Run (macOS)
```bash
brew install ollama
cd ~/Downloads/codeatlas
chmod +x run.sh
./run.sh
```
Open http://localhost:8501. First run downloads `llama3.2:3b` and `qwen2.5:3b` (about 2 GB each).

## Any type of repository
CodeAtlas first classifies the project (web app, API, CLI, library, notebook/data-science, documentation, mobile app, infrastructure...) and lists what it contains, then the AI explains: **Project Overview · What's inside · What it allows users to do · Main Technologies · How it works · How to run it**. Pick **Upload a repo** to use a folder on your computer (zip it first); it never leaves your machine. Three colour themes (Aurora / Emerald / Sunset) are in the sidebar.

## What you see while it runs
- **Live preview** – paste the URL, press Enter: avatar, description, stars, language bar, folder structure (basic card if GitHub's rate limit is hit).
- **Model Console** – the model working in real time: state (loading → reading → generating), elapsed time, tokens, **tokens/second**, the last tokens it produced, a progress bar, and a terminal log of every step.
- **Fast or Deep mode** – Fast = one pass. **Deep = the model reads the 7 most important files one by one** and you watch what it understood from each, then it combines its notes.
- **Compare two models** – runs Llama 3.2 and Qwen 2.5 on the *same* cached input, side by side, with a speed and accuracy table.
- **Install models from the UI** – the sidebar shows which models are installed and has an Install button (Llama 3.2 1B/3B, Qwen 2.5, Qwen Coder, Phi-3, Gemma 2).
- **Explanation level** – Like I'm 12 / Beginner / Technical.

## How accuracy is improved
1. **Static analysis first** (`analyzer.py`): dependency files, Python `ast`, and import graphs give *verified facts*: stack, entry points, HTTP routes, which file imports which, key classes/functions, README summary. These are given to the LLM as ground truth. Tests, examples and docs are ignored so they don't pollute the result.
2. **Smart file ranking**: README → manifests → entry points → most-connected modules (by import graph) → the rest. Notebooks are read as code cells only.
3. **Low temperature** and strict prompt rules ("only state what is in the facts or code").
4. **Grounding check**: every technology the model lists is checked against the repo; unsupported ones are flagged ⚠️.
5. **Architecture map** drawn from real imports (not by the LLM).

## Tabs after a run
🗺️ Architecture map · 🔍 Verified facts · 🧠 Model working (log, per-file notes, the exact prompt) · 💬 Ask the repo (follow-up chat)

## API
`GET /health` · `GET /models` · `POST /models/pull` · `POST /preview` · `POST /upload` · `POST /explain/stream` · `POST /chat/stream`

## Options
- `GITHUB_TOKEN` env var raises GitHub's API limit (60/hour without it).
- `OLLAMA_MODEL`, `OLLAMA_URL`, `HF_MODEL` override defaults.
- Hugging Face engine: `pip install torch transformers accelerate`, then pick it in the sidebar.
