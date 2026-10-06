# Viva Guide, Testing and Troubleshooting

## Part 1: Viva questions and answers

**Why GitPython?**
It lets Python run `git clone` without writing shell commands. We do a shallow clone (`depth=1`), which downloads only the latest version and keeps it fast. It needs Git installed on the machine.

**Why Streamlit?**
It builds a web interface in pure Python: no HTML, CSS or JavaScript. It has ready-made widgets (text box, button, spinner, metrics, expanders) and deploys free on Streamlit Cloud, which suits a college demo.

**Why FastAPI?**
It exposes the same pipeline as a REST API (`POST /explain`, `GET /health`) so another app could use it. It validates the request with Pydantic and gives free interactive docs at `/docs`. The Streamlit app does not depend on it, so it still works on Streamlit Cloud.

**What does Ollama do?**
Ollama is a program that downloads and runs open-source LLMs on your own computer and offers a simple HTTP API on `localhost:11434`. We call `POST /api/generate` with `stream: false` and a low temperature (0.1).

**What is Qwen 2.5?**
A family of open-source language models from Alibaba. We use the smallest instruction-tuned one, with 0.5 billion parameters. "Instruct" means it was trained to follow instructions and chat.

**Why a small model?**
It runs on a student laptop CPU (about 400 MB through Ollama), loads quickly, needs no GPU or paid API, and fits on free cloud hosting. The trade-off is lower quality, which is why we give it a small, well-selected context and a strict prompt.

**What is Hugging Face Transformers?**
A Python library that downloads and runs models from the Hugging Face Hub. We use `AutoTokenizer` (text to numbers) and `AutoModelForCausalLM` (predicts the next token), then `generate()` to write the answer.

**How is the repository cloned?**
`parse_github_url()` validates the URL with `urllib.parse`. `clone_repository()` creates a temp folder and calls `Repo.clone_from(url, folder, depth=1)`. Git is told never to ask for a password, so private repositories fail quickly. A `finally` block calls `cleanup_repository()` so nothing stays on disk.

**How are files selected?**
The walker skips junk folders, tests, lock files, binaries and files over 300 KB. Each remaining file gets a score: README and config files come first, then entry points like `main.py` or `app.py`, then source files that are shallow in the tree, in `src/`/`lib/`/`app/` folders, and have real content. Example or demo folders score lower. The top files are taken until the limits are reached (10 files, about 8,000 characters).

**How is the repository context built?**
One text block containing: repository name, file counts, languages (from file extensions), dependencies (from `requirements.txt`, `package.json`, `pyproject.toml` or `pom.xml`), the directory tree, the README (with badges removed) and the selected code snippets.

**How is the prompt created?**
`llm.py` has a system prompt ("You are a software project explainer ... Do not invent features ...") and a user prompt that contains the context followed by the seven required headings. Both are combined as chat messages.

**How does the LLM generate the explanation?**
It predicts one token (word piece) at a time, each time using the prompt and everything it has written so far. With a low temperature (Ollama) or greedy decoding (Hugging Face) it stays close to the facts in the context.

**Local vs cloud inference?**
Local: Ollama runs the model as a separate server on your computer. Cloud: Streamlit Cloud only runs our Python app, so `transformers` loads the model inside the app process. Same model family, different runner.

**Why can't Ollama just be used on Streamlit Cloud?**
`localhost:11434` means "this very machine". On Streamlit Cloud that machine is Streamlit's server, where no Ollama is installed, and you cannot install a background service there. So `explain_repository()` first checks Ollama, and if it is missing it falls back to Hugging Face.

**Why is the `sys.path` fix needed?**
Streamlit runs `frontend/app.py` and puts only the `frontend/` folder on Python's import path. Our code lives in `backend/`, one level up, so `import backend...` fails with `ModuleNotFoundError`. The first lines of `app.py` add the project root to `sys.path` before any backend import. (`backend/__init__.py` also has to exist so Python treats `backend` as a package.)

**How does the app explain the whole repository with such a small model?**
A 0.5B model cannot read a whole repository at once, so the work is split into small steps (a "map then combine" approach). First a static scan builds an index of every folder and file with its classes and functions. That index goes into the main prompt. Then, in Deep scan, the model reads one important file at a time and writes one sentence about it. Finally it writes a sentence for each folder from those file sentences. Each step fits easily in the model's context window.

**What is the difference between "Files read by AI" and "Files described"?**
"Files read by AI" are the README, config files and source files sent whole to the model for the main overview. "Files described" are the files the model summarised one by one in Deep scan.

**What are the limitations?**
A small model can make mistakes or repeat itself. Only a slice of a big repository is read. Only public GitHub repositories work. Cloud inference is slow on a shared CPU. Output quality varies from run to run and from repository to repository.

**How do you know the explanation is not hard-coded?**
There is no explanation text in the code. The app has a button to show the exact context sent to the model, and different repositories produce different answers.

---

## Part 2: Testing instructions

1. **URL validation**: try each of these. All except the first should show a friendly error.
   - `https://github.com/psf/requests` → accepted
   - `https://github.com/psf/requests/blob/main/setup.py` → "looks like a link to a file ..."
   - `https://gitlab.com/a/b` → "Only public github.com repositories ..."
   - `https://github.com/nobody-xyz/does-not-exist-123` → "Repository not found ..."
   - empty input → "Please enter a GitHub repository URL."
2. **Local run with Ollama**: start Ollama, run the Streamlit app, analyze `psf/requests`. The engine metric should say **Ollama**.
3. **Fallback**: stop Ollama (or set `OLLAMA_URL=http://localhost:9`) and analyze again. The first run downloads the Hugging Face model, then the engine should say **Hugging Face**.
4. **Deep scan**: tick "Deep scan", analyze a repository and open the **Repository map** and **File guide** tabs. Each folder and top files should have a sentence written by the AI. Untick it and analyze again to see the faster quick scan.
5. **Different languages**: try `expressjs/express` (JavaScript) and `pallets/flask` (Python).
6. **API**: run `uvicorn backend.main:app --reload`, open `/docs`, call `GET /health` (expect `{"status":"ok"}`) and `POST /explain`.
7. **Cleanup**: while or after running, check your temp folder (`%TEMP%` on Windows). No `repolens_*` folders should remain.
8. **Streamlit Cloud**: open the deployed URL, analyze a small repository and wait for the first (slow) run to finish.

---

## Part 3: Common errors and fixes

| Error / symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'backend'` | Project root not on `sys.path`, or `backend/__init__.py` missing | Keep the `sys.path` block at the top of `frontend/app.py`, before the backend imports. Check `backend/__init__.py` exists. Run commands from the project root. |
| "Git is not installed on this machine" | Git missing | Install Git from https://git-scm.com and restart the terminal |
| "Repository not found. It may not exist, or it may be private" | Typo, deleted or private repository | Check the URL in a browser. Only public repositories work. |
| "Could not download the repository from GitHub" | No internet or slow connection | Check the connection and retry |
| "This looks like a link to a file, folder or page" | A `/blob/...` or `/tree/...` link was pasted | Paste only `https://github.com/owner/repo` |
| "Ollama is not running" / app uses Hugging Face unexpectedly | Ollama not started | Start the Ollama app or run `ollama serve`. Test at http://localhost:11434 |
| "model 'qwen2.5:0.5b' is not installed" | Model not pulled | `ollama pull qwen2.5:0.5b` |
| "The AI model took too long to answer" | Slow CPU or very large prompt | Try a smaller repository, close other programs, and retry |
| "The AI model could not be downloaded from Hugging Face" | No internet, or Hugging Face blocked | Check the connection and retry. Behind a college firewall, use Ollama locally instead. |
| "The server ran out of memory" (Streamlit Cloud) | Free tier RAM limit | Reboot the app from the Streamlit dashboard and try a smaller repository, or run locally |
| `'streamlit' is not recognized` | Virtual environment not active | Run `venv\Scripts\activate`, or use `python -m streamlit run frontend/app.py` |
| `pip install` of torch is very slow or huge | The default Linux wheel includes CUDA | Keep the `--extra-index-url .../whl/cpu` line in `requirements.txt` |
| Deep scan is slow | One AI call per file and folder | Lower the "Files to describe" slider (try 5), or untick Deep scan |
| First online analysis takes minutes | The model (about 1 GB) is downloading and loading | Wait. Later runs reuse the cached model. |
| Streamlit Cloud build fails on torch | Python version too new | In Advanced settings choose Python 3.11 and redeploy |
