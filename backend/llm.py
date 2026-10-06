"""
llm.py
======
Sends the repository context to a small LLM and returns the explanation.

Two ways to run the SAME model family (Qwen 2.5, 0.5B parameters):

  LOCAL   -> Ollama server on this computer      model "qwen2.5:0.5b"
  ONLINE  -> Hugging Face Transformers in-process model "Qwen/Qwen2.5-0.5B-Instruct"

explain_repository() tries Ollama first and falls back to Hugging Face, so the
exact same code works on a laptop (Ollama running) and on Streamlit Cloud
(no Ollama available).

Environment variables (all optional)
------------------------------------
OLLAMA_URL      default http://localhost:11434
OLLAMA_MODEL    default qwen2.5:0.5b
LLM_BACKEND     auto (default) | ollama | huggingface
HF_DTYPE        bfloat16 (default on CPU) | float32
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# ------------------------------------------------------------- settings -----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_TIMEOUT = 300           # seconds to wait for Ollama to finish writing
OLLAMA_NUM_CTX = 8192          # keep constant: changing it makes Ollama reload the model

HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HF_MAX_INPUT_TOKENS = 4500     # context is trimmed if the prompt is longer
HF_MAX_NEW_TOKENS = 600
HF_MAX_SECONDS = 240           # generation stops after this long

LLM_BACKEND = os.getenv("LLM_BACKEND", "auto").lower()

# ---------------------------------------------------------------- prompt ----
SYSTEM_PROMPT = (
    "You are a software project explainer.\n\n"
    "Analyze the provided GitHub repository context and explain the project in simple "
    "language suitable for a college student.\n\n"
    "Only describe functionality that is supported by the provided repository context.\n"
    "Do not invent features.\n"
    "Do not reproduce large sections of source code.\n"
    "Explain the actual project."
)

USER_TEMPLATE = """Below is the context extracted from a GitHub repository: its folder structure, README, configuration files and important source files.

{context}

---
Using ONLY the context above, explain this project. Write in Markdown using exactly these seven headings, in this order:

## 1. Project Overview
## 2. Main Features
## 3. Project Structure
## 4. How the Application Works
## 5. Technologies Used
## 6. Important Code Components
## 7. Simple Summary

Guidelines:
- The context contains a "Repository index" that lists the folders and the important files with the classes/functions they define. Use it to cover the WHOLE repository, not just the files whose code is shown.
- Under "Project Structure", go through each main folder and say what it contains.
- Under "How the Application Works", mention the entry point, the main workflow and the data flow when they are visible.
- Under "Important Code Components", mention the important modules/files and the dependencies.
- If something is not visible in the context, write "not clear from the provided files" instead of guessing.
- Keep each section short (bullet points are fine) and finish with a 2-3 sentence Simple Summary."""


def build_messages(context_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context_text)},
    ]


# ---------------------------------------------------------------- results ---
class LLMError(Exception):
    """The explanation could not be generated. The message is safe for users."""


class OllamaUnavailable(Exception):
    """Ollama is not running, or the model is not installed (triggers the fallback)."""


@dataclass
class ExplanationResult:
    text: str
    engine: str     # e.g. "Ollama · qwen2.5:0.5b"


# ------------------------------------------------------------ Ollama (local) -
def ollama_status() -> tuple[bool, str]:
    """(is_ready, reason_if_not). Quick check that Ollama runs and has our model."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        resp.raise_for_status()
        names = {m.get("name") for m in resp.json().get("models", [])}
    except (requests.RequestException, ValueError):
        return False, f"Ollama is not running at {OLLAMA_URL}."
    if OLLAMA_MODEL in names or f"{OLLAMA_MODEL}:latest" in names:
        return True, ""
    return False, f"Ollama is running but model '{OLLAMA_MODEL}' is not installed (run: ollama pull {OLLAMA_MODEL})."


def _ollama_generate(system: str, prompt: str, num_predict: int) -> str:
    """One call to Ollama's /api/generate. Raises OllamaUnavailable or LLMError."""
    payload = {
        "model": OLLAMA_MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": OLLAMA_NUM_CTX, "num_predict": num_predict},
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=(5, OLLAMA_TIMEOUT))
    except requests.exceptions.ConnectionError as exc:      # includes connect timeouts
        raise OllamaUnavailable(f"Could not connect to Ollama at {OLLAMA_URL}.") from exc
    except requests.exceptions.Timeout as exc:
        raise LLMError(
            "The AI model took too long to answer. Try a smaller repository or try again."
        ) from exc
    except requests.RequestException as exc:
        logger.warning("Ollama request failed: %s", exc)
        raise LLMError("The local AI model (Ollama) returned an error.") from exc

    if resp.status_code == 404:
        raise OllamaUnavailable(f"Model '{OLLAMA_MODEL}' is not installed in Ollama.")
    if not resp.ok:
        logger.warning("Ollama HTTP %s: %s", resp.status_code, resp.text[:300])
        raise LLMError("The local AI model (Ollama) returned an error.")
    try:
        text = (resp.json().get("response") or "").strip()
    except ValueError as exc:
        raise LLMError("The local AI model (Ollama) sent an unreadable reply.") from exc
    if not text:
        raise LLMError("The AI model returned an empty answer. Please try again.")
    return text


def _explain_with_ollama(context_text: str) -> str:
    ready, reason = ollama_status()
    if not ready:
        raise OllamaUnavailable(reason)
    messages = build_messages(context_text)
    return _ollama_generate(messages[0]["content"], messages[1]["content"], num_predict=900)


# -------------------------------------------------- Hugging Face (in-process) -
def _make_cache():
    """Cache the model so it loads once, not on every click.

    Inside a running Streamlit app we use @st.cache_resource (as required).
    Under FastAPI/CLI there is no Streamlit runtime, so we use functools.lru_cache,
    which does the same job.
    """
    try:
        from streamlit.runtime import exists as streamlit_runtime_exists

        if streamlit_runtime_exists():
            import streamlit as st

            return st.cache_resource(show_spinner=False)
    except Exception:  # streamlit missing or API changed -> plain Python cache
        pass
    return functools.lru_cache(maxsize=1)


_cache = _make_cache()
_hf_lock = threading.Lock()   # one generation at a time (CPU, small machine)


@_cache
def load_hf_model():
    """Download (first time only) and load Qwen2.5-0.5B-Instruct. Returns (tokenizer, model)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading %s ...", HF_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, low_cpu_mem_usage=True)

    if torch.cuda.is_available():
        model = model.half().to("cuda")
    else:
        # bfloat16 halves RAM (~1 GB instead of ~2 GB), important on Streamlit Cloud.
        # Set HF_DTYPE=float32 if your CPU is old and bfloat16 feels slow.
        wanted = os.getenv("HF_DTYPE", "bfloat16").lower()
        model = model.float() if wanted == "float32" else model.to(torch.bfloat16)
    model.eval()
    return tokenizer, model


def _chat_prompt(tokenizer, messages: list[dict]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback if a tokenizer has no chat template
    return "\n\n".join(m["content"] for m in messages) + "\n\n"


def _hf_generate(messages: list[dict], max_new_tokens: int, text_for_trim: str | None = None) -> str:
    """Run the Hugging Face model on a chat. If `text_for_trim` is the context text
    inside the user message, it is shortened until the prompt fits the token limit."""
    import torch

    tokenizer, model = load_hf_model()

    for _ in range(3):
        prompt = _chat_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        n_tokens = inputs["input_ids"].shape[1]
        if text_for_trim is None or n_tokens <= HF_MAX_INPUT_TOKENS:
            break
        text_for_trim = text_for_trim[: int(len(text_for_trim) * HF_MAX_INPUT_TOKENS / n_tokens * 0.9)]
        messages = build_messages(text_for_trim)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with _hf_lock, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_time=HF_MAX_SECONDS,
            do_sample=False,                 # deterministic, factual
            repetition_penalty=1.1,          # small models like to repeat themselves
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not answer:
        raise LLMError("The AI model returned an empty answer. Please try again.")
    return answer


def _explain_with_huggingface(context_text: str) -> str:
    return _hf_generate(build_messages(context_text), HF_MAX_NEW_TOKENS, text_for_trim=context_text)


def _friendly_hf_error(exc: Exception) -> str:
    text = str(exc).lower()
    if isinstance(exc, ImportError):
        return "The AI libraries (transformers / torch) are not installed. Run: pip install -r requirements.txt"
    if isinstance(exc, MemoryError) or "out of memory" in text or "cannot allocate" in text:
        return "The server ran out of memory while running the AI model. Please try again later or run the app locally with Ollama."
    if isinstance(exc, OSError) or "connection" in text or "huggingface.co" in text:
        return "The AI model could not be downloaded from Hugging Face. Check your internet connection and try again."
    return "The AI model failed to generate an explanation. Please try again."


# --------------------------------------------------------------- public API -
def explain_repository(repository_context) -> ExplanationResult:
    """Generate the explanation. Accepts a RepositoryContext or a plain string.

    1. Try Ollama (local).
    2. If Ollama is unavailable -> Hugging Face Transformers (works on Streamlit Cloud).
    Raises LLMError with a friendly message if everything fails.
    """
    context_text = getattr(repository_context, "text", repository_context)
    if not isinstance(context_text, str) or not context_text.strip():
        raise LLMError("There was no repository content to explain.")

    if LLM_BACKEND in ("auto", "ollama"):
        try:
            text = _explain_with_ollama(context_text)
            return ExplanationResult(text, f"Ollama · {OLLAMA_MODEL}")
        except OllamaUnavailable as exc:
            logger.info("Ollama not usable (%s)", exc)
            if LLM_BACKEND == "ollama":
                raise LLMError(str(exc)) from exc
            logger.info("Falling back to Hugging Face Transformers.")

    try:
        text = _explain_with_huggingface(context_text)
        return ExplanationResult(text, f"Hugging Face · {HF_MODEL}")
    except LLMError:
        raise
    except Exception as exc:  # never leak a traceback to the UI
        logger.exception("Hugging Face inference failed")
        raise LLMError(_friendly_hf_error(exc)) from exc


# ===================================================== deep scan (whole repo) ==
# After the main explanation, the model also writes a one-sentence description of
# each important file ("File Guide") and of each top-level folder. Each call is small
# and quick, and together they let the app explain *what is in* the whole repository.
FILE_SYSTEM_PROMPT = (
    "You are a code reader. In ONE or TWO short plain-English sentences (maximum 35 words), "
    "say what the given file does in its project. Use only the code shown. "
    "Do not write code, markdown or a preamble. Do not start with 'This file'."
)
FOLDER_SYSTEM_PROMPT = (
    "You are a software project explainer. Given a folder name and the files in it, say in ONE or "
    "TWO short plain-English sentences (maximum 40 words) what this folder is for. "
    "Use only the information given. Do not write markdown or a preamble."
)
MAX_FOLDERS_DESCRIBED = 8
_MAX_FAILURES_IN_A_ROW = 3


def pick_engine() -> str:
    """'ollama' if it is running with our model, otherwise 'huggingface'."""
    if LLM_BACKEND == "huggingface":
        return "huggingface"
    ready, reason = ollama_status()
    if ready:
        return "ollama"
    if LLM_BACKEND == "ollama":
        raise LLMError(reason)
    return "huggingface"


def _short_answer(engine: str, system: str, user: str, max_tokens: int = 90) -> str:
    if engine == "ollama":
        return _ollama_generate(system, user, num_predict=max_tokens)
    return _hf_generate(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens
    )


def _clean_summary(text: str) -> str:
    """Make a model answer fit on a card: one paragraph, no markdown, max ~2 sentences."""
    text = " ".join(text.replace("`", "").replace("*", "").replace("#", "").split()).strip(" \"'-")
    sentences = [x.strip() for x in text.replace("? ", ". ").split(". ") if x.strip()]
    text = ". ".join(sentences[:2])
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text[:320]


def _run_batch(jobs, engine, progress, label):
    """jobs = [(name, system, user, setter)]. Stops early after repeated failures."""
    failures = 0
    for i, (name, system, user, setter) in enumerate(jobs, start=1):
        if progress:
            progress(i - 1, len(jobs), f"{label} {name}")
        try:
            summary = _clean_summary(_short_answer(engine, system, user))
            if summary:
                setter(summary)
            failures = 0
        except (LLMError, OllamaUnavailable) as exc:
            logger.warning("Could not describe %s: %s", name, exc)
            failures += 1
        except Exception as exc:  # e.g. Hugging Face runtime problems
            logger.warning("Could not describe %s: %s", name, exc)
            failures += 1
        if failures >= _MAX_FAILURES_IN_A_ROW:
            raise LLMError("The AI model stopped answering while describing the files.")
    if progress:
        progress(len(jobs), len(jobs), "done")


def describe_files(context, limit: int = 15, progress=None) -> int:
    """Fill `summary` of the most important source files. Returns how many were described.

    progress(done, total, label) is called before each file so the UI can show a bar.
    Partial results are kept if the model fails half way.
    """
    items = sorted((i for i in context.inventory if i.snippet), key=lambda i: -i.importance)[:limit]
    if not items:
        return 0
    engine = pick_engine()
    jobs = []
    for item in items:
        user = (
            f"Project: {context.repo_name}\nFile: {item.path}\n\n"
            f"```\n{item.snippet}\n```\n\nWhat does this file do?"
        )
        jobs.append((item.path, FILE_SYSTEM_PROMPT, user, lambda text, it=item: setattr(it, "summary", text)))
    _run_batch(jobs, engine, progress, "Reading")
    return sum(1 for i in items if i.summary)


def describe_folders(context, progress=None) -> int:
    """Fill `summary` of each top-level folder using the file summaries. Returns count."""
    by_path = {i.path: i for i in context.inventory}
    folders = [f for f in context.folders if f.files][:MAX_FOLDERS_DESCRIBED]
    if not folders:
        return 0
    engine = pick_engine()
    jobs = []
    for folder in folders:
        lines = []
        for path in folder.files[:10]:
            item = by_path.get(path)
            if item is None:
                continue
            if item.summary:
                lines.append(f"- {path}: {item.summary}")
            else:
                extra = f" (defines {', '.join(item.symbols[:5])})" if item.symbols else ""
                lines.append(f"- {path} [{item.language}]{extra}")
        title = "the project root (top-level files)" if folder.name == "(root)" else f"the folder '{folder.name}/'"
        user = (
            f"Project: {context.repo_name}\nThis is {title}, containing {folder.file_count} files. "
            f"Main files:\n" + "\n".join(lines) + "\n\nWhat is this folder for?"
        )
        jobs.append((folder.name, FOLDER_SYSTEM_PROMPT, user, lambda text, f=folder: setattr(f, "summary", text)))
    _run_batch(jobs, engine, progress, "Summarizing folder")
    return sum(1 for f in folders if f.summary)
