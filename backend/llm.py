"""Local LLM layer: Ollama (default) or Hugging Face Transformers, plus prompt builders."""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from threading import Thread
from typing import Dict, Iterator, List, Optional

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

CATALOG = [
    {"id": "llama3.2:3b", "name": "Llama 3.2 · 3B", "vendor": "Meta", "icon": "🦙", "size": "2.0 GB",
     "blurb": "Meta's compact Llama. Strong instruction-following and clean, readable explanations."},
    {"id": "llama3.2:1b", "name": "Llama 3.2 · 1B", "vendor": "Meta", "icon": "🦙", "size": "1.3 GB",
     "blurb": "Smallest Llama. Fastest and light on RAM, but less accurate on big repos."},
    {"id": "qwen2.5:3b", "name": "Qwen 2.5 · 3B", "vendor": "Alibaba", "icon": "🐉", "size": "1.9 GB",
     "blurb": "Well-rounded small model with good multilingual and structured-output skills."},
    {"id": "qwen2.5-coder:3b", "name": "Qwen 2.5 Coder · 3B", "vendor": "Alibaba", "icon": "💻", "size": "1.9 GB",
     "blurb": "Code-specialised. Best at reading source files and spotting what they do."},
    {"id": "phi3:mini", "name": "Phi-3 Mini · 3.8B", "vendor": "Microsoft", "icon": "🔷", "size": "2.2 GB",
     "blurb": "Compact reasoning-focused model from Microsoft."},
    {"id": "gemma2:2b", "name": "Gemma 2 · 2B", "vendor": "Google", "icon": "💎", "size": "1.6 GB",
     "blurb": "Google's lightweight open model; quick and tidy summaries."},
]

SYSTEM = (
    "You are a careful senior engineer who explains any kind of repository (apps, libraries, notebooks, datasets, documentation, infrastructure) to people in simple language. "
    "You only state things supported by the facts and code you are given. "
    "If something is unclear from the material, say it is not clear from the code. Never invent features, files or libraries."
)

LEVELS = {
    "eli5": "Audience: a curious 12-year-old. Use everyday analogies and very short sentences. Avoid jargon; if a technical word is unavoidable, explain it in a few words.",
    "beginner": "Audience: a beginner who knows basic programming. Use plain, simple English and short sentences.",
    "technical": "Audience: a software engineer. Be concise but precise: name the key files, modules, routes and the data flow.",
}


# ---------------------------------------------------------------- catalogue
def list_ollama_models() -> List[str]:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def pick_default(installed: List[str]) -> Optional[str]:
    names = set(installed) | {n[:-7] for n in installed if n.endswith(":latest")}
    for pref in (DEFAULT_OLLAMA_MODEL, "llama3.2:3b", "qwen2.5:3b", "qwen2.5-coder:3b"):
        if pref in names:
            return pref
    return installed[0] if installed else DEFAULT_OLLAMA_MODEL


# ------------------------------------------------------------------ prompts
FORMAT_BASE = """Write the answer in EXACTLY this markdown format, starting directly with the first heading (no preamble):

## Project Overview
2 or 3 simple sentences: what this repository is about, who it is for and what problem it solves.

## What's inside
- 4 to 8 bullets describing the main folders and files and what each one holds (use the real names from FILE STRUCTURE and the REPOSITORY CONTENTS facts).

## What it allows users to do
- 3 to 6 short bullets of what a user can do with it. For repositories that are not software (documentation, datasets, notes, configuration), say what a reader can learn or use it for.

## Main Technologies
- **Name** – what it is used for here (only technologies listed in the VERIFIED FACTS or visible in the code; for non-software repos list the formats and tools used)

## How it works
4 to 7 short sentences describing the flow: what the user interacts with, how the parts talk to each other, where data is stored or processed, and what comes back to the user. For non-software repos, describe how the content is organised and meant to be used.

## How to run or use it
- Numbered steps ONLY if the README, run scripts or dependency files show them. Otherwise write exactly: Not described in the repository."""

FORMAT_TECH_EXTRA = """

## Key Files
- `path` – its role (only files that appear in the FILE STRUCTURE)"""


def explain_messages(repo: str, tree: List[str], facts_txt: str, level: str,
                     files: Optional[List[dict]] = None, summaries: Optional[List[dict]] = None) -> List[dict]:
    parts = [f'Explain the repository "{repo}".', "", LEVELS.get(level, LEVELS["beginner"]), "",
             "VERIFIED FACTS (found by static analysis of the repository – trust these over guesses):",
             facts_txt or "(none found)", "", "FILE STRUCTURE:", "\n".join(tree[:60])]
    if summaries:
        parts += ["", "FILE-BY-FILE NOTES (you wrote these while reading each file):"]
        parts += [f"- {s['path']}: {s['summary']}" for s in summaries]
    if files:
        parts += ["", "SOURCE CODE (trimmed):"]
        parts += [f"### FILE: {f['path']}\n{f['content']}" for f in files]
    parts += ["", FORMAT_BASE + (FORMAT_TECH_EXTRA if level == "technical" else "")]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


def file_messages(path: str, content: str) -> List[dict]:
    user = (f"File: {path}\n```\n{content}\n```\n\n"
            "In ONE or TWO plain sentences, say what this file is, what it contains or what it is responsible for in the project. "
            "Mention concrete features, routes, data or commands it handles. Use only what is visible in the file.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def chat_messages(repo: str, tree: List[str], facts_txt: str, files: List[dict],
                  history: List[dict], question: str) -> List[dict]:
    ctx, used = [], 0
    for f in files:
        if used > 9000:
            break
        chunk = f["content"][:3000]
        used += len(chunk)
        ctx.append(f"### FILE: {f['path']}\n{chunk}")
    system = (
        f'{SYSTEM}\n\nYou are answering follow-up questions about the GitHub repository "{repo}". '
        "Answer concisely in simple language using ONLY the material below. "
        "If the answer is not in the material, say so.\n\n"
        f"VERIFIED FACTS:\n{facts_txt}\n\nFILE STRUCTURE:\n" + "\n".join(tree[:60]) + "\n\nSOURCE CODE:\n" + "\n\n".join(ctx)
    )
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    msgs.append({"role": "user", "content": question})
    return msgs


# ------------------------------------------------------------------- Ollama
def ollama_chat(model: str, messages: List[dict], num_predict: int = 900, temperature: float = 0.2) -> Iterator[dict]:
    payload = {
        "model": model, "messages": messages, "stream": True, "keep_alive": "10m",
        "options": {"temperature": temperature, "top_p": 0.9, "repeat_penalty": 1.1,
                    "num_ctx": 8192, "num_predict": num_predict},
    }
    with httpx.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload,
                      timeout=httpx.Timeout(10.0, read=600.0)) as r:
        if r.status_code != 200:
            r.read()
            try:
                msg = r.json().get("error", r.text)
            except Exception:
                msg = r.text
            if r.status_code == 404 or "not found" in msg.lower():
                raise RuntimeError(f"Model '{model}' is not installed. Run:  ollama pull {model}  (or use the Install button in the sidebar)")
            raise RuntimeError(msg)
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if d.get("error"):
                raise RuntimeError(d["error"])
            piece = (d.get("message") or {}).get("content", "")
            if piece:
                yield {"t": "token", "text": piece}
            if d.get("done"):
                n = d.get("eval_count", 0)
                dur = d.get("eval_duration", 0) / 1e9
                yield {"t": "stats", "eval_count": n, "prompt_eval_count": d.get("prompt_eval_count", 0),
                       "load_s": round(d.get("load_duration", 0) / 1e9, 2), "eval_s": round(dur, 2),
                       "total_s": round(d.get("total_duration", 0) / 1e9, 2),
                       "tps": round(n / dur, 1) if dur > 0 else 0.0,
                       "done_reason": d.get("done_reason", "stop")}
                break


# ------------------------------------------------------------ Hugging Face
@lru_cache(maxsize=1)
def _load_hf(model: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
    return tok, mdl


def hf_chat(model: str, messages: List[dict], num_predict: int = 700, temperature: float = 0.2) -> Iterator[dict]:
    from transformers import TextIteratorStreamer

    t0 = time.time()
    tok, mdl = _load_hf(model)
    load_s = time.time() - t0
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(mdl.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    Thread(target=mdl.generate, daemon=True,
           kwargs=dict(**inputs, streamer=streamer, max_new_tokens=num_predict, do_sample=True, temperature=temperature)).start()
    t1, n = time.time(), 0
    for chunk in streamer:
        n += 1
        yield {"t": "token", "text": chunk}
    dur = max(time.time() - t1, 1e-6)
    yield {"t": "stats", "eval_count": n, "prompt_eval_count": int(inputs["input_ids"].shape[1]),
           "load_s": round(load_s, 2), "eval_s": round(dur, 2), "total_s": round(time.time() - t0, 2),
           "tps": round(n / dur, 1), "done_reason": "stop"}


def generate(engine: str, model: str, messages: List[dict], num_predict: int = 900, temperature: float = 0.2) -> Iterator[dict]:
    fn = ollama_chat if engine == "ollama" else hf_chat
    return fn(model, messages, num_predict, temperature)
