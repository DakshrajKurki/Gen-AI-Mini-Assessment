"""Run state + event handling for the live "model working" view (no Streamlit imports)."""
from __future__ import annotations

import json
import re
import time
from collections import deque
from typing import Dict, Iterator

import requests


def new_state(model: str, model_name: str, icon: str, vendor: str, mode: str, level: str, engine: str) -> dict:
    return {
        "model": model, "model_name": model_name, "icon": icon, "vendor": vendor,
        "mode": mode, "level": level, "engine": engine,
        "phase": "queued", "t0": time.time(), "elapsed": 0.0,
        "tokens": 0, "tps": 0.0, "seg_start": None, "seg_tokens": 0,
        "prompt_tokens": 0, "prompt": "", "recent": deque(maxlen=14),
        "files_total": 0, "files_done": 0, "current_file": None, "file_summaries": [],
        "log": [], "text": "", "clean": "", "facts": None, "files": [], "stats": {}, "cached": False,
        "final_stats": None, "verify": None, "error": None, "total_s": None,
    }


def stream_events(api: str, path: str, payload: dict) -> Iterator[dict]:
    with requests.post(f"{api}{path}", json=payload, stream=True, timeout=(10, 900)) as r:
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            if isinstance(detail, list):
                detail = detail[0].get("msg", "Invalid request")
            yield {"type": "error", "text": f"Backend error {r.status_code}: {str(detail)[:300]}"}
            return
        for line in r.iter_lines(decode_unicode=True):
            if line:
                yield json.loads(line)


def _tick(s: dict, text: str) -> None:
    s["tokens"] += 1
    s["seg_tokens"] += 1
    s["recent"].append(text)
    now = time.time()
    if s["seg_start"] is None:
        s["seg_start"] = now
    dt = now - s["seg_start"]
    if dt > 0.4:
        s["tps"] = s["seg_tokens"] / dt


def _log(s: dict, text: str) -> None:
    s["log"].append((round(time.time() - s["t0"], 1), text))


def apply_event(s: dict, ev: dict) -> None:
    t = ev["type"]
    s["elapsed"] = time.time() - s["t0"]
    if t == "status":
        s["phase"] = ["clone", "extract", "loading"][ev["stage"]]
    elif t == "log":
        s["log"].append((ev["t"], ev["text"]))
    elif t == "meta":
        s["files"], s["stats"], s["facts"], s["cached"] = ev["files"], ev["stats"], ev["facts"], ev.get("cached", False)
    elif t == "file_start":
        s["phase"] = "reading"
        s["current_file"] = ev["path"]
        s["files_total"] = ev["n"]
        s["file_summaries"].append({"path": ev["path"], "summary": "", "done": False, "tokens": 0, "seconds": None})
        s["seg_start"], s["seg_tokens"], s["tps"] = None, 0, 0.0
        _log(s, f"Reading file {ev['i']}/{ev['n']}: {ev['path']}")
    elif t == "file_token":
        s["phase"] = "generating"
        s["file_summaries"][-1]["summary"] += ev["text"]
        s["file_summaries"][-1]["tokens"] += 1
        _tick(s, ev["text"])
    elif t == "file_done":
        fs = s["file_summaries"][-1]
        fs.update(done=True, summary=ev["summary"], tokens=ev["stats"].get("eval_count", fs["tokens"]),
                  seconds=ev["stats"].get("total_s"))
        s["files_done"] += 1
        s["current_file"] = None
        _log(s, f"  ↳ {ev['path']}: {fs['tokens']} tok @ {ev['stats'].get('tps', 0)} tok/s")
    elif t == "prompt":
        s["phase"] = "reading"
        s["prompt_tokens"], s["prompt"] = ev["tokens_est"], ev["text"]
        s["seg_start"], s["seg_tokens"], s["tps"] = None, 0, 0.0
    elif t == "token":
        if s["phase"] != "generating" or s["current_file"] is not None or not s["text"]:
            if not s["text"]:
                _log(s, "First token received – model is writing the explanation")
        s["phase"] = "generating"
        s["current_file"] = None
        s["text"] += ev["text"]
        _tick(s, ev["text"])
    elif t == "stats":
        if ev.get("scope") == "final":
            s["final_stats"] = ev
            s["tps"] = ev.get("tps", s["tps"])
            _log(s, f"Done: {ev['eval_count']} tokens · {ev['tps']} tok/s · model load {ev['load_s']}s · prompt {ev['prompt_eval_count']} tok")
    elif t == "verify":
        s["verify"] = ev
        _log(s, f"Grounding check: {int(ev['score'] * 100)}% of listed technologies found in the code")
    elif t == "done":
        s["phase"] = "done"
        s["total_s"] = ev.get("elapsed", s["elapsed"])
        s["elapsed"] = s["total_s"]
        s["clean"] = clean_markdown(s["text"])
    elif t == "error":
        s["phase"] = "error"
        s["error"] = ev["text"]
        _log(s, "ERROR: " + ev["text"])


def clean_markdown(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:markdown|md)?\s*\n", "", t)
    t = re.sub(r"\n```\s*$", "", t)
    i = t.find("## Project Overview")
    if i > 0:
        t = t[i:]
    return t


def stage_of(s: dict) -> tuple[int, int]:
    """(active node, last done node) for the pipeline strip."""
    ph = s["phase"]
    if ph == "done":
        return -1, 5
    if ph == "clone":
        return 0, -1
    if ph == "extract":
        return 1, 0
    if ph == "generating" and s["text"]:
        return 4, 3
    if ph in ("loading", "reading", "generating"):
        return 2, 1
    return -1, -1
