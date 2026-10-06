"""FastAPI backend.

GET  /health, /models          POST /models/pull (install a model, streamed)
POST /preview                  live GitHub metadata for the pasted URL
POST /upload                   upload a zip / tar / loose files instead of a link -> source_id
POST /explain/stream           clone -> analyse -> local LLM (fast or file-by-file), streamed NDJSON
POST /chat/stream              follow-up questions about the repo
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import analyzer
import llm
import repo_processor as rp
from typing import List

from schemas import ChatRequest, ExplainRequest, PullRequest, RepoEntry, RepoPreview, RepoRequest, parse_repo

app = FastAPI(title="CodeAtlas · Local AI repository explainer", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GH_API = "https://api.github.com"


def _gh_headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "codeatlas"}
    if os.getenv("GITHUB_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return h


def _ev(**kw) -> str:
    return json.dumps(kw) + "\n"


def _public(facts: dict) -> dict:
    return {k: v for k, v in facts.items() if not k.startswith("_")}


def _stats_event(chunk: dict, scope: str) -> str:
    return _ev(type="stats", scope=scope, **{k: v for k, v in chunk.items() if k != "t"})


# ------------------------------------------------------------------ meta
@app.get("/health")
def health():
    models = llm.list_ollama_models()
    return {"status": "ok", "ollama_running": bool(models) or llm.ollama_up(), "ollama_models": models}


@app.get("/models")
def models():
    installed = llm.list_ollama_models()
    names = set(installed) | {n[:-7] for n in installed if n.endswith(":latest")}
    catalog = [{**m, "installed": m["id"] in names} for m in llm.CATALOG]
    known = {m["id"] for m in llm.CATALOG}
    for n in installed:
        base = n[:-7] if n.endswith(":latest") else n
        if base not in known:
            catalog.append({"id": base, "name": base, "vendor": "Local", "icon": "🧩", "size": "",
                            "blurb": "Installed on this machine.", "installed": True})
    return {"catalog": catalog, "installed": installed, "default": llm.pick_default(installed),
            "ollama_running": bool(installed) or llm.ollama_up(), "hf": [llm.DEFAULT_HF_MODEL]}


@app.post("/models/pull")
def pull(req: PullRequest):
    def gen() -> Iterator[str]:
        try:
            with httpx.stream("POST", f"{llm.OLLAMA_URL}/api/pull", json={"model": req.model, "stream": True},
                              timeout=httpx.Timeout(10.0, read=1800.0)) as r:
                if r.status_code != 200:
                    r.read()
                    yield _ev(type="error", text=f"Could not pull '{req.model}': {r.text[:200]}")
                    return
                for line in r.iter_lines():
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("error"):
                        yield _ev(type="error", text=d["error"])
                        return
                    yield _ev(type="pull", status=d.get("status", ""), completed=d.get("completed", 0), total=d.get("total", 0))
            yield _ev(type="done")
        except httpx.ConnectError:
            yield _ev(type="error", text="Cannot reach Ollama. Start it with:  ollama serve")
        except Exception as e:  # noqa: BLE001
            yield _ev(type="error", text=f"{type(e).__name__}: {e}")

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# --------------------------------------------------------------- preview
def _basic_preview(owner: str, repo: str, note: str) -> RepoPreview:
    return RepoPreview(owner=owner, name=repo, avatar=f"https://github.com/{owner}.png?size=120",
                       html_url=f"https://github.com/{owner}/{repo}", note=note)


@app.post("/preview", response_model=RepoPreview)
def preview(req: RepoRequest):
    owner, repo = parse_repo(req.url)
    try:
        with httpx.Client(headers=_gh_headers(), timeout=10) as c:
            r = c.get(f"{GH_API}/repos/{owner}/{repo}")
            if r.status_code == 404:
                raise HTTPException(404, "Repository not found (or it is private).")
            if r.status_code in (403, 429):
                return _basic_preview(owner, repo, "GitHub API rate limit reached, so this is a basic preview. "
                                                   "Set a GITHUB_TOKEN environment variable (or wait a few minutes) for full details. "
                                                   "Explaining the repo still works.")
            r.raise_for_status()
            d = r.json()

            langs = {}
            lr = c.get(f"{GH_API}/repos/{owner}/{repo}/languages")
            if lr.status_code == 200:
                raw = lr.json()
                total = sum(raw.values()) or 1
                langs = {k: round(v * 100 / total, 1) for k, v in raw.items()}

            root = []
            cr = c.get(f"{GH_API}/repos/{owner}/{repo}/contents")
            if cr.status_code == 200 and isinstance(cr.json(), list):
                items = sorted(cr.json(), key=lambda x: (x["type"] != "dir", x["name"].lower()))
                root = [RepoEntry(name=i["name"], type="dir" if i["type"] == "dir" else "file") for i in items[:40]]
    except httpx.HTTPError:
        return _basic_preview(owner, repo, "Could not reach the GitHub API, so this is a basic preview.")

    return RepoPreview(
        owner=d["owner"]["login"], name=d["name"], avatar=d["owner"]["avatar_url"],
        description=d.get("description"), html_url=d["html_url"], stars=d.get("stargazers_count", 0),
        forks=d.get("forks_count", 0), open_issues=d.get("open_issues_count", 0), language=d.get("language"),
        license=(d.get("license") or {}).get("spdx_id"), topics=d.get("topics", []),
        updated_at=d.get("pushed_at"), default_branch=d.get("default_branch", "main"),
        languages=langs, root=root,
    )


# ---------------------------------------------------------------- upload
MAX_UPLOAD_BYTES = 300 * 1024 * 1024


def _upload_summary(sid: str, data: dict) -> dict:
    return {"source_id": sid, "name": data["name"], "stats": data["stats"], "facts": _public(data["facts"]),
            "tree": data["tree"][:60]}


@app.post("/upload")
def upload(files: List[UploadFile] = File(...)):
    """Upload a repository as .zip / .tar.gz (or loose code files). Returns a source_id used by /explain/stream."""
    blobs, total = [], 0
    for f in files:
        b = f.file.read()
        total += len(b)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Upload too large (limit 300 MB). Remove big data/media folders and try again.")
        blobs.append((f.filename or "file", b))
    if not blobs:
        raise HTTPException(400, "No files received.")
    try:
        sid, data, _ = rp.extract_upload(blobs)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return _upload_summary(sid, data)


def _resolve(req):
    """-> (display name, extraction data, was cached). Works for GitHub links and uploads."""
    if req.source_id:
        data = rp.get_source(req.source_id)
        return data["name"], data, True
    data, cached = rp.get_extraction(req.url, getattr(req, "refresh", False))
    return data["name"], data, cached


# --------------------------------------------------------------- explain
@app.post("/explain/stream")
def explain_stream(req: ExplainRequest):
    """NDJSON events: status, log, meta, model, file_start/file_token/file_done, prompt, token, stats, verify, done, error."""

    def gen() -> Iterator[str]:
        t0 = time.time()

        def log(text: str) -> str:
            return _ev(type="log", t=round(time.time() - t0, 1), text=text)

        try:
            model = req.model or (llm.DEFAULT_OLLAMA_MODEL if req.engine == "ollama" else llm.DEFAULT_HF_MODEL)
            yield _ev(type="model", model=model, engine=req.engine, mode=req.mode, level=req.level)

            uploaded = bool(req.source_id)
            yield _ev(type="status", stage=0, text="Opening uploaded repository…" if uploaded else "Cloning repository…")
            if not uploaded:
                yield log(f"git clone --depth 1 {req.url}")
            name, data, cached = _resolve(req)
            if uploaded:
                yield log(f"Using the uploaded repository “{name}” (already unpacked and analysed)")
            elif cached:
                yield log("Using cached clone + analysis (identical input for every model)")
            else:
                yield log(f"Cloned in {data['clone_s']}s")

            yield _ev(type="status", stage=1, text="Identifying and extracting source code…")
            facts = data["facts"]
            yield log(f"Scanned {data['stats']['total_files']} files · static analysis took {data['analyze_s']}s")
            if facts["technologies"]:
                yield log("Detected stack: " + ", ".join(t["name"] for t in facts["technologies"][:8]))
            if facts["endpoints"]:
                yield log(f"Found {len(facts['endpoints'])} HTTP endpoint(s) via AST/regex")
            if facts["entry_points"]:
                yield log("Entry point(s): " + ", ".join(facts["entry_points"][:3]))
            if facts.get("project_type"):
                yield log(f"Project type (heuristic): {facts['project_type']}")
            if data["stats"]["total_files"] == 0:
                yield _ev(type="error", text="This repository has no files to explain.")
                return
            yield _ev(type="meta", files=[f["path"] for f in data["files"]], stats=data["stats"],
                      facts=_public(facts), cached=cached)

            facts_txt = analyzer.facts_text(facts)
            yield _ev(type="status", stage=2, text=f"Local LLM ({model}) is working…")
            yield log(f"Engine: {req.engine} · model: {model} · mode: {req.mode} · level: {req.level}")

            summaries = []
            if req.mode == "deep" and data["ranked"]:
                chosen = data["ranked"][:7]
                yield log(f"Deep mode: the model will read {len(chosen)} files one at a time")
                for i, f in enumerate(chosen, 1):
                    yield _ev(type="file_start", i=i, n=len(chosen), path=f["path"])
                    text, stats = "", None
                    for chunk in llm.generate(req.engine, model, llm.file_messages(f["path"], f["content"]),
                                              num_predict=110, temperature=0.1):
                        if chunk["t"] == "token":
                            text += chunk["text"]
                            yield _ev(type="file_token", text=chunk["text"])
                        else:
                            stats = chunk
                    text = text.strip()
                    summaries.append({"path": f["path"], "summary": text})
                    yield _ev(type="file_done", i=i, path=f["path"], summary=text,
                              stats={k: v for k, v in (stats or {}).items() if k != "t"})
                msgs = llm.explain_messages(name, data["tree"], facts_txt, req.level, summaries=summaries)
            else:
                msgs = llm.explain_messages(name, data["tree"], facts_txt, req.level, files=data["files"])
                if not data["files"]:
                    yield log("No readable text files – the model will explain the repo from its structure and metadata")

            prompt = msgs[-1]["content"]
            yield _ev(type="prompt", chars=len(prompt), tokens_est=len(prompt) // 4, text=prompt[:24000])
            yield log(f"Prompt built: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens) → sending to {model}")

            full = ""
            for chunk in llm.generate(req.engine, model, msgs, num_predict=1200, temperature=0.2):
                if chunk["t"] == "token":
                    full += chunk["text"]
                    yield _ev(type="token", text=chunk["text"])
                else:
                    yield _stats_event(chunk, "final")

            items = analyzer.verify_technologies(full, data["corpus"])
            if items:
                score = round(sum(1 for i in items if i["found"]) / len(items), 2)
                yield _ev(type="verify", items=items, score=score)
            yield _ev(type="done", elapsed=round(time.time() - t0, 1))
        except httpx.ConnectError:
            yield _ev(type="error", text="Cannot reach Ollama. Start it with:  ollama serve")
        except Exception as e:  # noqa: BLE001
            yield _ev(type="error", text=rp.friendly_error(e))

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ------------------------------------------------------------------ chat
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def gen() -> Iterator[str]:
        try:
            model = req.model or (llm.DEFAULT_OLLAMA_MODEL if req.engine == "ollama" else llm.DEFAULT_HF_MODEL)
            name, data, _ = _resolve(req)
            msgs = llm.chat_messages(name, data["tree"], analyzer.facts_text(data["facts"]),
                                     data["ranked"], [m.model_dump() for m in req.history], req.question)
            for chunk in llm.generate(req.engine, model, msgs, num_predict=500, temperature=0.2):
                if chunk["t"] == "token":
                    yield _ev(type="token", text=chunk["text"])
                else:
                    yield _stats_event(chunk, "chat")
            yield _ev(type="done")
        except httpx.ConnectError:
            yield _ev(type="error", text="Cannot reach Ollama. Start it with:  ollama serve")
        except Exception as e:  # noqa: BLE001
            yield _ev(type="error", text=rp.friendly_error(e))

    return StreamingResponse(gen(), media_type="application/x-ndjson")
