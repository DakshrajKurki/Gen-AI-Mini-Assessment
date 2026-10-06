"""CodeAtlas frontend (Streamlit): live model console, compare mode, architecture map, grounding check, chat."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

import runner
import ui

st.set_page_config(page_title=f"{ui.APP_NAME} · {ui.TAGLINE}", page_icon="🧭", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>", unsafe_allow_html=True)

DEFAULT_API = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
GH_RE = re.compile(r"^https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+?(?:\.git)?(?:/.*)?$")
EXAMPLES = [("🐍 pallets/click", "https://github.com/pallets/click"),
            ("🌐 expressjs/express", "https://github.com/expressjs/express"),
            ("📊 streamlit-example", "https://github.com/streamlit/streamlit-example")]
LEVELS = {"Like I'm 12": "eli5", "Beginner": "beginner", "Technical": "technical"}

# ------------------------------------------------------------------ helpers
@st.cache_data(ttl=4, show_spinner=False)
def get_json(api_url: str, path: str):
    try:
        return requests.get(f"{api_url}{path}", timeout=3).json()
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_preview(api_url: str, repo_url: str) -> dict:
    try:
        r = requests.post(f"{api_url}/preview", json={"url": repo_url}, timeout=15)
        if r.status_code == 200:
            return r.json()
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        if isinstance(detail, list):
            detail = detail[0].get("msg", "Invalid request")
        return {"error": str(detail)}
    except requests.exceptions.ConnectionError:
        return {"error": "Backend is not running. Start it with ./run.sh"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def do_upload(api_url: str, sig: str, _blobs: list) -> dict:
    try:
        files = [("files", (n, b)) for n, b in _blobs]
        r = requests.post(f"{api_url}/upload", files=files, timeout=300)
        if r.status_code == 200:
            return r.json()
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        if isinstance(detail, list):
            detail = detail[0].get("msg", "Invalid upload")
        return {"error": str(detail)}
    except requests.exceptions.ConnectionError:
        return {"error": "Backend is not running. Start it with ./run.sh"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def pull_model(api_url: str, model: str) -> bool:
    bar, txt = st.sidebar.progress(0.0), st.sidebar.empty()
    try:
        for ev in runner.stream_events(api_url, "/models/pull", {"model": model}):
            if ev["type"] == "pull":
                if ev.get("total"):
                    bar.progress(min(ev["completed"] / ev["total"], 1.0))
                txt.caption(f"{ev['status']}  {ev['completed'] / 1e6:,.0f} / {ev['total'] / 1e6:,.0f} MB" if ev.get("total") else ev["status"])
            elif ev["type"] == "error":
                st.sidebar.error(ev["text"])
                return False
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(str(e))
        return False
    get_json.clear()
    return True


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(ui.brand_html(), unsafe_allow_html=True)
    palette = st.selectbox("🎨 Theme", list(ui.PALETTES), help="Pick a colour palette.")
    st.markdown("### ⚙️ Settings")
    api = st.text_input("Backend URL", DEFAULT_API)
    engine_label = st.radio("Local LLM engine", ["Ollama", "Hugging Face (local)"])
    engine = "ollama" if engine_label == "Ollama" else "hf"

    health = get_json(api, "/health")
    info = get_json(api, "/models") or {}
    catalog = info.get("catalog", [])
    by_id = {m["id"]: m for m in catalog}

    if health is None:
        st.error("Backend offline — run `./run.sh`")
    elif engine == "ollama" and not health.get("ollama_running"):
        st.warning("Backend online · Ollama not running (`ollama serve`)")
    elif engine == "ollama":
        st.success("Backend + Ollama online")

    model, model2, compare = None, None, False
    if engine == "ollama" and catalog:
        ids = [m["id"] for m in catalog]

        def fmt(i: str) -> str:
            m = by_id[i]
            return f"{m['icon']} {m['name']}  {'✅' if m['installed'] else '⬇ not installed'}"

        default = info.get("default")
        model = st.selectbox("Model", ids, index=ids.index(default) if default in ids else 0, format_func=fmt)
        st.markdown(ui.model_card(by_id[model]), unsafe_allow_html=True)
        if not by_id[model]["installed"] and st.button(f"⬇ Install {by_id[model]['name']} ({by_id[model]['size']})", key="pull_main"):
            if pull_model(api, model):
                st.rerun()

        compare = st.toggle("⚔️ Compare two models", help="Runs the same repo through two models, one after the other, and compares speed + accuracy.")
        if compare:
            others = [i for i in ids if i != model]
            model2 = st.selectbox("Second model", others, index=others.index("qwen2.5:3b") if "qwen2.5:3b" in others else 0,
                                  format_func=fmt, key="model2")
            if not by_id[model2]["installed"] and st.button(f"⬇ Install {by_id[model2]['name']} ({by_id[model2]['size']})", key="pull_2"):
                if pull_model(api, model2):
                    st.rerun()
    elif engine == "hf":
        model = st.text_input("HF model id", (info.get("hf") or ["Qwen/Qwen2.5-0.5B-Instruct"])[0])
        st.caption("Needs `pip install torch transformers accelerate`. First run downloads the model.")
    elif engine == "ollama":
        st.info("Waiting for the backend to list models…")

    st.divider()
    mode_label = st.radio("Analysis mode", ["⚡ Fast (single pass)", "🔬 Deep (file-by-file)"],
                          help="Deep mode makes the model read the most important files one at a time, then combine its notes. Slower, usually more accurate.")
    mode = "fast" if mode_label.startswith("⚡") else "deep"
    level = LEVELS[st.select_slider("Explanation level", list(LEVELS), value="Beginner")]
    refresh = st.checkbox("Re-clone repo (ignore cache)")

# --------------------------------------------------------------------- hero
st.markdown(f"<style>{ui.theme_css(palette)}</style>", unsafe_allow_html=True)
st.markdown(ui.hero_html(), unsafe_allow_html=True)
pipe_ph = st.empty()

# -------------------------------------------------------------------- input
def _set_url(u: str) -> None:
    st.session_state["url_input"] = u


src_mode = st.segmented_control("Source", ["🔗 GitHub link", "📦 Upload a repo"], default="🔗 GitHub link",
                                key="src_mode", label_visibility="collapsed") or "🔗 GitHub link"
is_upload = src_mode.startswith("📦")
FIRST = "Uploaded Repo" if is_upload else "GitHub Repo"
pipe_ph.markdown(ui.pipeline_html(first=FIRST), unsafe_allow_html=True)

source, src_name, src_ok = None, "", False
if not is_upload:
    url = st.text_input("GitHub repository URL", key="url_input", placeholder="https://github.com/username/repository",
                        help="Paste the link and press Enter — the live preview appears instantly.").strip()
    ex_cols = st.columns(len(EXAMPLES))
    for c, (label, u) in zip(ex_cols, EXAMPLES):
        c.button(label, key=f"ex_{label}", on_click=_set_url, args=(u,), use_container_width=True)
    valid = bool(GH_RE.match(url)) if url else False
    if url and not valid:
        st.warning("That doesn't look like a GitHub repository URL.")
    if valid:
        with st.spinner("Fetching live preview…"):
            preview = fetch_preview(api, url)
        if "error" in preview:
            st.error(preview["error"])
        else:
            st.markdown(ui.preview_card(preview), unsafe_allow_html=True)
            st.write("")
        source = {"key": url, "payload": {"url": url}}
        src_name, src_ok = url.rstrip("/").split("/")[-1].removesuffix(".git"), True
else:
    ups = st.file_uploader("Upload your project", accept_multiple_files=True, key="uploader",
                           help="Any kind of repository: code, notebooks, docs, data, configs. Zip or tar.gz your folder (best), or drop loose files.")
    st.caption("🔒 Zip your project folder and drop it here. It is analysed on your own computer — nothing is sent to any cloud.")
    if ups:
        blobs = [(u.name, u.getvalue()) for u in ups]
        sig = "|".join(f"{n}:{len(b)}" for n, b in blobs)
        with st.spinner("Unpacking and scanning your project…"):
            up = do_upload(api, sig, blobs)
        if "error" in up:
            st.error(up["error"])
        else:
            st.markdown(ui.upload_card(up), unsafe_allow_html=True)
            st.write("")
            source = {"key": up["source_id"], "payload": {"source_id": up["source_id"]}}
            src_name, src_ok = up["name"], True

needs = [m for m in ([model] + ([model2] if compare else [])) if m]
missing = [m for m in needs if engine == "ollama" and by_id.get(m) and not by_id[m]["installed"]]
if missing:
    st.warning("Install these models from the sidebar first: " + ", ".join(missing))
what = "this uploaded repository" if is_upload else "this GitHub repository"
go = st.button(f"⚔️ Compare models on {what}" if compare else f"✨ Explain {what}",
               disabled=not src_ok or bool(missing) or health is None or not model, use_container_width=True)
skey = source["key"] if source else None


# ---------------------------------------------------------------- run view
def make_slots(console_parent, text_parent, idx: int) -> dict:
    with console_parent:
        console = st.empty()
        files = st.empty()
    with text_parent:
        box = st.container(key=f"result_{idx}")
        with box:
            text = st.empty()
    return {"console": console, "files": files, "text": text}


def layout(n: int):
    if n == 1:
        left, right = st.columns([1, 1.5], gap="large")
        return [(left, right)]
    cols = st.columns(n, gap="medium")
    return [(c, c) for c in cols]


def draw(slots: dict, s: dict) -> None:
    slots["console"].markdown(ui.console_html(s), unsafe_allow_html=True)
    slots["files"].markdown(ui.files_html(s), unsafe_allow_html=True)
    if s["phase"] == "error":
        slots["text"].error(s["error"])
    elif s["phase"] == "done":
        slots["text"].markdown(s["clean"] or s["text"])
    elif s["text"]:
        slots["text"].markdown(s["text"] + " ▌")
    else:
        slots["text"].markdown("_The explanation will stream here as soon as the model starts writing…_")


def model_meta(mid: str):
    m = by_id.get(mid) or {"name": mid, "icon": "🧩", "vendor": "Local"}
    return m["name"], m["icon"], m["vendor"]


def execute(slots: dict, s: dict, payload: dict) -> None:
    last, last_stage = 0.0, None
    draw(slots, s)
    try:
        for ev in runner.stream_events(api, "/explain/stream", payload):
            runner.apply_event(s, ev)
            now = time.time()
            stage = runner.stage_of(s)
            if stage != last_stage:
                pipe_ph.markdown(ui.pipeline_html(*stage, first=FIRST), unsafe_allow_html=True)
                last_stage = stage
            if ev["type"] not in ("token", "file_token") or now - last > 0.12:
                last = now
                draw(slots, s)
            if s["phase"] == "error":
                break
    except requests.exceptions.ConnectionError:
        runner.apply_event(s, {"type": "error", "text": "Cannot reach the backend. Start it with ./run.sh"})
    except Exception as e:  # noqa: BLE001
        runner.apply_event(s, {"type": "error", "text": str(e)})
    if s["phase"] == "error" and "no longer in memory" in (s["error"] or ""):
        do_upload.clear()
    if s["phase"] not in ("done", "error"):
        s["clean"], s["phase"] = runner.clean_markdown(s["text"]), "done"
    s["elapsed"] = s["total_s"] or (time.time() - s["t0"])
    draw(slots, s)


def comparison_table(runs: list) -> None:
    rows = []
    for s in runs:
        fs = s["final_stats"] or {}
        rows.append({
            "Model": f"{s['icon']} {s['model_name']}",
            "Total time": f"{s['elapsed']:.1f}s",
            "Speed (tok/s)": f"{fs['tps']:.1f}" if fs.get("tps") else "—",
            "Tokens written": str(fs.get("eval_count", s["tokens"])),
            "Model load": f"{fs.get('load_s', '—')}s",
            "Grounding": f"{int(s['verify']['score'] * 100)}% verified" if s["verify"] else "—",
        })
    st.markdown("#### ⚔️ Side-by-side results")
    st.table(rows)


runs_url_ok = bool(skey) and st.session_state.get("runs") and st.session_state.get("runs_url") == skey
if go:
    models = [model] + ([model2] if compare else [])
    runs = []
    for mid in models:
        n, icon, vendor = model_meta(mid)
        runs.append(runner.new_state(mid, n, icon, vendor, mode, level, engine))
    slot_list = [make_slots(c, t, i) for i, (c, t) in enumerate(layout(len(models)))]
    for slots, s in zip(slot_list, runs):
        draw(slots, s)
    for slots, s in zip(slot_list, runs):
        s["t0"] = time.time()
        execute(slots, s, {**source["payload"], "engine": engine, "model": s["model"], "mode": mode, "level": level,
                           "refresh": refresh and s is runs[0]})
    st.session_state["runs"], st.session_state["runs_url"], st.session_state["runs_name"] = runs, skey, src_name
    pipe_ph.markdown(ui.pipeline_html(done_upto=5, first=FIRST), unsafe_allow_html=True)
    runs_url_ok = True
elif runs_url_ok:
    runs = st.session_state["runs"]
    slot_list = [make_slots(c, t, i) for i, (c, t) in enumerate(layout(len(runs)))]
    for slots, s in zip(slot_list, runs):
        draw(slots, s)
    pipe_ph.markdown(ui.pipeline_html(done_upto=5, first=FIRST), unsafe_allow_html=True)

# ------------------------------------------------------------ after the run
if runs_url_ok:
    runs = st.session_state["runs"]
    ok_runs = [s for s in runs if s["phase"] == "done" and s["text"]]
    if len(runs) > 1:
        comparison_table(runs)
    if ok_runs:
        d1, d2 = st.columns(len(ok_runs)) if len(ok_runs) > 1 else (st.container(), None)
        for col, s in zip([d1, d2] if d2 else [d1], ok_runs):
            with col:
                st.download_button(f"⬇️ Download ({s['model_name']})", s["clean"], key=f"dl_{s['model']}",
                                   file_name=f"codeatlas_{src_name or 'repo'}_{s['model'].replace(':', '_')}.md", use_container_width=True)

    base = next((s for s in runs if s["facts"]), None)
    if base:
        facts = base["facts"]
        st.markdown(ui.repo_header(src_name or "repository", facts, base["stats"], "📦" if is_upload else "🐙"), unsafe_allow_html=True)
        t_graph, t_facts, t_work, t_chat = st.tabs(["🗺️ Architecture map", "🔍 Verified facts", "🧠 Model working", "💬 Ask the repo"])

        with t_graph:
            if facts.get("graph_dot"):
                st.caption(f"Built by static analysis (not by the LLM): {facts['graph_nodes']} files, {facts['graph_edges']} import links. "
                           "▶ = entry point · dashed purple = external technology.")
                st.graphviz_chart(facts["graph_dot"], use_container_width=True)
            else:
                st.info("Not enough local imports to draw a map (single-file or non-Python/JS project).")

        with t_facts:
            st.caption("Everything here was found by parsing the repo (ast, manifests, regex) — it is given to the model as ground truth.")
            st.markdown(f"**Project type:** {facts['project_type']}")
            if facts.get("inventory"):
                st.markdown("**What the repository contains**")
                st.markdown(ui.inventory_html(facts["inventory"]), unsafe_allow_html=True)
            if facts.get("structure"):
                st.markdown("**Top-level layout**")
                st.markdown(ui.structure_html(facts["structure"]), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Files in repo", base["stats"].get("total_files", "-"))
            c2.metric("Files the model read", base["stats"].get("files_used", "-"))
            c3.metric("Characters sent", f"{base['stats'].get('chars_sent', 0):,}")
            if facts["technologies"]:
                st.markdown("**Detected stack**")
                st.markdown(ui.tech_html(facts["technologies"]), unsafe_allow_html=True)
            if facts["languages"]:
                st.markdown("**Code size by language (non-blank lines)**")
                st.bar_chart(pd.DataFrame({"lines": facts["languages"]}), color="#22d3ee", height=220)
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("**Entry points**")
                st.code("\n".join(facts["entry_points"]) or "none detected", language=None)
            with e2:
                st.markdown("**HTTP endpoints**")
                st.code("\n".join(f"{e['method']:6} {e['path']}   ({e['file']})" for e in facts["endpoints"]) or "none detected", language=None)
            if facts["readme"]["summary"]:
                st.markdown("**README says**")
                st.info(facts["readme"]["summary"])
            for s in runs:
                if s["verify"]:
                    st.markdown(f"**Grounding check — {s['icon']} {s['model_name']}** · "
                                f"{int(s['verify']['score'] * 100)}% of the technologies it listed were found in the code")
                    st.markdown(ui.verify_html(s["verify"]), unsafe_allow_html=True)
                    st.caption("✅ found in dependencies / imports / file types · ⚠️ not found – possibly invented (heuristic check)")

        with t_work:
            for s in runs:
                st.markdown(f"##### {s['icon']} {s['model_name']} — how it worked")
                fs = s["final_stats"] or {}
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("Total time", f"{s['elapsed']:.1f}s")
                w2.metric("Speed", f"{fs.get('tps', '—')} tok/s")
                w3.metric("Prompt tokens", f"{fs.get('prompt_eval_count', '—')}")
                w4.metric("Model load", f"{fs.get('load_s', '—')}s")
                st.code("\n".join(f"[{t:5.1f}s] {x}" for t, x in s["log"]) or "no log", language=None)
                if s["file_summaries"]:
                    st.markdown(ui.files_html({**s, "file_summaries": s["file_summaries"]}), unsafe_allow_html=True)
                if s["prompt"]:
                    with st.expander("🔎 The exact prompt sent to the model"):
                        st.code(s["prompt"], language=None)
                with st.expander("Files that were read"):
                    st.code("\n".join(s["files"]) or "—", language=None)
                st.divider()

        with t_chat:
            st.caption("Ask follow-up questions. The local model answers using only the extracted code and verified facts.")
            hist = st.session_state.setdefault("chat", {}).setdefault(skey, [])
            sugg = ["How do I run this project?", "Where is the main entry point?", "What does each folder do?", "How would I add a new feature?"]
            for col, q_ in zip(st.columns(len(sugg)), sugg):
                col.button(q_, key=f"sg_{q_}", use_container_width=True,
                           on_click=lambda q=q_: st.session_state.__setitem__("pending_q", q))
            q = st.chat_input("Ask anything about this repository…") or st.session_state.pop("pending_q", None)
            for m in hist:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            if q:
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    ph, ans, tps = st.empty(), "", None
                    try:
                        payload = {**source["payload"], "engine": engine, "model": model, "question": q, "history": hist[-6:]}
                        for ev in runner.stream_events(api, "/chat/stream", payload):
                            if ev["type"] == "token":
                                ans += ev["text"]
                                ph.markdown(ans + " ▌")
                            elif ev["type"] == "stats":
                                tps = ev.get("tps")
                            elif ev["type"] == "error":
                                ans = f"⚠️ {ev['text']}"
                                break
                    except Exception as e:  # noqa: BLE001
                        ans = f"⚠️ {e}"
                    ph.markdown(ans)
                    if tps:
                        st.caption(f"{model} · {tps} tok/s")
                hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]

st.markdown('<div class="foot">🧭 CodeAtlas · GitHub / Uploaded Repo → Code Processing → Local LLM → Backend → Frontend → Explanation</div>', unsafe_allow_html=True)
