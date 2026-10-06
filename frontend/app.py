"""
frontend/app.py  -  RepoLens AI (Streamlit)
===========================================
Run from the project root:   python -m streamlit run frontend/app.py

IMPORTANT (Streamlit Cloud fix)
-------------------------------
Streamlit puts *this file's folder* (frontend/) on sys.path, NOT the project
root, so `import backend...` would fail with "No module named 'backend'".
We therefore add the project root to sys.path BEFORE importing backend modules.
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import html
import logging
import re

import streamlit as st

st.set_page_config(page_title="RepoLens AI", page_icon="🔍", layout="wide")

from backend.repository import (  # noqa: E402
    clone_repository,
    cleanup_repository,
    RepositoryError,
)

from backend.code_processor import (  # noqa: E402
    build_repository_context,
)

from backend.llm import (  # noqa: E402
    explain_repository,
    describe_files,
    describe_folders,
    LLMError,
)

logger = logging.getLogger(__name__)

EXAMPLES = {
    "psf/requests (Python)": "https://github.com/psf/requests",
    "pallets/flask (Python)": "https://github.com/pallets/flask",
    "expressjs/express (JavaScript)": "https://github.com/expressjs/express",
}

LANG_ICONS = {
    "Python": "🐍", "JavaScript": "🟨", "JavaScript (React)": "⚛️", "TypeScript": "🔷",
    "TypeScript (React)": "⚛️", "Java": "☕", "Go": "🐹", "Rust": "🦀", "HTML": "🌐",
    "CSS": "🎨", "SQL": "🗄️", "Markdown": "📝", "JSON": "🧾", "YAML": "⚙️", "TOML": "⚙️",
    "C": "🔧", "C++": "🔧", "C#": "🟣", "PHP": "🐘", "Docker": "🐳", "License": "⚖️",
}
SECTION_ICONS = [
    ("overview", "🧭"), ("feature", "⚡"), ("structure", "🗂️"), ("work", "⚙️"),
    ("technolog", "🧰"), ("component", "🧩"), ("summary", "📝"),
]
BAR_COLORS = ["#8b5cf6", "#06b6d4", "#f59e0b", "#10b981", "#ec4899", "#6366f1", "#84cc16", "#f97316"]

# ------------------------------------------------------------------ styling --
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg: #0b0f1e; --panel: rgba(255,255,255,.045); --panel-hover: rgba(255,255,255,.075);
  --border: rgba(255,255,255,.10); --accent: #8b5cf6; --accent2: #06b6d4;
  --text: #e6e9f5; --muted: #98a3c4; --ok: #34d399;
}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea {
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
}
.stApp {
  background:
    radial-gradient(900px 500px at 8% -5%, rgba(139,92,246,.22), transparent 60%),
    radial-gradient(800px 480px at 95% 0%, rgba(6,182,212,.16), transparent 60%),
    var(--bg);
}
[data-testid="stHeader"] {background: transparent;}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.4rem; max-width: 1180px;}
[data-testid="stSidebar"] {background: rgba(19,26,51,.85); border-right: 1px solid var(--border);}

/* Force the dark palette even if .streamlit/config.toml is not picked up */
.stApp, [data-testid="stSidebar"], [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] li, label, h1, h2, h3, h4 {color: var(--text) !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: var(--muted) !important;}
div[data-baseweb="input"] input, div[data-baseweb="base-input"] {background: transparent !important; color: var(--text) !important;}
div[data-baseweb="input"] input::placeholder {color: #7480a6 !important;}
div[data-baseweb="select"] > div {background: rgba(255,255,255,.06) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; color: var(--text) !important;}
div[data-baseweb="select"] * {color: var(--text) !important;}
[data-baseweb="popover"] li, [data-baseweb="menu"] {background: #161d3a !important; color: var(--text) !important;}
[data-testid="stSlider"] [role="slider"] {background: var(--accent) !important; box-shadow: 0 0 0 4px rgba(139,92,246,.3);}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] {color: var(--muted) !important;}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important;}
[data-testid="stSlider"] [data-testid="stThumbValue"] {color: #c4b5fd !important;}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > span:first-child {background-color: var(--accent) !important; border-color: var(--accent) !important;}
[data-testid="stTabs"] button[role="tab"] {color: var(--muted) !important;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color: #fff !important;}
[data-testid="stCode"], [data-testid="stCode"] pre {background: rgba(0,0,0,.35) !important; border-radius: 14px;}
[data-testid="stAlert"] {border-radius: 14px;}

@keyframes fadeUp {from {opacity: 0; transform: translateY(16px);} to {opacity: 1; transform: none;}}
@keyframes gradientShift {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
@keyframes floaty {0%,100% {transform: translateY(0) scale(1);} 50% {transform: translateY(-18px) scale(1.06);}}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(139,92,246,.65);} 100% {box-shadow: 0 0 0 14px rgba(139,92,246,0);}}
@keyframes grow {from {width: 0;} to {width: var(--w);}}
@keyframes shimmer {0% {background-position: -200% 0;} 100% {background-position: 200% 0;}}

/* ---------- hero ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 24px; padding: 2.3rem 2.4rem; margin-bottom: 1.5rem;
  background: linear-gradient(120deg, #4c1d95, #6d28d9, #0e7490, #0891b2, #4c1d95);
  background-size: 300% 300%; animation: gradientShift 14s ease infinite, fadeUp .7s ease both;
  border: 1px solid rgba(255,255,255,.18); box-shadow: 0 20px 60px rgba(76,29,149,.35);
}
.hero::before, .hero::after {content: ""; position: absolute; border-radius: 50%; filter: blur(8px); opacity: .5;}
.hero::before {width: 220px; height: 220px; right: -50px; top: -70px; background: radial-gradient(circle, #67e8f9, transparent 70%); animation: floaty 7s ease-in-out infinite;}
.hero::after {width: 180px; height: 180px; right: 22%; bottom: -90px; background: radial-gradient(circle, #c4b5fd, transparent 70%); animation: floaty 9s ease-in-out infinite reverse;}
.hero h1 {position: relative; z-index: 1; margin: 0; font-size: 2.7rem; font-weight: 800; letter-spacing: -.02em; color: #fff !important;}
.hero p {position: relative; z-index: 1; margin: .35rem 0 1rem; font-size: 1.1rem; color: rgba(255,255,255,.88) !important;}
.hero .badges {position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: .5rem;}
.hero .badge {font-size: .78rem; font-weight: 600; padding: .3rem .75rem; border-radius: 999px; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25); backdrop-filter: blur(6px); color: #fff;}

/* ---------- form / inputs ---------- */
[data-testid="stForm"] {
  background: var(--panel); border: 1px solid var(--border); border-radius: 20px;
  padding: 1.3rem 1.5rem; backdrop-filter: blur(10px); animation: fadeUp .7s .1s ease both;
}
div[data-baseweb="input"] {
  border-radius: 14px !important; background: rgba(255,255,255,.06) !important;
  border: 1px solid var(--border) !important; transition: box-shadow .25s, border-color .25s;
}
div[data-baseweb="input"]:focus-within {border-color: var(--accent) !important; box-shadow: 0 0 0 4px rgba(139,92,246,.28);}
div[data-baseweb="input"] input {padding: .8rem 1rem !important; font-size: 1rem;}
[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(120deg, #8b5cf6, #06b6d4); background-size: 200% 200%;
  color: #fff; border: none; font-weight: 700; padding: .65rem 1.8rem; border-radius: 14px;
  transition: transform .2s ease, box-shadow .25s ease, background-position .4s ease;
}
[data-testid="stFormSubmitButton"] button:hover {transform: translateY(-2px); box-shadow: 0 10px 28px rgba(139,92,246,.5); background-position: 100% 0; color: #fff;}
[data-testid="stFormSubmitButton"] button:active {transform: translateY(0);}
[data-testid="stSidebar"] .stButton button {
  border-radius: 12px; border: 1px solid var(--border); background: var(--panel);
  transition: transform .2s ease, background .2s ease, border-color .2s ease;
}
[data-testid="stSidebar"] .stButton button:hover {transform: translateX(4px); background: var(--panel-hover); border-color: var(--accent);}
[data-testid="stDownloadButton"] button {border-radius: 12px; border: 1px solid var(--border); transition: transform .2s, border-color .2s;}
[data-testid="stDownloadButton"] button:hover {transform: translateY(-2px); border-color: var(--accent2);}

/* ---------- progress ---------- */
[data-testid="stProgress"] [role="progressbar"] > div {background: linear-gradient(90deg, #8b5cf6, #06b6d4, #8b5cf6) !important; background-size: 200% 100%; animation: shimmer 2.2s linear infinite; transition: width .5s ease;}
.steps {display: flex; align-items: center; gap: .4rem; flex-wrap: wrap; margin: 1.1rem 0 .6rem; animation: fadeUp .4s ease both;}
.step {display: flex; align-items: center; gap: .5rem; font-size: .88rem; color: var(--muted); font-weight: 500; transition: color .3s;}
.step .dot {width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-size: .8rem; font-weight: 700; background: var(--panel); border: 1px solid var(--border); transition: all .3s;}
.step.done {color: var(--text);} .step.done .dot {background: var(--ok); border-color: var(--ok); color: #052e1b;}
.step.active {color: #fff;} .step.active .dot {background: var(--accent); border-color: var(--accent); color: #fff; animation: pulse 1.3s ease-out infinite;}
.steps .line {width: 28px; height: 2px; background: var(--border); border-radius: 2px;}

/* ---------- pipeline (empty state) ---------- */
.pipeline {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-top: 1.4rem;}
.pipe {background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 1.1rem 1.1rem 1.2rem; animation: fadeUp .6s ease both; transition: transform .25s ease, background .25s, border-color .25s;}
.pipe:hover {transform: translateY(-6px); background: var(--panel-hover); border-color: var(--accent);}
.pipe .ico {font-size: 1.7rem;} .pipe b {display: block; margin: .45rem 0 .15rem; color: var(--text);} .pipe span {font-size: .84rem; color: var(--muted); line-height: 1.4;}

/* ---------- result header + stats ---------- */
.repo-head {display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: .8rem; background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 1rem 1.3rem; margin: 1rem 0; animation: fadeUp .5s ease both;}
.repo-head a {color: #fff; font-weight: 700; font-size: 1.25rem; text-decoration: none; background: linear-gradient(90deg, #c4b5fd, #67e8f9); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;}
.pill {display: inline-block; font-size: .76rem; font-weight: 600; padding: .25rem .7rem; border-radius: 999px; background: rgba(139,92,246,.18); border: 1px solid rgba(139,92,246,.4); color: #ddd6fe; margin-left: .4rem;}
.pill.cyan {background: rgba(6,182,212,.15); border-color: rgba(6,182,212,.4); color: #a5f3fc;}
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 1.2rem;}
.stat {background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 1rem 1.2rem; animation: fadeUp .6s ease both; transition: transform .25s, border-color .25s, background .25s;}
.stat:hover {transform: translateY(-4px); border-color: var(--accent2); background: var(--panel-hover);}
.stat .k {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600;}
.stat .v {font-size: 2rem; font-weight: 800; line-height: 1.15; background: linear-gradient(90deg, #fff, #c4b5fd); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;}
.stat .s {font-size: .78rem; color: var(--muted);}

/* ---------- tabs + sections ---------- */
[data-testid="stTabs"] [role="tablist"] {gap: .4rem; border-bottom: 1px solid var(--border);}
[data-testid="stTabs"] button[role="tab"] {font-weight: 600; border-radius: 12px 12px 0 0; padding: .6rem 1rem; transition: background .2s, color .2s;}
[data-testid="stTabs"] button[role="tab"]:hover {background: var(--panel);}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background: linear-gradient(90deg, #8b5cf6, #06b6d4); height: 3px; border-radius: 3px;}
[data-baseweb="tab-panel"] {animation: fadeUp .5s ease both;}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel); border-color: var(--border) !important; border-radius: 18px !important;
  animation: fadeUp .55s ease both; transition: transform .25s ease, border-color .25s ease, background .25s;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {border-color: rgba(139,92,246,.55) !important; background: var(--panel-hover);}
.sec-title {display: flex; align-items: center; gap: .6rem; font-size: 1.15rem; font-weight: 700; margin-bottom: .4rem;}
.sec-title .ico {width: 38px; height: 38px; border-radius: 12px; display: grid; place-items: center; background: linear-gradient(135deg, rgba(139,92,246,.35), rgba(6,182,212,.3)); font-size: 1.1rem;}

/* ---------- cards: folders, files, bars ---------- */
.grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; margin: .6rem 0 1rem;}
.card {background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 1.1rem 1.2rem; animation: fadeUp .55s ease both; transition: transform .25s ease, border-color .25s ease, background .25s ease, box-shadow .25s ease;}
.card:hover {transform: translateY(-5px); border-color: var(--accent); background: var(--panel-hover); box-shadow: 0 14px 36px rgba(0,0,0,.35);}
.card h4 {margin: 0 0 .15rem; font-size: 1.05rem; font-weight: 700; color: #fff;}
.card .meta {font-size: .78rem; color: var(--muted); margin-bottom: .55rem;}
.card .desc {font-size: .92rem; line-height: 1.5; color: var(--text); margin: .4rem 0 .7rem;}
.card .desc.none {color: var(--muted); font-style: italic;}
.chip {display: inline-block; font-size: .74rem; padding: .18rem .6rem; margin: 0 .3rem .3rem 0; border-radius: 999px; background: rgba(255,255,255,.07); border: 1px solid var(--border); color: #cbd5f5;}
.chip.mono {font-family: 'JetBrains Mono', monospace; font-size: .72rem; background: rgba(6,182,212,.1); border-color: rgba(6,182,212,.3); color: #a5f3fc;}
.chip.lang {background: rgba(139,92,246,.16); border-color: rgba(139,92,246,.4); color: #ddd6fe;}
.fpath {font-family: 'JetBrains Mono', monospace; font-size: .86rem; color: #fff; word-break: break-all;}
.file-card {padding: .95rem 1.2rem;}
.bar-row {margin: .65rem 0; animation: fadeUp .5s ease both;}
.bar-row .lbl {display: flex; justify-content: space-between; font-size: .88rem; margin-bottom: .3rem;}
.bar {height: 10px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden;}
.bar .fill {height: 100%; border-radius: 99px; width: var(--w); animation: grow 1.1s cubic-bezier(.2,.8,.2,1) both;}
.note {font-size: .85rem; color: var(--muted); margin: .3rem 0 .8rem;}

@media (prefers-reduced-motion: reduce) {* {animation: none !important; transition: none !important;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers --
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def repo_label(url: str) -> str:
    """'https://github.com/psf/requests' -> 'psf/requests'."""
    parsed = urlparse(url if "://" in url else "https://" + url)
    label = parsed.path.strip("/")
    if label.endswith(".git"):
        label = label[:-4]
    return label or url


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split the LLM's Markdown answer into (heading, body) pairs.

    Small models do not always follow the format, so the caller falls back to
    showing the raw text when fewer than 3 sections are found.
    """
    heading = re.compile(r"^\s{0,3}#{1,4}\s*(.+?)\s*#*\s*$")
    sections: list[tuple[str, str]] = []
    title, buf, in_code = None, [], False

    def flush():
        body = "\n".join(buf).strip()
        if title is not None or body:
            sections.append((title or "Overview", body))

    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else heading.match(line)
        if match:
            flush()
            title, buf = match.group(1).strip("* ").strip(), []
        else:
            buf.append(line)
    flush()
    return [s for s in sections if s[1]]


def section_icon(title: str) -> str:
    lower = title.lower()
    return next((icon for key, icon in SECTION_ICONS if key in lower), "📌")


def render_steps(steps: list[str], current: int) -> str:
    """Animated progress stepper. Steps before `current` are done, `current` pulses."""
    parts = []
    for i, label in enumerate(steps):
        state = "done" if i < current else "active" if i == current else ""
        dot = "✓" if state == "done" else str(i + 1)
        parts.append(f'<div class="step {state}"><span class="dot">{dot}</span><span>{esc(label)}</span></div>')
        if i < len(steps) - 1:
            parts.append('<div class="line"></div>')
    return '<div class="steps">' + "".join(parts) + "</div>"


def lang_chip(lang: str) -> str:
    return f'<span class="chip lang">{LANG_ICONS.get(lang, "📄")} {esc(lang)}</span>'


def set_example(url: str) -> None:
    st.session_state["repo_url"] = url


def run_analysis(url: str, deep: bool, n_files: int) -> None:
    """Clone -> process -> LLM -> (deep scan) -> session_state. Always cleans up."""
    st.session_state.pop("error", None)
    st.session_state.pop("result", None)

    steps = ["Clone repository", "Analyze code", "Explain project"]
    if deep:
        steps += ["Read each file", "Summarize folders"]
    stepper = st.empty()
    progress = st.progress(0)

    def show(index: int, text: str, fraction: float = 0.0) -> None:
        stepper.markdown(render_steps(steps, index), unsafe_allow_html=True)
        progress.progress(min(99, int(100 * (index + fraction) / len(steps))), text=text)

    repo_path = None
    notice = None
    try:
        show(0, "Cloning repository from GitHub...")
        repo_path = clone_repository(url)

        show(1, "Analyzing repository structure and selecting files...")
        context = build_repository_context(repo_path)

        show(2, "Asking the AI model to explain the project...")
        with st.spinner("Writing the overview. A small model on CPU can take 30-90 seconds "
                        "(the first online run also downloads the model)..."):
            result = explain_repository(context)

        if deep:
            try:
                def on_file(done, total, label):
                    show(3, f"Reading file {min(done + 1, total)} of {total}: {label.replace('Reading ', '')}",
                         done / max(total, 1))

                def on_folder(done, total, label):
                    show(4, f"Summarizing {label.replace('Summarizing folder ', 'folder ')}",
                         done / max(total, 1))

                describe_files(context, limit=n_files, progress=on_file)
                describe_folders(context, progress=on_folder)
            except LLMError as exc:   # the main explanation is still valuable
                notice = f"The file guide is incomplete: {exc}"

        stepper.markdown(render_steps(steps, len(steps)), unsafe_allow_html=True)
        progress.progress(100, text="Done!")
        st.session_state["result"] = {
            "url": url,
            "label": repo_label(url),
            "context": context,
            "explanation": result.text,
            "engine": result.engine,
            "deep": deep,
            "notice": notice,
        }
    except (RepositoryError, LLMError) as exc:
        st.session_state["error"] = str(exc)
    except Exception:  # last line of defence: log details, show something friendly
        logger.exception("Unexpected error while analyzing %s", url)
        st.session_state["error"] = "Something unexpected went wrong. Please try again."
    finally:
        cleanup_repository(repo_path)   # the temporary clone is ALWAYS deleted
        stepper.empty()
        progress.empty()


def build_report(result: dict) -> str:
    """Everything on screen as one Markdown file (for the download button)."""
    context = result["context"]
    out = [f"# {result['label']}", f"*Explained by {result['engine']}*", "", result["explanation"], ""]
    if context.folders:
        out += ["## Folder guide", ""]
        for f in context.folders:
            out.append(f"- **{f.name}** ({f.file_count} files): {f.summary or 'no summary'}")
        out.append("")
    described = [i for i in context.inventory if i.summary]
    if described:
        out += ["## File guide", ""]
        for i in sorted(described, key=lambda x: x.path):
            out.append(f"- `{i.path}` ({i.language}): {i.summary}")
        out.append("")
    out += ["## Project structure", "", "```text", context.structure, "```", ""]
    return "\n".join(out)


# ------------------------------------------------------------------- hero ----
st.markdown(
    '<div class="hero"><h1>🔍 RepoLens AI</h1>'
    "<p>Local GitHub Repository Code Explainer. Paste a repo, get the whole project explained.</p>"
    '<div class="badges"><span class="badge">🧠 Qwen 2.5 · 0.5B</span>'
    '<span class="badge">🦙 Ollama (local)</span><span class="badge">🤗 Transformers (cloud)</span>'
    '<span class="badge">🌿 GitPython</span><span class="badge">⚡ FastAPI + Streamlit</span></div></div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Try an example")
    for name, example_url in EXAMPLES.items():
        st.button(name, on_click=set_example, args=(example_url,), use_container_width=True)
    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. Clone the repo (GitPython)\n"
        "2. Scan and pick the important files\n"
        "3. Build a compact text context\n"
        "4. Qwen 2.5 (0.5B) explains the project\n"
        "5. *Deep scan:* it also describes every important file and folder"
    )
    st.caption(
        "AI engine: **Ollama** if it is running on this computer, otherwise "
        "the **Hugging Face hosted API** when a token is set in Streamlit Secrets (used on Streamlit Cloud), "
        "else local Transformers (slow)."
    )

# ------------------------------------------------------------------- input --
with st.form("analyze_form"):
    repo_url = st.text_input(
        "GitHub repository URL",
        key="repo_url",
        placeholder="https://github.com/psf/requests",
        help="Paste the URL of a PUBLIC repository (not a link to a single file).",
    )
    col_a, col_b = st.columns([3, 2])
    with col_a:
        deep_scan = st.checkbox(
            "Deep scan: describe every important file and folder",
            value=False,
            help="Adds a File Guide and Folder Guide written by the AI. Much slower (one AI call per file). Leave it off for a fast overview.",
        )
    with col_b:
        n_files = st.select_slider(
            "Files to describe", options=[5, 10, 15, 20, 30], value=5,
            help="More files = a fuller guide but a longer wait (about 5-10 seconds per file on a laptop).",
        )
    submitted = st.form_submit_button("Analyze Repository", type="primary")

if submitted:
    run_analysis(repo_url, deep_scan, n_files)

# ----------------------------------------------------------------- results --
if "error" in st.session_state:
    st.error(st.session_state["error"])

result = st.session_state.get("result")

if not result and "error" not in st.session_state:
    st.markdown(
        '<div class="pipeline">'
        '<div class="pipe" style="animation-delay:.05s"><div class="ico">🌿</div><b>Clone</b><span>Shallow-clones the public repo into a temp folder, deleted afterwards.</span></div>'
        '<div class="pipe" style="animation-delay:.15s"><div class="ico">🔎</div><b>Scan</b><span>Skips junk, ranks files, indexes every folder, class and function.</span></div>'
        '<div class="pipe" style="animation-delay:.25s"><div class="ico">🧩</div><b>Context</b><span>Builds a compact text the small model can handle.</span></div>'
        '<div class="pipe" style="animation-delay:.35s"><div class="ico">🧠</div><b>Qwen 2.5</b><span>Writes the overview, a file guide and a folder guide.</span></div>'
        '<div class="pipe" style="animation-delay:.45s"><div class="ico">✨</div><b>Explore</b><span>Browse the map, search files, download the report.</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

if result:
    context = result["context"]
    described = sum(1 for i in context.inventory if i.summary)

    st.markdown(
        f'<div class="repo-head"><div><a href="{esc(result["url"])}" target="_blank">{esc(result["label"])}</a></div>'
        f'<div><span class="pill cyan">{esc(result["engine"])}</span>'
        f'<span class="pill">{"Deep scan" if result["deep"] else "Quick scan"}</span></div></div>',
        unsafe_allow_html=True,
    )
    if result.get("notice"):
        st.warning(result["notice"])

    stats = [
        ("Files scanned", context.total_files, "after skipping junk"),
        ("Files read by AI", context.analyzed_files, "sent whole to the model"),
        ("Files described", described, "in the File Guide"),
        ("Languages", len(context.languages), ", ".join(list(context.languages)[:3]) or "-"),
        ("Folders", len([f for f in context.folders if f.name != "(root)"]), "top-level"),
    ]
    st.markdown(
        '<div class="stats">' + "".join(
            f'<div class="stat" style="animation-delay:{i * 70}ms"><div class="k">{esc(k)}</div>'
            f'<div class="v">{v}</div><div class="s">{esc(s)}</div></div>'
            for i, (k, v, s) in enumerate(stats)
        ) + "</div>",
        unsafe_allow_html=True,
    )

    tab_overview, tab_map, tab_files, tab_tech, tab_ctx = st.tabs(
        ["✨ Overview", "🗺️ Repository map", "📄 File guide", "🧰 Tech stack", "🔬 Context sent"]
    )

    # ---- Overview: the LLM's explanation, one card per section
    with tab_overview:
        sections = split_sections(result["explanation"])
        if len(sections) >= 3:
            for n, (title, body) in enumerate(sections):
                with st.container(border=True, key=f"sec_{n}"):
                    st.markdown(
                        f'<div class="sec-title"><span class="ico">{section_icon(title)}</span>{esc(title)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(body)
        else:
            with st.container(border=True):
                st.markdown(result["explanation"])
        st.download_button(
            "⬇ Download full report (.md)",
            data=build_report(result),
            file_name=f"{context.repo_name}_explained.md",
            mime="text/markdown",
        )

    # ---- Repository map: folder cards + tree
    with tab_map:
        st.markdown("#### What is in each folder")
        if any(f.summary for f in context.folders):
            st.markdown('<div class="note">Folder descriptions were written by the AI from the files inside each folder.</div>', unsafe_allow_html=True)
        cards = []
        for i, f in enumerate(context.folders):
            title = "Project root" if f.name == "(root)" else f.name + "/"
            desc = (f'<div class="desc">{esc(f.summary)}</div>' if f.summary
                    else '<div class="desc none">No AI description (turn on Deep scan to get one).</div>')
            files = "".join(f'<span class="chip mono">{esc(p.split("/")[-1])}</span>' for p in f.files[:8])
            more = f'<span class="chip">+{f.file_count - 8} more</span>' if f.file_count > 8 else ""
            langs = "".join(lang_chip(l) for l in list(f.languages)[:3])
            cards.append(
                f'<div class="card" style="animation-delay:{i * 60}ms"><h4>📁 {esc(title)}</h4>'
                f'<div class="meta">{f.file_count} file{"" if f.file_count == 1 else "s"}</div>'
                f"{desc}<div>{langs}</div><div style=\"margin-top:.4rem\">{files}{more}</div></div>"
            )
        st.markdown('<div class="grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
        st.markdown("#### Project structure")
        st.code(context.structure, language="text")

    # ---- File guide: searchable cards
    with tab_files:
        c1, c2, c3 = st.columns([3, 2, 2])
        query = c1.text_input("Search files or functions", placeholder="e.g. auth, session, router ...", key="file_query")
        languages = ["All"] + sorted({i.language for i in context.inventory})
        language = c2.selectbox("Language", languages, key="file_lang")
        only_ai = c3.checkbox("Only files with AI summary", value=described > 0, key="file_only_ai")

        rows = sorted(context.inventory, key=lambda i: -i.importance)
        q = query.strip().lower()
        rows = [
            i for i in rows
            if (language == "All" or i.language == language)
            and (not only_ai or i.summary)
            and (not q or q in i.path.lower() or q in i.summary.lower() or any(q in s.lower() for s in i.symbols))
        ]
        st.markdown(f'<div class="note">Showing {min(len(rows), 80)} of {len(rows)} matching files, most important first.</div>', unsafe_allow_html=True)
        file_cards = []
        for n, i in enumerate(rows[:80]):
            summary = (f'<div class="desc">{esc(i.summary)}</div>' if i.summary
                       else '<div class="desc none">Not summarized by the AI (it describes the most important files only).</div>')
            symbols = "".join(f'<span class="chip mono">{esc(s)}</span>' for s in i.symbols[:8])
            lines = f" · {i.lines:,} lines" if i.lines else ""
            file_cards.append(
                f'<div class="card file-card" style="animation-delay:{min(n, 12) * 40}ms">'
                f'<div class="fpath">{esc(i.path)}</div>'
                f'<div class="meta" style="margin:.3rem 0 0">{lang_chip(i.language)}{i.size / 1024:.1f} KB{lines}</div>'
                f"{summary}<div>{symbols}</div></div>"
            )
        if file_cards:
            st.markdown('<div class="grid" style="grid-template-columns:1fr">' + "".join(file_cards) + "</div>", unsafe_allow_html=True)
        else:
            st.info("No files match. Try a different search or filter.")

    # ---- Tech stack
    with tab_tech:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Languages")
            total = sum(context.languages.values()) or 1
            bars = []
            for n, (lang, count) in enumerate(context.languages.items()):
                pct = 100 * count / total
                color = BAR_COLORS[n % len(BAR_COLORS)]
                bars.append(
                    f'<div class="bar-row" style="animation-delay:{n * 80}ms"><div class="lbl"><span>{LANG_ICONS.get(lang, "📄")} {esc(lang)}</span>'
                    f'<span style="color:var(--muted)">{count} files · {pct:.0f}%</span></div>'
                    f'<div class="bar"><div class="fill" style="--w:{pct:.1f}%;background:linear-gradient(90deg,{color},{color}aa)"></div></div></div>'
                )
            st.markdown("".join(bars) or "No source languages detected.", unsafe_allow_html=True)
            other: dict[str, int] = {}
            source_langs = set(context.languages)
            for i in context.inventory:
                if i.language not in source_langs:
                    other[i.language] = other.get(i.language, 0) + 1
            if other:
                st.markdown("#### Other file types")
                st.markdown("".join(f'<span class="chip">{LANG_ICONS.get(l, "📄")} {esc(l)} · {n}</span>' for l, n in sorted(other.items(), key=lambda kv: -kv[1])), unsafe_allow_html=True)
        with right:
            st.markdown("#### Declared dependencies")
            if context.dependencies:
                st.markdown("".join(f'<span class="chip mono">{esc(d)}</span>' for d in context.dependencies), unsafe_allow_html=True)
            else:
                st.caption("No dependency file (requirements.txt, package.json, pyproject.toml, pom.xml) was found.")
            st.markdown("#### AI model")
            st.markdown(f'<span class="chip lang">🧠 {esc(result["engine"])}</span>', unsafe_allow_html=True)

    # ---- Context transparency
    with tab_ctx:
        st.markdown("#### Files sent whole to the AI model")
        st.markdown("".join(
            f'<span class="chip mono">{esc(f.path)}</span><span class="chip">{esc(f.kind)} · {f.chars:,} chars{" · truncated" if f.truncated else ""}</span><br>'
            for f in context.files
        ), unsafe_allow_html=True)
        st.markdown("#### Exact context sent to the AI model")
        st.caption("This is the real text the model received. Nothing about the explanation is hard-coded.")
        st.code(context.text, language="markdown")
