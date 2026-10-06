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
    LLMError,
)

logger = logging.getLogger(__name__)

EXAMPLES = {
    "psf/requests (Python)": "https://github.com/psf/requests",
    "pallets/flask (Python)": "https://github.com/pallets/flask",
    "expressjs/express (JavaScript)": "https://github.com/expressjs/express",
}

# ----------------------------------------------------------------- styling --
st.markdown(
    """
    <style>
      .block-container {padding-top: 3.5rem; max-width: 1150px;}
      .hero {
        background: linear-gradient(120deg, #4f46e5 0%, #0ea5e9 100%);
        color: #fff; padding: 1.8rem 2rem; border-radius: 16px; margin-bottom: 1.4rem;
      }
      .hero h1 {margin: 0; font-size: 2.2rem; color: #fff;}
      .hero p  {margin: .3rem 0 0; font-size: 1.05rem; opacity: .92; color: #fff;}
      .repo-card {
        border: 1px solid rgba(128,128,128,.3); border-radius: 12px;
        padding: .8rem 1.1rem; margin: .8rem 0 1rem;
      }
      [data-testid="stFormSubmitButton"] button {
        background: #4f46e5; color: #fff; border: none; padding: .5rem 1.4rem; font-weight: 600;
      }
      [data-testid="stFormSubmitButton"] button:hover {background: #4338ca; color: #fff;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: .7rem 1rem;
      }
    </style>
    <div class="hero">
      <h1>🔍 RepoLens AI</h1>
      <p>Local GitHub Repository Code Explainer</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- helpers --
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


def set_example(url: str) -> None:
    st.session_state["repo_url"] = url


def run_analysis(url: str) -> None:
    """Clone -> process -> LLM -> store in session_state. Always cleans up."""
    st.session_state.pop("error", None)
    st.session_state.pop("result", None)
    progress = st.progress(0, text="Validating URL...")
    repo_path = None
    try:
        progress.progress(10, text="Cloning repository from GitHub...")
        repo_path = clone_repository(url)

        progress.progress(40, text="Analyzing repository structure and selecting files...")
        context = build_repository_context(repo_path)

        progress.progress(55, text="Asking the AI model to explain the project...")
        with st.spinner("Generating explanation. A small model on CPU can take 30-90 seconds "
                        "(the first online run also downloads the model)..."):
            result = explain_repository(context)

        progress.progress(100, text="Done!")
        st.session_state["result"] = {
            "url": url,
            "label": repo_label(url),
            "context": context,
            "explanation": result.text,
            "engine": result.engine,
        }
    except (RepositoryError, LLMError) as exc:
        st.session_state["error"] = str(exc)
    except Exception:  # last line of defence: log details, show something friendly
        logger.exception("Unexpected error while analyzing %s", url)
        st.session_state["error"] = "Something unexpected went wrong. Please try again."
    finally:
        cleanup_repository(repo_path)   # the temporary clone is ALWAYS deleted
        progress.empty()


# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Try an example")
    for name, example_url in EXAMPLES.items():
        st.button(name, on_click=set_example, args=(example_url,), use_container_width=True)
    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. Clone the repo (GitPython)\n"
        "2. Pick the important files\n"
        "3. Build a small text context\n"
        "4. Qwen 2.5 (0.5B) writes the explanation\n"
        "5. Show it here"
    )
    st.caption(
        "AI engine: **Ollama** if it is running on this computer, otherwise "
        "**Hugging Face Transformers** (used on Streamlit Cloud). Same model family: Qwen 2.5 0.5B."
    )

# ------------------------------------------------------------------- input --
with st.form("analyze_form"):
    repo_url = st.text_input(
        "GitHub repository URL",
        key="repo_url",
        placeholder="https://github.com/psf/requests",
        help="Paste the URL of a PUBLIC repository (not a link to a single file).",
    )
    submitted = st.form_submit_button("Analyze Repository", type="primary")

if submitted:
    run_analysis(repo_url)

# ----------------------------------------------------------------- results --
if "error" in st.session_state:
    st.error(st.session_state["error"])

result = st.session_state.get("result")
if result:
    context = result["context"]
    st.success("Analysis complete! The explanation below was generated by the AI model from the repository's actual files.")

    st.markdown(
        f'<div class="repo-card"><b>Repository:</b> '
        f'<a href="{result["url"]}" target="_blank">{result["label"]}</a></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files analyzed", context.analyzed_files)
    c2.metric("Files scanned", context.total_files)
    c3.metric("Languages", len(context.languages))
    c4.metric("AI engine", result["engine"].split(" · ")[0])

    st.subheader("Generated explanation")
    sections = split_sections(result["explanation"])
    if len(sections) >= 3:
        for title, body in sections:
            with st.expander(title, expanded=True):
                st.markdown(body)
    else:
        st.markdown(result["explanation"])

    st.download_button(
        "Download explanation (.md)",
        data=f"# {result['label']}\n\n{result['explanation']}\n",
        file_name=f"{context.repo_name}_explanation.md",
        mime="text/markdown",
    )

    st.subheader("Repository details")
    with st.expander("Important files analyzed"):
        for f in context.files:
            note = " (truncated)" if f.truncated else ""
            st.markdown(f"- `{f.path}` &mdash; {f.kind}, {f.chars:,} characters{note}")

    with st.expander("Project structure"):
        st.code(context.structure, language="text")

    with st.expander("Technology information"):
        if context.languages:
            st.markdown("**Languages detected (by file extension)**")
            st.markdown(", ".join(f"`{lang}` ({n})" for lang, n in context.languages.items()))
        if context.dependencies:
            st.markdown("**Declared dependencies**")
            st.markdown(", ".join(f"`{d}`" for d in context.dependencies))
        st.markdown(f"**Model used for the explanation:** {result['engine']}")

    with st.expander("Exact context sent to the AI model (for transparency)"):
        st.code(context.text, language="markdown")
