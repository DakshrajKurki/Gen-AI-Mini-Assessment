"""
code_processor.py
=================
Turns a cloned repository into a SMALL, structured text "context" for the LLM.

Pipeline
--------
1. Walk the repository, skipping junk folders (.git, node_modules, venv, ...).
2. Skip test files, lock files, minified files, binaries and huge files.
3. Score the remaining files so that README, config files and entry points come
   first, followed by the most "central" source files.
4. Read each chosen file safely (UTF-8, errors replaced) and cut it down to a
   fixed budget so the total never exceeds MAX_TOTAL_CHARS.
5. Assemble: structure tree + README + config files + source snippets.

A 0.5B-parameter model has a small context window, so the limits below matter.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .repository import RepositoryError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- limits ----
MAX_FILES = 10             # README + config + source files sent to the LLM
MAX_TOTAL_CHARS = 8_000    # cap on README/config/code text sent (~2.5k tokens)
README_CHARS = 2_000
CONFIG_CHARS = 800
SOURCE_CHARS = 1_200       # per source file
MAX_CONFIG_FILES = 3
MAX_FILE_BYTES = 300_000   # larger files are probably generated: skip
MAX_FILES_SCANNED = 20_000 # stop walking absurdly large repositories

# ----------------------------------------------------------- file rules ----
IGNORED_DIRS = {
    ".git", ".github", ".idea", ".vscode", "venv", ".venv", "env", "node_modules",
    "__pycache__", "dist", "build", "coverage", "htmlcov", "target", "out",
    "vendor", "site-packages", "docs", "doc", ".tox", ".pytest_cache",
    ".mypy_cache", ".next", ".nuxt", "bower_components", "migrations",
}
TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "testing", "e2e"}
IGNORED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "pipfile.lock",
    "composer.lock", "cargo.lock", "go.sum", "conftest.py", "setup.cfg",
}

LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript (React)",
    ".ts": "TypeScript", ".tsx": "TypeScript (React)", ".java": "Java",
    ".cpp": "C++", ".c": "C", ".cs": "C#", ".go": "Go", ".rs": "Rust",
    ".php": "PHP", ".html": "HTML", ".css": "CSS", ".sql": "SQL",
}
SOURCE_EXTENSIONS = set(LANGUAGES)

MANIFEST_NAMES = [  # in order of usefulness
    "requirements.txt", "package.json", "pom.xml", "build.gradle", "pyproject.toml",
    "go.mod", "cargo.toml", "composer.json",
]
ENTRY_NAMES = {
    "main.py", "app.py", "server.py", "index.py", "manage.py", "wsgi.py", "__main__.py",
    "index.js", "server.js", "app.js", "main.js", "index.ts", "main.ts", "app.ts",
    "app.jsx", "app.tsx", "main.jsx", "main.tsx", "main.go", "main.rs", "main.c",
    "main.cpp", "program.cs", "main.java", "application.java", "index.php", "index.html",
}
IMPORTANT_FOLDERS = {"src", "app", "lib", "core", "backend", "server", "api", "frontend", "cmd"}
# Code in these folders is usually demo/helper code, not the heart of the project.
LOW_VALUE_FOLDERS = {
    "examples", "example", "samples", "sample", "demo", "demos", "benchmark",
    "benchmarks", "scripts", "tools", "fixtures", "playground", "ext",
}
# Not worth listing in the directory tree shown to the LLM.
TREE_HIDDEN_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".ai", ".pdf", ".woff", ".woff2",
    ".ttf", ".eot", ".mp4", ".zip", ".lock",
}

# --- "whole repository" inventory (used for the file index and the File Guide) ---
INDEX_CHARS = 2_500        # budget for the compact file index inside the LLM context
MAX_INVENTORY = 250        # files listed in the inventory
MAX_SNIPPETS = 30          # files that keep a short code snippet for per-file summaries
SNIPPET_CHARS = 1_000
MAX_SYMBOLS = 10

# Display names for files that are not "source code" but still part of the repo.
OTHER_LANGUAGES = {
    ".md": "Markdown", ".rst": "reStructuredText", ".txt": "Text", ".json": "JSON",
    ".yml": "YAML", ".yaml": "YAML", ".toml": "TOML", ".ini": "Config", ".cfg": "Config",
    ".xml": "XML", ".sh": "Shell", ".bat": "Batch", ".ipynb": "Notebook", ".rb": "Ruby",
    ".kt": "Kotlin", ".swift": "Swift", ".vue": "Vue", ".scss": "SCSS", ".env": "Config",
    ".gradle": "Gradle", ".csv": "Data",
}
SPECIAL_FILES = {"dockerfile": "Docker", "makefile": "Makefile", "license": "License", "procfile": "Config"}

# Regexes that find the main "things" defined in a file (classes, functions ...).
_SYMBOL_PATTERNS = {
    ".py": r"^(?:async\s+def|def|class)\s+([A-Za-z_]\w*)",
    ".js": r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class)\s+([A-Za-z_$][\w$]*)"
           r"|^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>"
           r"|^(?:module\.)?exports\.([A-Za-z_$][\w$]*)\s*=",
    ".java": r"\b(?:class|interface|enum|record)\s+([A-Za-z_]\w*)",
    ".cs": r"\b(?:class|interface|enum|struct|record)\s+([A-Za-z_]\w*)",
    ".go": r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)|^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)",
    ".rs": r"^\s*(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait)\s+([A-Za-z_]\w*)",
    ".php": r"\b(?:function|class|interface|trait)\s+([A-Za-z_]\w*)",
    ".sql": r"(?i)create\s+(?:or\s+replace\s+)?(?:table|view|function|procedure)\s+(?:if\s+not\s+exists\s+)?([\w.]+)",
}
_SYMBOL_PATTERNS[".jsx"] = _SYMBOL_PATTERNS[".ts"] = _SYMBOL_PATTERNS[".tsx"] = _SYMBOL_PATTERNS[".js"]


# ---------------------------------------------------------------- results ---
@dataclass
class FileEntry:
    """One file that was (partly) sent to the LLM."""
    path: str       # relative path, forward slashes
    kind: str       # README | Config | Entry point | Source
    chars: int      # characters actually sent
    truncated: bool


@dataclass
class InventoryItem:
    """One file of the repository, described statically (no LLM involved)."""
    path: str
    language: str
    size: int                       # bytes
    lines: int | None               # None when the file was too large to count
    symbols: list[str] = field(default_factory=list)   # classes / functions found
    importance: float = 0.0
    snippet: str = ""               # first ~1000 chars (only kept for the top files)
    summary: str = ""               # filled in later by the LLM ("File Guide")


@dataclass
class FolderInfo:
    """A top-level folder (or the repository root) and what it contains."""
    name: str                       # "src", "backend", ... or "(root)"
    file_count: int
    languages: dict[str, int] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)      # most important files first
    summary: str = ""               # filled in later by the LLM


@dataclass
class RepositoryContext:
    repo_name: str
    text: str                      # <- this is what is sent to the LLM
    structure: str                 # pretty directory tree
    files: list[FileEntry] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)   # language -> file count
    dependencies: list[str] = field(default_factory=list)
    total_files: int = 0           # files scanned (after ignoring junk)
    inventory: list[InventoryItem] = field(default_factory=list)
    folders: list[FolderInfo] = field(default_factory=list)

    @property
    def analyzed_files(self) -> int:
        return len(self.files)

    def __str__(self) -> str:
        return self.text


@dataclass
class _Candidate:
    rel: str
    abs: Path
    size: int
    ext: str
    name: str
    depth: int


# ------------------------------------------------------------ public API ----
def build_repository_context(repo_path: str | os.PathLike) -> RepositoryContext:
    """Analyze a cloned repository and return the context for the LLM."""
    root = Path(repo_path)
    if not root.is_dir():
        raise RepositoryError("The repository folder could not be found.")

    candidates, test_candidates, truncated_scan = _scan(root)
    all_files = candidates + test_candidates
    if not all_files:
        raise RepositoryError(
            "This repository looks empty. There are no files to analyze."
        )
    if not candidates:  # a repo that is *only* tests: better than nothing
        candidates = test_candidates

    repo_name = root.name
    languages = _count_languages(all_files)
    dependencies = _read_dependencies(candidates)
    structure = _render_tree(
        repo_name,
        [c.rel for c in candidates
         if not c.name.startswith(".") and c.ext not in TREE_HIDDEN_EXTENSIONS],
    )

    inventory = _build_inventory(candidates)
    folders = _build_folders(candidates, inventory)

    chosen = _choose_files(candidates)
    if not chosen:
        raise RepositoryError(
            "No readable source code was found. This tool understands Python, JavaScript, "
            "TypeScript, Java, C/C++, C#, Go, Rust, PHP, HTML, CSS and SQL projects."
        )

    entries: list[FileEntry] = []
    sections: list[str] = []
    remaining = MAX_TOTAL_CHARS
    for cand, kind, cap in chosen:
        if remaining < 300 or len(entries) >= MAX_FILES:
            break
        read = _read_text(cand.abs, min(cap, remaining))
        if read is None:
            continue
        text, was_truncated = read
        if kind == "README":
            text = _clean_readme(text)
        if len(text.strip()) < 30:      # empty / trivial file, not worth the budget
            continue
        remaining -= len(text)
        entries.append(FileEntry(cand.rel, kind, len(text), was_truncated))
        sections.append(_format_section(cand, kind, text, was_truncated))

    if not entries:
        raise RepositoryError("The files in this repository could not be read as text.")

    header = [
        "# REPOSITORY CONTEXT",
        f"Repository name: {repo_name}",
        f"Files scanned: {len(all_files)}"
        + (" (very large repository, scan stopped early)" if truncated_scan else "")
        + f" | Files shown below: {len(entries)}",
    ]
    if languages:
        header.append(
            "Languages detected (by file extension): "
            + ", ".join(f"{lang} ({n} files)" for lang, n in languages.items())
        )
    if dependencies:
        header.append("Declared dependencies: " + ", ".join(dependencies))
    header += ["", "## Directory structure", structure, ""]
    header += [_format_index(inventory, folders), ""]

    text = "\n".join(header) + "\n" + "\n\n".join(sections)
    return RepositoryContext(
        repo_name=repo_name,
        text=text,
        structure=structure,
        files=entries,
        languages=languages,
        dependencies=dependencies,
        total_files=len(all_files),
        inventory=inventory,
        folders=folders,
    )


# --------------------------------------------------------------- scanning ---
def _is_test_file(name: str) -> bool:
    lower = name.lower()
    return (
        lower.startswith("test_")
        or lower.endswith(("_test.py", "_test.go", "_test.js", "_spec.rb"))
        or ".test." in lower
        or ".spec." in lower
    )


def _scan(root: Path) -> tuple[list[_Candidate], list[_Candidate], bool]:
    """Walk the repo. Returns (normal files, test files, stopped_early)."""
    normal: list[_Candidate] = []
    tests: list[_Candidate] = []
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):  # symlinked dirs are not followed
        in_test_dir = False
        rel_dir = Path(dirpath).relative_to(root)
        if any(part.lower() in TEST_DIRS for part in rel_dir.parts):
            in_test_dir = True
        # prune in place so os.walk never enters ignored folders
        dirnames[:] = sorted(
            d for d in dirnames
            if d.lower() not in IGNORED_DIRS and not d.startswith(".")
            and not os.path.islink(os.path.join(dirpath, d))
        )
        for filename in sorted(filenames):
            full = Path(dirpath) / filename
            lower = filename.lower()
            if (
                os.path.islink(full)
                or lower in IGNORED_FILES
                or lower.endswith((".min.js", ".min.css", ".map"))
            ):
                continue
            try:
                size = full.stat().st_size
            except OSError:
                continue
            seen += 1
            if seen > MAX_FILES_SCANNED:
                return normal, tests, True
            rel = (rel_dir / filename).as_posix()
            cand = _Candidate(rel, full, size, full.suffix.lower(), lower, len(rel_dir.parts))
            (tests if in_test_dir or _is_test_file(filename) else normal).append(cand)
    return normal, tests, False


def _count_languages(files: list[_Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in files:
        lang = LANGUAGES.get(f.ext)
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# -------------------------------------------------------- file selection ----
def _score(c: _Candidate) -> float:
    """Higher = more useful for explaining the project."""
    score = 30.0
    if c.name in ENTRY_NAMES:
        score += 40
    score -= 6 * c.depth
    folders = [part.lower() for part in c.rel.split("/")[:-1]]
    if any(f in IMPORTANT_FOLDERS for f in folders):
        score += 10
    if any(f in LOW_VALUE_FOLDERS for f in folders):
        score -= 40
    if c.name == "__init__.py":
        score -= 25
    if c.size < 200:
        score -= 15
    score += min(c.size, 6000) / 600      # prefer files with real content (max +10)
    return score


def _choose_files(candidates: list[_Candidate]) -> list[tuple[_Candidate, str, int]]:
    """Return [(file, kind, char_budget)] in the order they appear in the prompt."""
    chosen: list[tuple[_Candidate, str, int]] = []
    used: set[str] = set()

    # 1) top-level README (any extension: README, README.md, README.rst ...)
    readmes = [c for c in candidates if c.depth == 0 and c.name.startswith("readme")]
    readmes.sort(key=lambda c: (c.ext != ".md", c.name))
    if readmes and readmes[0].size <= MAX_FILE_BYTES:
        chosen.append((readmes[0], "README", README_CHARS))
        used.add(readmes[0].rel)

    # 2) build / dependency files near the top of the tree
    manifests = [c for c in candidates if c.name in MANIFEST_NAMES and c.depth <= 1 and c.size <= MAX_FILE_BYTES]
    manifests.sort(key=lambda c: (c.depth, MANIFEST_NAMES.index(c.name)))
    for c in manifests[:MAX_CONFIG_FILES]:
        chosen.append((c, "Config", CONFIG_CHARS))
        used.add(c.rel)

    # 3) source files, best score first
    sources = [
        c for c in candidates
        if c.ext in SOURCE_EXTENSIONS and c.rel not in used and 0 < c.size <= MAX_FILE_BYTES
    ]
    sources.sort(key=lambda c: (-_score(c), c.rel))
    for c in sources:
        kind = "Entry point" if c.name in ENTRY_NAMES else "Source"
        chosen.append((c, kind, SOURCE_CHARS))
    return chosen


# ------------------------------------------------- whole-repo inventory ------
def _language_of(c: _Candidate) -> str:
    if c.ext in LANGUAGES:
        return LANGUAGES[c.ext]
    if c.name in SPECIAL_FILES:
        return SPECIAL_FILES[c.name]
    if c.name.startswith("readme"):
        return "Markdown" if c.ext == ".md" else "Text"
    return OTHER_LANGUAGES.get(c.ext, "")


def _extract_symbols(ext: str, text: str) -> list[str]:
    pattern = _SYMBOL_PATTERNS.get(ext)
    if not pattern:
        return []
    found: list[str] = []
    for match in re.finditer(pattern, text, flags=re.M):
        name = next((g for g in match.groups() if g), None)
        if name and name not in found and not name.startswith("__"):
            found.append(name)
            if len(found) >= MAX_SYMBOLS:
                break
    return found


def _build_inventory(candidates: list[_Candidate]) -> list[InventoryItem]:
    """List every (non-junk) file with language, size, line count and main symbols."""
    eligible = [
        c for c in candidates
        if not c.name.startswith(".") and c.ext not in TREE_HIDDEN_EXTENSIONS and _language_of(c)
    ]
    # If there are too many files keep the most important ones.
    eligible.sort(key=lambda c: (c.ext in SOURCE_EXTENSIONS, _score(c)), reverse=True)
    eligible = eligible[:MAX_INVENTORY]

    # Only the best source files keep a code snippet (for the per-file LLM summaries).
    snippet_paths = {
        c.rel for c in sorted(
            (c for c in eligible if c.ext in SOURCE_EXTENSIONS and c.size > 0),
            key=lambda c: (-_score(c), c.rel),
        )[:MAX_SNIPPETS]
    }

    items: list[InventoryItem] = []
    for c in eligible:
        lines, symbols, snippet = None, [], ""
        if c.ext in SOURCE_EXTENSIONS and 0 < c.size <= MAX_FILE_BYTES:
            try:
                data = c.abs.read_bytes()
            except OSError:
                data = b""
            if data and b"\x00" not in data[:4096]:
                text = data.decode("utf-8", errors="replace")
                lines = text.count("\n") + 1
                symbols = _extract_symbols(c.ext, text[:150_000])
                if c.rel in snippet_paths:
                    read = _read_text(c.abs, SNIPPET_CHARS)
                    if read and len(read[0].strip()) >= 30:
                        snippet = read[0]
        items.append(InventoryItem(
            path=c.rel, language=_language_of(c), size=c.size, lines=lines,
            symbols=symbols, snippet=snippet,
            importance=round(_score(c) + (20 if c.ext in SOURCE_EXTENSIONS else -20), 1),
        ))
    items.sort(key=lambda i: i.path)
    return items


def _build_folders(candidates: list[_Candidate], inventory: list[InventoryItem]) -> list[FolderInfo]:
    """Group files by top-level folder. '(root)' holds files in the top directory."""
    counts: dict[str, int] = {}
    langs: dict[str, dict[str, int]] = {}
    for c in candidates:
        if c.name.startswith(".") or c.ext in TREE_HIDDEN_EXTENSIONS:
            continue
        key = c.rel.split("/")[0] if "/" in c.rel else "(root)"
        counts[key] = counts.get(key, 0) + 1
        lang = _language_of(c)
        if lang:
            langs.setdefault(key, {})
            langs[key][lang] = langs[key].get(lang, 0) + 1

    by_folder: dict[str, list[InventoryItem]] = {}
    for item in inventory:
        key = item.path.split("/")[0] if "/" in item.path else "(root)"
        by_folder.setdefault(key, []).append(item)

    folders = []
    for key, n in counts.items():
        ranked = sorted(by_folder.get(key, []), key=lambda i: -i.importance)
        folders.append(FolderInfo(
            name=key, file_count=n,
            languages=dict(sorted(langs.get(key, {}).items(), key=lambda kv: -kv[1])),
            files=[i.path for i in ranked[:12]],
        ))
    # real folders first (largest first), the root files last
    folders.sort(key=lambda f: (f.name == "(root)", -f.file_count, f.name))
    return folders


def _format_index(inventory: list[InventoryItem], folders: list[FolderInfo]) -> str:
    """Compact text index of the WHOLE repository for the LLM (budgeted)."""
    lines = ["## Repository index (what is in the repository)"]
    lines.append("Folders: " + "; ".join(
        f"{f.name}{'' if f.name == '(root)' else '/'} ({f.file_count} file{'' if f.file_count == 1 else 's'}"
        + (f", mostly {next(iter(f.languages))}" if f.languages else "") + ")"
        for f in folders[:15]
    ))
    used = sum(len(x) for x in lines)
    for item in sorted(inventory, key=lambda i: -i.importance):
        detail = item.language + (f", {item.lines} lines" if item.lines else "")
        if item.symbols:
            detail += ": " + ", ".join(item.symbols[:6])
        line = f"- {item.path} ({detail})"
        if used + len(line) > INDEX_CHARS:
            lines.append("- ... (more files not listed)")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


# ----------------------------------------------------------- file reading ---
def _read_text(path: Path, limit: int) -> tuple[str, bool] | None:
    """Read at most `limit` characters. Returns (text, was_truncated) or None."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(limit * 4 + 4)  # UTF-8 uses up to 4 bytes per character
            more_on_disk = bool(fh.read(1))
    except OSError:
        return None
    if b"\x00" in data[:4096]:             # binary file
        return None
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    truncated = more_on_disk or len(text) > limit
    if len(text) > limit:
        cut = text[:limit]
        newline = cut.rfind("\n")
        text = cut[:newline] if newline > limit * 0.5 else cut   # cut at a line boundary
    return text.rstrip(), truncated


def _clean_readme(text: str) -> str:
    """Drop badges, images and HTML comments that waste the LLM's context."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = [
        ln for ln in text.splitlines()
        if not ln.lstrip().startswith(("![", "[![", "<img", "<p align", "</p>"))
        and "shields.io" not in ln and "badge" not in ln.lower()
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _format_section(c: _Candidate, kind: str, text: str, truncated: bool) -> str:
    note = "\n... [file truncated]" if truncated else ""
    if kind == "README":
        return f"## README ({c.rel})\n{text}{note}"
    lang = c.ext.lstrip(".")
    return f"## File: {c.rel}  [{kind}]\n```{lang}\n{text}{note}\n```"


# ---------------------------------------------------------- dependencies ----
def _read_dependencies(candidates: list[_Candidate], limit: int = 25) -> list[str]:
    """Pull dependency names out of requirements.txt / package.json / pyproject.toml / pom.xml."""
    found: list[str] = []
    for c in sorted(candidates, key=lambda c: c.depth):
        if c.depth > 1 or c.size > MAX_FILE_BYTES or c.name not in (
            "requirements.txt", "package.json", "pom.xml", "pyproject.toml"
        ):
            continue
        try:
            raw = c.abs.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if c.name == "requirements.txt":
            for line in raw.splitlines():
                line = line.split("#")[0].strip()
                if line and not line.startswith(("-", "git+", "http")):
                    found.append(re.split(r"[<>=!~\[;@ ]", line)[0])
        elif c.name == "package.json":
            try:
                found += list(json.loads(raw).get("dependencies", {}))
            except (ValueError, AttributeError):
                pass
        elif c.name == "pyproject.toml":
            block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", raw, flags=re.S)
            for spec in re.findall(r"[\"']([^\"']+)[\"']", block.group(1)) if block else []:
                found.append(re.split(r"[<>=!~\[;@ ]", spec.strip())[0])
        else:  # pom.xml
            found += re.findall(r"<artifactId>([^<]+)</artifactId>", raw)
    unique = list(dict.fromkeys(n for n in found if n))
    return unique[:limit]


# ----------------------------------------------------------- tree drawing ---
def _render_tree(name: str, paths: list[str], max_depth: int = 3,
                 max_per_dir: int = 12, max_lines: int = 50) -> str:
    tree: dict = {}
    for p in paths:
        node = tree
        parts = p.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None            # None marks a file

    def count_files(node: dict) -> int:
        return sum(1 if v is None else count_files(v) for v in node.values())

    lines = [f"{name}/"]

    def walk(node: dict, indent: str, depth: int) -> None:
        entries = sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0].lower()))  # folders first
        for key, child in entries[:max_per_dir]:
            if len(lines) >= max_lines:
                return
            if child is None:
                lines.append(f"{indent}{key}")
            elif depth >= max_depth:
                lines.append(f"{indent}{key}/  ({count_files(child)} files)")
            else:
                lines.append(f"{indent}{key}/")
                walk(child, indent + "  ", depth + 1)
        hidden = len(entries) - max_per_dir
        if hidden > 0 and len(lines) < max_lines:
            lines.append(f"{indent}... (+{hidden} more)")

    walk(tree, "  ", 1)
    if len(lines) >= max_lines:
        lines.append("  ...")
    return "\n".join(lines)
