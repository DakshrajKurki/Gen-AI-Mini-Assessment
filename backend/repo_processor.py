"""Repository processing: GitHub clone OR uploaded archive -> pick relevant files -> analyse -> extract."""
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from git import Repo
from git.exc import GitCommandError

from analyzer import (CODE_EXT, CONFIG_EXT, DATA_EXT, DOC_EXT, ENTRY_STEMS, MANIFESTS, analyze, read_source)
from schemas import parse_repo

IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__", "dist", "build",
    ".next", ".nuxt", "target", "vendor", ".idea", ".vscode", "coverage", ".pytest_cache",
    ".mypy_cache", "site-packages", "bower_components", "Pods", ".gradle", "out", "__MACOSX", ".svn",
}
KEEP_DOT_DIRS = {".github"}
SKIP_NAMES = ("-lock.json", ".lock", ".min.js", ".min.css", ".map", "package-lock.json", "yarn.lock", ".DS_Store")

MAX_FILE_BYTES = 80_000
MAX_EXTRA_BYTES = 60_000
MAX_NOTEBOOK_BYTES = 5_000_000
PER_FILE_CHARS = 3_500
TOTAL_CHARS = 12_000
RANKED_KEEP = 14

# upload safety limits
MAX_ARCHIVE_FILES = 30_000
MAX_UNPACKED_BYTES = 800 * 1024 * 1024
MAX_SINGLE_FILE = 50 * 1024 * 1024
TEXT_DATA_EXT = {".csv", ".tsv", ".jsonl"}

_CACHE: "OrderedDict[str, Tuple[float, dict]]" = OrderedDict()
_LOCK = threading.Lock()
_TTL = 3 * 3600


# ------------------------------------------------------------------ git clone
def clone_repo(url: str) -> Path:
    """Shallow-clone into a temp dir (history is not needed)."""
    tmp = tempfile.mkdtemp(prefix="codeatlas_")
    try:
        Repo.clone_from(url, tmp, depth=1, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return Path(tmp)


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def friendly_error(e: Exception) -> str:
    if isinstance(e, GitCommandError):
        s = (e.stderr or str(e)).lower()
        if "not found" in s or "could not read username" in s or "authentication" in s:
            return "Repository not found, or it is private (private repos need a login)."
        return "Git could not clone this repository: " + (e.stderr or str(e)).strip().splitlines()[-1][:200]
    if isinstance(e, (RuntimeError, ValueError)):
        return str(e)
    return f"{type(e).__name__}: {e}"


# ------------------------------------------------------------- safe archives
def _skip_member(name: str) -> bool:
    parts = Path(name).parts
    return any(p in IGNORE_DIRS for p in parts) or Path(name).name in (".DS_Store",)


def _inside(base: Path, target: Path) -> bool:
    return target == base or str(target).startswith(str(base) + os.sep)


def _safe_unzip(data: bytes, dest: Path) -> None:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("That file is not a valid .zip archive.")
    with z:
        infos = z.infolist()
        if len(infos) > MAX_ARCHIVE_FILES:
            raise ValueError(f"The archive has too many files (limit {MAX_ARCHIVE_FILES:,}).")
        if sum(i.file_size for i in infos) > MAX_UNPACKED_BYTES:
            raise ValueError("The archive is too large once unpacked (limit 800 MB).")
        base = dest.resolve()
        for i in infos:
            if _skip_member(i.filename):
                continue
            target = (base / i.filename).resolve()
            if not _inside(base, target):  # zip-slip protection
                continue
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if i.file_size > MAX_SINGLE_FILE:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _safe_untar(data: bytes, dest: Path) -> None:
    try:
        t = tarfile.open(fileobj=io.BytesIO(data))
    except tarfile.TarError:
        raise ValueError("That file is not a valid tar archive.")
    with t:
        base = dest.resolve()
        count = 0
        for m in t:
            count += 1
            if count > MAX_ARCHIVE_FILES:
                raise ValueError(f"The archive has too many files (limit {MAX_ARCHIVE_FILES:,}).")
            if not (m.isfile() or m.isdir()) or _skip_member(m.name):  # no symlinks / devices
                continue
            target = (base / m.name).resolve()
            if not _inside(base, target):
                continue
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if m.size > MAX_SINGLE_FILE:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            f = t.extractfile(m)
            if f:
                with open(target, "wb") as out:
                    shutil.copyfileobj(f, out)


def _single_root(tmp: Path) -> Path:
    """A zipped folder usually unpacks to one top-level directory: use it as the repo root."""
    root = tmp
    for _ in range(3):
        entries = [e for e in root.iterdir() if e.name not in ("__MACOSX", ".DS_Store")]
        if len(entries) == 1 and entries[0].is_dir():
            root = entries[0]
        else:
            break
    return root


def _archive_stem(name: str) -> str:
    n = Path(name).name
    for suf in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar", ".zip"):
        if n.lower().endswith(suf):
            return n[: -len(suf)] or "uploaded-repo"
    return n


# ------------------------------------------------------------------ scanning
def _tier(p: Path, root: Path) -> int:
    name = p.name.lower()
    parts = p.relative_to(root).parts
    depth = len(parts)
    is_doc = name.startswith("readme") or name in MANIFESTS
    if is_doc and depth == 1:
        return 0 if name.startswith("readme") else 1
    if p.stem.lower() in ENTRY_STEMS and depth <= 3:
        return 2
    if is_doc:  # nested READMEs / manifests are low priority
        return 5
    if any(x in ("tests", "test", "examples", "docs", "example", "__tests__") for x in parts[:-1]):
        return 4
    return 3


def scan(root: Path) -> Tuple[List[Path], List[Path], List[str]]:
    """Return (code/readme/manifest candidates, extra docs/config/data samples, all relative paths)."""
    candidates: List[Path] = []
    extras: List[Path] = []
    tree: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS and (not d.startswith(".") or d in KEEP_DOT_DIRS))
        for f in sorted(filenames):
            p = Path(dirpath) / f
            rel = p.relative_to(root).as_posix()
            tree.append(rel)
            ext, low = p.suffix.lower(), f.lower()
            if low.endswith(SKIP_NAMES):
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size == 0:
                continue
            if ext in CODE_EXT or low.startswith("readme") or low in MANIFESTS:
                limit = MAX_NOTEBOOK_BYTES if ext == ".ipynb" else MAX_FILE_BYTES
                if size <= limit:
                    candidates.append(p)
            elif rel.count("/") <= 3 and (
                (ext in DOC_EXT | CONFIG_EXT and size <= MAX_EXTRA_BYTES) or ext in TEXT_DATA_EXT
            ):
                extras.append(p)
    return candidates, extras, tree


def _read_for_prompt(p: Path) -> str:
    if p.suffix.lower() in TEXT_DATA_EXT:  # data files: just a peek at the first rows
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                head = [next(fh, "") for _ in range(6)]
            return "(first rows of a data file)\n" + "".join(head).strip()[:700]
        except OSError:
            return ""
    return read_source(p).strip()


def extract_code(root: Path, name: str) -> dict:
    t0 = time.time()
    candidates, extras, tree = scan(root)
    facts = analyze(root, candidates, tree)
    degree: Dict[str, int] = facts.pop("_degree")
    corpus: str = facts.pop("_corpus")

    def key(p: Path):
        rel = p.relative_to(root).as_posix()
        try:
            size = min(p.stat().st_size, PER_FILE_CHARS)
        except OSError:
            size = 0
        return (_tier(p, root), -degree.get(rel, 0), rel.count("/"), -size)

    def extra_key(p: Path):
        ext = p.suffix.lower()
        kind = 0 if ext in DOC_EXT else (1 if ext in CONFIG_EXT else 2)
        return (kind, p.relative_to(root).as_posix().count("/"), p.name.lower())

    candidates.sort(key=key)
    extras.sort(key=extra_key)

    ranked: List[dict] = []
    for p in candidates + extras:
        if len(ranked) >= RANKED_KEEP:
            break
        text = _read_for_prompt(p)
        if not text:
            continue
        ranked.append({"path": p.relative_to(root).as_posix(), "content": text[:PER_FILE_CHARS],
                       "truncated": len(text) > PER_FILE_CHARS})

    files, used = [], 0
    for f in ranked:
        if used >= TOTAL_CHARS:
            break
        chunk = f["content"][: TOTAL_CHARS - used]
        used += len(chunk)
        files.append({**f, "content": chunk})

    overview = sorted(tree, key=lambda t: (t.count("/"), t))[:70]
    return {
        "name": name,
        "files": files,
        "ranked": ranked,
        "tree": overview,
        "facts": facts,
        "corpus": corpus,
        "stats": {"total_files": len(tree), "files_used": len(files), "chars_sent": used,
                  "languages": facts["languages"]},
        "analyze_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------- cache
def _put(key: str, data: dict) -> None:
    with _LOCK:
        _CACHE[key] = (time.time(), data)
        _CACHE.move_to_end(key)
        while len(_CACHE) > 10:
            _CACHE.popitem(last=False)


def _get(key: str) -> Optional[dict]:
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < _TTL:
            return hit[1]
    return None


def get_source(source_id: str) -> dict:
    data = _get(source_id)
    if not data:
        raise RuntimeError("This uploaded repository is no longer in memory (the backend restarted or it expired). Please upload it again.")
    return data


def get_extraction(url: str, refresh: bool = False) -> Tuple[dict, bool]:
    """Clone + analyse a GitHub repo (cached so compare-mode runs see identical input)."""
    owner, repo = parse_repo(url)
    key = f"gh:{owner}/{repo}".lower()
    if not refresh:
        hit = _get(key)
        if hit:
            return hit, True
    t0 = time.time()
    root = clone_repo(url)
    clone_s = round(time.time() - t0, 2)
    try:
        data = extract_code(root, f"{owner}/{repo}")
    finally:
        cleanup(root)
    data["clone_s"] = clone_s
    data["source"] = "github"
    _put(key, data)
    return data, False


def extract_upload(blobs: List[Tuple[str, bytes]]) -> Tuple[str, dict, bool]:
    """Unpack uploaded archives / loose files and analyse them. Returns (source_id, data, was_cached)."""
    h = hashlib.sha1()
    for fname, b in blobs:
        h.update(fname.encode())
        h.update(hashlib.sha1(b).digest())
    sid = "upload:" + h.hexdigest()[:16]
    hit = _get(sid)
    if hit:
        return sid, hit, True

    tmp = Path(tempfile.mkdtemp(prefix="codeatlas_up_"))
    try:
        archives = []
        for fname, b in blobs:
            low = fname.lower()
            if low.endswith(".zip"):
                _safe_unzip(b, tmp)
                archives.append(fname)
            elif low.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz")):
                _safe_untar(b, tmp)
                archives.append(fname)
            else:  # loose file
                safe = re.sub(r"[^\w.\- ]", "_", Path(fname).name) or "file"
                (tmp / safe).write_bytes(b)
        root = _single_root(tmp)
        if len(archives) == 1 and len(blobs) == 1:
            name = root.name if root != tmp else _archive_stem(archives[0])
        elif len(blobs) == 1:
            name = Path(blobs[0][0]).name
        else:
            name = "uploaded-files"
        if not any(root.rglob("*")):
            raise ValueError("The upload is empty (or contained only ignored folders such as node_modules).")
        data = extract_code(root, name)
    finally:
        cleanup(tmp)
    data["clone_s"] = 0.0
    data["source"] = "upload"
    _put(sid, data)
    return sid, data, False
