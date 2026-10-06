"""Static analysis -> deterministic "verified facts" about a repository.

The LLM only *explains*. It should never have to guess the stack, entry points,
routes or module relationships, so we extract those with real parsers (ast, json,
toml, regex) and hand them to the model as ground truth.
"""
from __future__ import annotations

import ast
import json
import posixpath
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:  # Python 3.11+
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

EXT_LANG = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript (React)", ".ts": "TypeScript",
    ".tsx": "TypeScript (React)", ".java": "Java", ".kt": "Kotlin", ".go": "Go", ".rs": "Rust",
    ".c": "C", ".h": "C/C++ header", ".cpp": "C++", ".hpp": "C++", ".cs": "C#", ".php": "PHP",
    ".rb": "Ruby", ".swift": "Swift", ".dart": "Dart", ".scala": "Scala", ".sql": "SQL",
    ".sh": "Shell", ".html": "HTML", ".css": "CSS", ".vue": "Vue", ".svelte": "Svelte",
    ".ipynb": "Jupyter Notebook", ".r": "R", ".m": "Objective-C/MATLAB",
}
CODE_EXT = set(EXT_LANG)
MANIFESTS = {
    "requirements.txt", "package.json", "pyproject.toml", "pom.xml", "build.gradle", "go.mod",
    "cargo.toml", "composer.json", "gemfile", "pubspec.yaml", "dockerfile", "docker-compose.yml",
    "setup.py",
}
ENTRY_STEMS = {"main", "app", "index", "server", "manage", "run", "cli", "__main__", "program", "wsgi", "asgi",
               "streamlit_app", "bot", "start", "api"}
AUX_DIRS = {"tests", "test", "examples", "example", "docs", "doc", "__tests__", "benchmarks", "samples", "demo", "demos"}


def is_aux(rel: str) -> bool:
    return any(part in AUX_DIRS for part in rel.split("/")[:-1])
JS_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".vue", ".svelte")

# dependency / import name (lowercase) -> (display name, category)
KNOWN_TECH: Dict[str, Tuple[str, str]] = {
    "flask": ("Flask", "Web framework"), "fastapi": ("FastAPI", "Web framework"),
    "django": ("Django", "Web framework"), "express": ("Express", "Web framework"),
    "koa": ("Koa", "Web framework"), "@nestjs/core": ("NestJS", "Web framework"),
    "spring-boot-starter-web": ("Spring Boot", "Web framework"), "gin": ("Gin", "Web framework"),
    "laravel/framework": ("Laravel", "Web framework"), "uvicorn": ("Uvicorn", "Server"),
    "gunicorn": ("Gunicorn", "Server"), "streamlit": ("Streamlit", "UI framework"),
    "gradio": ("Gradio", "UI framework"), "tkinter": ("Tkinter", "UI framework"),
    "pygame": ("Pygame", "Game / graphics"), "react": ("React", "Frontend"),
    "next": ("Next.js", "Frontend"), "vue": ("Vue", "Frontend"), "svelte": ("Svelte", "Frontend"),
    "@angular/core": ("Angular", "Frontend"), "jquery": ("jQuery", "Frontend"),
    "tailwindcss": ("Tailwind CSS", "Styling"), "bootstrap": ("Bootstrap", "Styling"),
    "vite": ("Vite", "Tooling"), "webpack": ("Webpack", "Tooling"),
    "typescript": ("TypeScript", "Language tooling"), "axios": ("Axios", "HTTP client"),
    "electron": ("Electron", "Desktop"),
    "sqlite3": ("SQLite", "Database"), "sqlalchemy": ("SQLAlchemy", "Database / ORM"),
    "psycopg2": ("PostgreSQL", "Database"), "psycopg2-binary": ("PostgreSQL", "Database"),
    "asyncpg": ("PostgreSQL", "Database"), "pymysql": ("MySQL", "Database"),
    "mysql-connector-python": ("MySQL", "Database"), "mysqlclient": ("MySQL", "Database"),
    "pymongo": ("MongoDB", "Database"), "mongoose": ("MongoDB", "Database"),
    "mongodb": ("MongoDB", "Database"), "redis": ("Redis", "Cache"),
    "prisma": ("Prisma", "Database / ORM"), "@prisma/client": ("Prisma", "Database / ORM"),
    "sequelize": ("Sequelize", "Database / ORM"), "firebase": ("Firebase", "Database"),
    "firebase-admin": ("Firebase", "Database"), "supabase": ("Supabase", "Database"),
    "pandas": ("Pandas", "Data"), "numpy": ("NumPy", "Data"), "scipy": ("SciPy", "Data"),
    "matplotlib": ("Matplotlib", "Visualisation"), "seaborn": ("Seaborn", "Visualisation"),
    "plotly": ("Plotly", "Visualisation"), "scikit-learn": ("scikit-learn", "Machine learning"),
    "tensorflow": ("TensorFlow", "Machine learning"), "torch": ("PyTorch", "Machine learning"),
    "keras": ("Keras", "Machine learning"), "xgboost": ("XGBoost", "Machine learning"),
    "lightgbm": ("LightGBM", "Machine learning"), "opencv-python": ("OpenCV", "Machine learning"),
    "pillow": ("Pillow", "Data"), "nltk": ("NLTK", "Machine learning"), "spacy": ("spaCy", "Machine learning"),
    "transformers": ("Hugging Face Transformers", "GenAI"), "langchain": ("LangChain", "GenAI"),
    "llama-index": ("LlamaIndex", "GenAI"), "openai": ("OpenAI API", "GenAI"),
    "anthropic": ("Anthropic API", "GenAI"), "ollama": ("Ollama", "GenAI"), "groq": ("Groq API", "GenAI"),
    "google-generativeai": ("Gemini API", "GenAI"), "sentence-transformers": ("Sentence Transformers", "GenAI"),
    "chromadb": ("ChromaDB", "GenAI"), "faiss-cpu": ("FAISS", "GenAI"),
    "requests": ("Requests", "HTTP client"), "httpx": ("HTTPX", "HTTP client"),
    "beautifulsoup4": ("BeautifulSoup", "Scraping"), "selenium": ("Selenium", "Scraping"),
    "scrapy": ("Scrapy", "Scraping"), "playwright": ("Playwright", "Scraping"),
    "pydantic": ("Pydantic", "Validation"), "gitpython": ("GitPython", "Git"),
    "celery": ("Celery", "Task queue"), "pytest": ("pytest", "Testing"), "jest": ("Jest", "Testing"),
    "python-dotenv": ("python-dotenv", "Config"), "click": ("Click", "CLI"), "typer": ("Typer", "CLI"),
    "jinja2": ("Jinja2", "Templating"), "werkzeug": ("Werkzeug", "Web framework"),
}
IMPORT_ALIAS = {
    "sklearn": "scikit-learn", "cv2": "opencv-python", "pil": "pillow", "bs4": "beautifulsoup4",
    "git": "gitpython", "dotenv": "python-dotenv",
}
CATEGORY_ORDER = [
    "Web framework", "UI framework", "Frontend", "GenAI", "Machine learning", "Data", "Visualisation",
    "Database", "Database / ORM", "Cache", "CLI", "HTTP client", "Scraping", "Server", "Validation",
    "Templating", "Git", "Task queue", "Styling", "Config", "Mobile", "Embedded", "DevOps", "Docs site",
    "Desktop", "Game / graphics", "Tooling", "Language tooling", "Testing",
]
HIDE_IN_GRAPH = {"Testing", "Tooling", "Language tooling", "Config"}
PALETTE = ["#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#60a5fa", "#fb923c"]


# ------------------------------------------------------- any-repo-type support
DOC_EXT = {".md", ".rst", ".txt", ".adoc", ".mdx"}
CONFIG_EXT = {".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".xml", ".env", ".tf", ".gradle", ".properties"}
DATA_EXT = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3", ".pkl", ".h5", ".npy",
            ".npz", ".jsonl", ".feather"}
MEDIA_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".mp4", ".mov", ".mp3", ".wav", ".pdf",
             ".ttf", ".woff", ".woff2"}
MODEL_EXT = {".pt", ".pth", ".onnx", ".pb", ".safetensors", ".gguf", ".ckpt", ".joblib", ".tflite"}
INVENTORY_ORDER = ["Source code", "Notebooks", "Docs", "Config", "Data", "ML models", "Images & media", "Other"]

BASENAME_MARKERS = {
    "androidmanifest.xml": ("Android", "Mobile"), "pubspec.yaml": ("Dart / Flutter", "Mobile"),
    "info.plist": ("iOS / macOS", "Mobile"), "podfile": ("CocoaPods", "Mobile"),
    "dockerfile": ("Docker", "DevOps"), "docker-compose.yml": ("Docker Compose", "DevOps"),
    "docker-compose.yaml": ("Docker Compose", "DevOps"), "makefile": ("Make", "Tooling"),
    "cmakelists.txt": ("CMake", "Tooling"), "mkdocs.yml": ("MkDocs", "Docs site"), "_config.yml": ("Jekyll", "Docs site"),
    "docusaurus.config.js": ("Docusaurus", "Docs site"), "vercel.json": ("Vercel", "DevOps"),
    "netlify.toml": ("Netlify", "DevOps"), "procfile": ("Heroku", "DevOps"), "chart.yaml": ("Helm", "DevOps"),
    "kustomization.yaml": ("Kubernetes", "DevOps"), "jenkinsfile": ("Jenkins", "DevOps"),
    ".gitlab-ci.yml": ("GitLab CI", "DevOps"), "environment.yml": ("Conda", "Tooling"),
    "platformio.ini": ("PlatformIO", "Embedded"), "ansible.cfg": ("Ansible", "DevOps"),
}
FRONTEND_TECH = {"React", "Next.js", "Vue", "Svelte", "Angular", "jQuery"}
BACKEND_TECH = {"Flask", "FastAPI", "Django", "Express", "Spring Boot", "Gin", "Laravel", "NestJS", "Koa"}
GENAI_TECH = {"Hugging Face Transformers", "LangChain", "LlamaIndex", "OpenAI API", "Anthropic API", "Ollama",
              "Groq API", "Gemini API", "Sentence Transformers", "ChromaDB", "FAISS"}
ML_TECH = {"scikit-learn", "TensorFlow", "PyTorch", "Keras", "XGBoost", "LightGBM", "OpenCV", "NLTK", "spaCy",
           "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "SciPy"}

SPRING = re.compile(r'@(Get|Post|Put|Delete|Patch|Request)Mapping\(\s*(?:value\s*=\s*|path\s*=\s*)?\{?\s*"([^"]*)"')
GO_ROUTE = re.compile(r'\.(GET|POST|PUT|DELETE|PATCH)\(\s*"(/[^"]*)"')
GO_HANDLE = re.compile(r'\bHandleFunc\(\s*"(/[^"]*)"')
LARAVEL = re.compile(r"Route::(get|post|put|delete|patch)\(\s*['\"](/?[^'\"]*)['\"]")
RAILS = re.compile(r"^\s*(get|post|put|patch|delete)\s+['\"]([^'\"]+)['\"]", re.M)
CS_ROUTE = re.compile(r'\[Http(Get|Post|Put|Delete|Patch)(?:\(\s*"([^"]*)"\s*\))?\]')
OTHER_ROUTE_EXT = {".java", ".kt", ".go", ".php", ".rb", ".cs"}


def find_other_routes(ext: str, name: str, text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if ext in (".java", ".kt"):
        out = [("ANY" if m.group(1) == "Request" else m.group(1).upper(), "/" + (m.group(2) or "").lstrip("/")) for m in SPRING.finditer(text)]
    elif ext == ".go":
        out = [(m.group(1), m.group(2)) for m in GO_ROUTE.finditer(text)] + [("ANY", m.group(1)) for m in GO_HANDLE.finditer(text)]
    elif ext == ".php":
        out = [(m.group(1).upper(), "/" + m.group(2).lstrip("/")) for m in LARAVEL.finditer(text)]
    elif ext == ".rb" and name == "routes.rb":
        out = [(m.group(1).upper(), m.group(2)) for m in RAILS.finditer(text)]
    elif ext == ".cs":
        out = [(m.group(1).upper(), "/" + (m.group(2) or "").lstrip("/")) for m in CS_ROUTE.finditer(text)]
    return out[:30]


def file_category(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".ipynb":
        return "Notebooks"
    if ext in CODE_EXT:
        return "Source code"
    if ext in DOC_EXT:
        return "Docs"
    if ext in CONFIG_EXT:
        return "Config"
    if ext in DATA_EXT:
        return "Data"
    if ext in MODEL_EXT:
        return "ML models"
    if ext in MEDIA_EXT:
        return "Images & media"
    return "Other"


def inventory(tree: List[str]) -> Dict[str, int]:
    c = Counter(file_category(p) for p in tree)
    return {k: c[k] for k in INVENTORY_ORDER if c.get(k)}


def structure(tree: List[str]) -> dict:
    dirs: Dict[str, List[str]] = {}
    root_files: List[str] = []
    for p in tree:
        if "/" in p:
            dirs.setdefault(p.split("/")[0], []).append(p)
        else:
            root_files.append(p)
    out = []
    for d, paths in dirs.items():
        kinds = Counter(EXT_LANG.get(Path(p).suffix.lower()) or file_category(p) for p in paths)
        out.append({"name": d + "/", "files": len(paths), "kind": kinds.most_common(1)[0][0]})
    out.sort(key=lambda x: -x["files"])
    return {"dirs": out[:14], "root_files": root_files[:16]}


def detect_markers(tree: List[str]) -> List[Tuple[str, str]]:
    found: Dict[str, str] = {}
    low = [p.lower() for p in tree]
    names = {Path(p).name for p in low}
    for n, (disp, cat) in BASENAME_MARKERS.items():
        if n in names:
            found.setdefault(disp, cat)
    if any(p.startswith(".github/workflows/") for p in low):
        found["GitHub Actions"] = "DevOps"
    if any(p.endswith(".tf") for p in low):
        found["Terraform"] = "DevOps"
    if any(".xcodeproj/" in p for p in low):
        found["iOS / macOS"] = "Mobile"
    if any(p.endswith(".ino") for p in low):
        found["Arduino"] = "Embedded"
    if "manifest.json" in names and names & {"background.js", "content.js", "popup.html", "service-worker.js"}:
        found["Browser extension"] = "Desktop"
    return list(found.items())


def classify_project(names: Set[str], inv: Dict[str, int], loc: Counter, endpoints: list, entries: List[str], total: int) -> str:
    src, nb, docs, data = inv.get("Source code", 0), inv.get("Notebooks", 0), inv.get("Docs", 0), inv.get("Data", 0)
    total = max(total, 1)
    if names & {"Android", "Dart / Flutter", "iOS / macOS"}:
        return "Mobile app"
    if "Browser extension" in names:
        return "Browser extension"
    if names & {"Arduino", "PlatformIO"}:
        return "Embedded / hardware project"
    if names & GENAI_TECH:
        return "GenAI / LLM application" + (" with a web API" if endpoints else "")
    if nb and (nb >= max(1, src // 2) or names & ML_TECH):
        return "Data-science / machine-learning project (notebooks)"
    if names & {"Streamlit", "Gradio"}:
        return "Interactive data app (Streamlit / Gradio)"
    has_front = bool(names & FRONTEND_TECH) or (loc.get("HTML", 0) + loc.get("CSS", 0) > 0 and loc.get("JavaScript", 0) > 0)
    backend = bool(endpoints) or bool(names & BACKEND_TECH)
    if backend and has_front:
        return "Full-stack web application"
    if backend:
        return "Backend API / web service"
    if has_front:
        return "Frontend web app / website"
    if names & {"Click", "Typer"} or any("cli" in Path(e).stem.lower() for e in entries):
        return "Command-line tool"
    if "Pygame" in names:
        return "Game"
    if names & {"Electron", "Tkinter"}:
        return "Desktop application"
    if src < max(3, total * 0.1) and names & {"Terraform", "Kubernetes", "Helm", "Docker", "Docker Compose", "Ansible"}:
        return "Infrastructure / DevOps configuration"
    if names & {"MkDocs", "Jekyll", "Docusaurus"} or (docs >= total * 0.5 and src <= 2):
        return "Documentation / knowledge-base repository"
    if data >= total * 0.4 or (data and src == 0):
        return "Dataset / data repository"
    if src >= 3 and not entries:
        return "Software library / package"
    if src:
        return "Software project"
    return "Mixed-content repository"


# ------------------------------------------------------------------ reading
def read_notebook(p: Path) -> str:
    try:
        nb = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    out = []
    for c in nb.get("cells", []):
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if not src.strip():
            continue
        if c.get("cell_type") == "code":
            out.append(src)
        elif c.get("cell_type") == "markdown":
            out.append("# " + src.replace("\n", "\n# "))
    return "\n\n".join(out)


def read_source(p: Path, limit: int = 200_000) -> str:
    try:
        if p.suffix.lower() == ".ipynb":
            return read_notebook(p)[:limit]
        return p.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


# ---------------------------------------------------------------- manifests
def parse_manifest(name: str, text: str) -> Tuple[List[str], Dict[str, str]]:
    """Return (dependency names lowercase, extra info like run scripts)."""
    deps: List[str] = []
    extra: Dict[str, str] = {}
    n = name.lower()
    try:
        if n.startswith("requirements") and n.endswith(".txt"):
            for line in text.splitlines():
                line = line.split("#")[0].strip()
                if not line or line.startswith(("-", "git+", "http")):
                    continue
                deps.append(re.split(r"[<>=!~;\[ @]", line)[0].lower())
        elif n == "package.json":
            d = json.loads(text)
            for key in ("dependencies", "devDependencies"):
                deps += [k.lower() for k in (d.get(key) or {})]
            for k, v in (d.get("scripts") or {}).items():
                if k in ("start", "dev", "build", "test"):
                    extra[f"npm run {k}"] = str(v)[:80]
            if d.get("main"):
                extra["main"] = str(d["main"])
        elif n == "pyproject.toml":
            if tomllib:
                d = tomllib.loads(text)
                for item in (d.get("project", {}).get("dependencies") or []):
                    deps.append(re.split(r"[<>=!~;\[ @]", item.strip())[0].lower())
                poetry = d.get("tool", {}).get("poetry", {}).get("dependencies", {})
                deps += [k.lower() for k in poetry if k.lower() != "python"]
            else:
                m = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
                if m:
                    deps += [re.split(r"[<>=!~;\[ @]", s)[0].lower() for s in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))]
        elif n == "setup.py":
            m = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.S)
            if m:
                deps += [re.split(r"[<>=!~;\[ @]", s)[0].lower() for s in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))]
        elif n == "pom.xml":
            deps += [a.lower() for a in re.findall(r"<artifactId>([^<]+)</artifactId>", text)]
        elif n == "build.gradle":
            deps += [g.split(":")[1].lower() for g in re.findall(r"['\"]([\w.\-]+:[\w.\-]+)(?::[^'\"]*)?['\"]", text)]
        elif n == "go.mod":
            deps += [ln.split()[0].split("/")[-1].lower() for ln in text.splitlines()
                     if re.match(r"^\s+[\w.\-]+\.[\w]+/", ln)]
        elif n == "cargo.toml" and tomllib:
            deps += [k.lower() for k in (tomllib.loads(text).get("dependencies") or {})]
        elif n == "composer.json":
            deps += [k.lower() for k in (json.loads(text).get("require") or {})]
        elif n == "gemfile":
            deps += [g.lower() for g in re.findall(r"gem\s+['\"]([^'\"]+)['\"]", text)]
    except Exception:
        pass
    return [d for d in deps if d], extra


# ------------------------------------------------------------------- python
ROUTE_ATTRS = {"route", "get", "post", "put", "delete", "patch", "head", "options"}
MAIN_RE = re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]")


def analyze_python(text: str) -> dict:
    info = {"raw_imports": [], "top_imports": set(), "classes": [], "functions": [], "routes": [],
            "has_main": bool(MAIN_RE.search(text))}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return info
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                info["raw_imports"].append((0, a.name, []))
                info["top_imports"].add(a.name.split(".")[0].lower())
        elif isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            info["raw_imports"].append((node.level or 0, node.module or "", names))
            if not node.level and node.module:
                info["top_imports"].add(node.module.split(".")[0].lower())
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr.lower() in ROUTE_ATTRS:
                    path = None
                    if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                        path = dec.args[0].value
                    for kw in dec.keywords:
                        if kw.arg == "path" and isinstance(kw.value, ast.Constant):
                            path = str(kw.value.value)
                    if path is None or not (path.startswith("/") or path == ""):
                        continue
                    attr = dec.func.attr.lower()
                    methods = ["GET"] if attr == "route" else [attr.upper()]
                    if attr == "route":
                        for kw in dec.keywords:
                            if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                                methods = [str(e.value).upper() for e in kw.value.elts if isinstance(e, ast.Constant)] or methods
                    for m in methods:
                        info["routes"].append({"method": m, "path": path or "/", "handler": node.name})
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            info["classes"].append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("__"):
            info["functions"].append(node.name)
    return info


def _resolve(cand: str, rels: Set[str]) -> List[str]:
    for c in (cand + ".py", cand + "/__init__.py"):
        if c in rels:
            return [c]
    return []


def py_edges(py_infos: Dict[str, dict]) -> Set[Tuple[str, str]]:
    rels = set(py_infos)
    index: Dict[str, str] = {}
    for rel in sorted(rels, key=lambda r: r.count("/")):
        parts = rel[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for i in range(len(parts)):
            index.setdefault(".".join(parts[i:]), rel)
    edges: Set[Tuple[str, str]] = set()
    for rel, info in py_infos.items():
        base = rel.split("/")[:-1]
        for level, module, names in info["raw_imports"]:
            targets: List[str] = []
            if level > 0:
                b = base[: max(len(base) - (level - 1), 0)]
                mp = module.split(".") if module else []
                targets += _resolve("/".join(b + mp), rels) if mp else []
                for n in names:
                    targets += _resolve("/".join(b + mp + [n]), rels)
            else:
                for name in [module] + [f"{module}.{n}" for n in names]:
                    parts = name.split(".")
                    for k in range(len(parts), 0, -1):
                        key = ".".join(parts[:k])
                        if key in index:
                            targets.append(index[key])
                            break
            for t in targets:
                if t != rel:
                    edges.add((rel, t))
    return edges


# ---------------------------------------------------------------- javascript
JS_REL = re.compile(r"""(?:from\s+|import\s+|require\(\s*|import\(\s*)['"](\.{1,2}/[^'"]*)['"]""")
JS_PKG = re.compile(r"""(?:from\s+|require\(\s*|import\s+)['"]([^./'"][^'"]*)['"]""")
JS_ROUTE = re.compile(r"""\b(?:app|router|server|api)\.(get|post|put|delete|patch)\(\s*['"`](/[^'"`]*)['"`]""")


def js_edges(files: Dict[str, str]) -> Set[Tuple[str, str]]:
    rels = set(files)
    edges: Set[Tuple[str, str]] = set()
    for rel, text in files.items():
        for m in JS_REL.finditer(text):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(rel), m.group(1)))
            cands = [target] + [target + e for e in JS_EXT] + [f"{target}/index{e}" for e in JS_EXT]
            for c in cands:
                if c in rels and c != rel:
                    edges.add((rel, c))
                    break
    return edges


def js_packages(text: str) -> Set[str]:
    out = set()
    for m in JS_PKG.finditer(text):
        name = m.group(1)
        parts = name.split("/")
        out.add((parts[0] + "/" + parts[1] if name.startswith("@") and len(parts) > 1 else parts[0]).lower())
    return out


# ------------------------------------------------------------------- README
def readme_summary(text: str) -> dict:
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    title, paras, cur = None, [], []
    for line in t.splitlines():
        s = line.strip()
        if s.startswith("#"):
            if title is None:
                title = s.lstrip("# ").strip()
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        if not s or set(s) <= set("-=*_|: "):
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    for para in paras:
        if len(para) >= 40:
            return {"title": title, "summary": para[:450]}
    return {"title": title, "summary": ""}


# --------------------------------------------------------------------- graph
def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def make_dot(edges: Set[Tuple[str, str]], entries: List[str], file_tech: Dict[str, Set[str]], max_nodes: int = 36):
    nodes: Counter = Counter()
    for a, b in edges:
        nodes[a] += 1
        nodes[b] += 1
    for f in file_tech:
        nodes.setdefault(f, 0)
    if not nodes:
        return None, 0, 0
    if len(nodes) > max_nodes:
        keep = set(entries) | {n for n, _ in nodes.most_common(max_nodes)}
        edges = {(a, b) for a, b in edges if a in keep and b in keep}
        nodes = Counter({n: c for n, c in nodes.items() if n in keep})
    tech_count = Counter(t for f, ts in file_tech.items() if f in nodes for t in ts)
    top_tech = {t for t, _ in tech_count.most_common(8)}
    if not edges and not top_tech:
        return None, 0, 0

    folders: Dict[str, str] = {}
    lines = [
        "digraph repo {",
        '  rankdir=LR; bgcolor="transparent"; pad=0.2; nodesep=0.3; ranksep=0.8;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fontcolor="#e6e9ff", fillcolor="#121638", color="#3a4080", penwidth=1.4];',
        '  edge [color="#6c72a3", arrowsize=0.7];',
    ]
    for n in nodes:
        top = n.split("/")[0] if "/" in n else "."
        color = folders.setdefault(top, PALETTE[len(folders) % len(PALETTE)])
        label = "/".join(n.split("/")[-2:])
        if n in entries:
            lines.append(f'  {_q(n)} [label={_q("▶ " + label)}, fillcolor="#14304a", color="#22d3ee", penwidth=2.6];')
        else:
            lines.append(f'  {_q(n)} [label={_q(label)}, color="{color}"];')
    for t in top_tech:
        lines.append(f'  {_q("tech:" + t)} [label={_q(t)}, shape=ellipse, fillcolor="#2a1a4a", color="#a78bfa"];')
    for a, b in sorted(edges):
        lines.append(f"  {_q(a)} -> {_q(b)};")
    for f, ts in file_tech.items():
        if f in nodes:
            for t in ts:
                if t in top_tech:
                    lines.append(f'  {_q(f)} -> {_q("tech:" + t)} [style=dashed, color="#a78bfa"];')
    lines.append("}")
    return "\n".join(lines), len(nodes), len(edges)


# ------------------------------------------------------------------ analyze
def analyze(root: Path, candidates: List[Path], tree: List[str]) -> dict:
    loc: Counter = Counter()
    deps: Set[str] = set()
    extras: Dict[str, str] = {}
    manifests: List[str] = []
    py_infos: Dict[str, dict] = {}
    js_texts: Dict[str, str] = {}
    all_imports: Set[str] = set()
    file_pkgs: Dict[str, Set[str]] = {}
    endpoints: List[dict] = []
    readme_text = ""
    snippets: List[str] = []
    has_docker = False

    ordered = sorted(candidates, key=lambda p: len(p.relative_to(root).parts))[:1500]
    for p in ordered:
        rel = p.relative_to(root).as_posix()
        name, ext = p.name.lower(), p.suffix.lower()
        text = read_source(p)
        if not text.strip():
            continue
        depth = rel.count("/")
        if name.startswith("readme") and depth == 0 and not readme_text:
            readme_text = text
        if name in MANIFESTS and depth <= 3 and not is_aux(rel):
            manifests.append(rel)
            if name in ("dockerfile", "docker-compose.yml"):
                has_docker = True
            d, e = parse_manifest(name, text)
            deps.update(d)
            extras.update(e)
        if ext in EXT_LANG:
            lines = sum(1 for ln in text.splitlines() if ln.strip())
            loc[EXT_LANG[ext]] += lines
            snippets.append(text[:1500].lower())
            if ext == ".py" and len(py_infos) < 400:
                py_infos[rel] = analyze_python(text)
            elif ext in JS_EXT:
                js_texts[rel] = text
                if not is_aux(rel):
                    for m in JS_ROUTE.finditer(text):
                        endpoints.append({"method": m.group(1).upper(), "path": m.group(2), "file": rel, "handler": ""})
            elif ext in OTHER_ROUTE_EXT and not is_aux(rel):
                for method, path in find_other_routes(ext, name, text):
                    endpoints.append({"method": method, "path": path, "file": rel, "handler": ""})

    # tests / examples / docs describe how the project is *used*, not what it *is*:
    # keep them out of the stack, graph and entry points unless the repo has nothing else.
    core_py = {r: i for r, i in py_infos.items() if not is_aux(r)} or py_infos
    core_js = {r: t for r, t in js_texts.items() if not is_aux(r)} or js_texts
    all_imports = set()
    file_pkgs = {}
    for rel, info in core_py.items():
        all_imports |= info["top_imports"]
        file_pkgs[rel] = info["top_imports"]
        for r in info["routes"]:
            endpoints.append({**r, "file": rel})
    for rel, text in core_js.items():
        pk = js_packages(text)
        all_imports |= pk
        file_pkgs[rel] = pk

    # technologies
    seen: Dict[str, str] = {}
    for n in list(deps) + list(all_imports):
        key = IMPORT_ALIAS.get(n, n)
        if key in KNOWN_TECH:
            disp, cat = KNOWN_TECH[key]
            seen.setdefault(disp, cat)
    if has_docker:
        seen.setdefault("Docker", "DevOps")
    for disp, cat in detect_markers(tree):
        seen.setdefault(disp, cat)
    technologies = [{"name": d, "category": c} for d, c in seen.items()]
    technologies.sort(key=lambda t: (CATEGORY_ORDER.index(t["category"]) if t["category"] in CATEGORY_ORDER else 99, t["name"]))

    # graph
    edges = py_edges(core_py) | js_edges(core_js)
    degree: Counter = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1

    entries: List[Tuple[int, str]] = []
    for rel in list(core_py) + list(core_js):
        stem = Path(rel).stem.lower()
        if (rel.count("/") <= 2 and stem in ENTRY_STEMS) or core_py.get(rel, {}).get("has_main"):
            entries.append((rel.count("/"), rel))
    if not entries:  # tiny projects: a lone top-level script is the entry point
        top = [r for r in list(core_py) + list(core_js) if "/" not in r]
        if 0 < len(top) <= 3:
            entries = [(0, r) for r in top]
    entry_points = []
    for _, rel in sorted(entries):
        if rel not in entry_points:
            entry_points.append(rel)
    entry_points = entry_points[:6]

    file_tech: Dict[str, Set[str]] = {}
    for rel, pk in file_pkgs.items():
        ts = set()
        for n in pk:
            key = IMPORT_ALIAS.get(n, n)
            if key in KNOWN_TECH and KNOWN_TECH[key][1] not in HIDE_IN_GRAPH:
                ts.add(KNOWN_TECH[key][0])
        if ts:
            file_tech[rel] = ts
    dot, n_nodes, n_edges = make_dot(edges, entry_points, file_tech)

    # symbols for the most relevant python files
    def _rank(rel: str):
        info = core_py[rel]
        return (rel in entry_points, degree.get(rel, 0), len(info["classes"]) + len(info["functions"]))

    symbols = {}
    for rel in sorted(core_py, key=_rank, reverse=True)[:8]:
        info = core_py[rel]
        if info["classes"] or info["functions"]:
            symbols[rel] = {"classes": info["classes"][:8], "functions": info["functions"][:10]}

    local_imports: Dict[str, List[str]] = {}
    for a, b in sorted(edges):
        local_imports.setdefault(a, []).append(b)

    corpus = " ".join([
        " ".join(tree).lower(), " ".join(sorted(deps)), " ".join(sorted(all_imports)),
        " ".join(t["name"].lower() for t in technologies),
        readme_text.lower()[:20000], " ".join(snippets),
    ])[:600_000]

    seen_ep = set()
    uniq_ep = []
    for e in endpoints:
        k = (e["method"], e["path"], e["file"])
        if k not in seen_ep:
            seen_ep.add(k)
            uniq_ep.append(e)

    inv = inventory(tree)
    tech_names = {t["name"] for t in technologies}
    project_type = classify_project(tech_names, inv, loc, uniq_ep, entry_points, len(tree))

    return {
        "project_type": project_type,
        "inventory": inv,
        "structure": structure(tree),
        "languages": dict(loc.most_common()),
        "technologies": technologies,
        "dependencies": sorted(deps)[:60],
        "manifests": manifests[:12],
        "entry_points": entry_points,
        "endpoints": uniq_ep[:30],
        "symbols": symbols,
        "local_imports": local_imports,
        "readme": readme_summary(readme_text) if readme_text else {"title": None, "summary": ""},
        "scripts": extras,
        "graph_dot": dot,
        "graph_nodes": n_nodes,
        "graph_edges": n_edges,
        "_degree": dict(degree),
        "_corpus": corpus,
    }


def facts_text(f: dict, max_chars: int = 3600) -> str:
    out: List[str] = []
    if f.get("project_type"):
        out.append(f"Project type (heuristic guess from the files – confirm with the README): {f['project_type']}")
    if f.get("inventory"):
        out.append("Repository contents: " + ", ".join(f"{v} {k.lower()}" for k, v in f["inventory"].items()))
    st = f.get("structure") or {}
    if st.get("dirs"):
        out.append("Top-level folders: " + "; ".join(f"{d['name']} ({d['files']} files, mostly {d['kind']})" for d in st["dirs"][:10]))
    if st.get("root_files"):
        out.append("Files in the repository root: " + ", ".join(st["root_files"][:14]))
    if f["languages"]:
        out.append("Languages (non-blank lines): " + ", ".join(f"{k} {v}" for k, v in list(f["languages"].items())[:6]))
    if f["technologies"]:
        out.append("Technologies detected from dependency files and imports: " +
                   ", ".join(f"{t['name']} ({t['category']})" for t in f["technologies"][:14]))
    elif f["dependencies"]:
        out.append("Dependencies: " + ", ".join(f["dependencies"][:20]))
    if f["entry_points"]:
        out.append("Entry points: " + ", ".join(f["entry_points"]))
    if f["scripts"]:
        out.append("Run scripts: " + "; ".join(f"{k} = {v}" for k, v in list(f["scripts"].items())[:5]))
    if f["endpoints"]:
        out.append("HTTP endpoints:\n" + "\n".join(
            f"  {e['method']} {e['path']}  ({e['file']}{(':' + e['handler']) if e['handler'] else ''})" for e in f["endpoints"][:15]))
    if f["local_imports"]:
        out.append("Which files import which (local modules):\n" + "\n".join(
            f"  {a} -> {', '.join(bs[:5])}" for a, bs in list(f["local_imports"].items())[:12]))
    if f["symbols"]:
        out.append("Key definitions:\n" + "\n".join(
            f"  {rel}: classes [{', '.join(s['classes'])}] functions [{', '.join(s['functions'])}]"
            for rel, s in list(f["symbols"].items())[:6]))
    if f["readme"]["summary"]:
        out.append("README says: " + f["readme"]["summary"])
    return "\n".join(out)[:max_chars]


# -------------------------------------------------------------- verification
GENERIC = {"language", "framework", "library", "database", "and", "the", "for", "web", "frontend", "backend",
           "front-end", "back-end", "programming", "api", "apis", "server", "markup", "stylesheet", "style", "sheets"}
LANG_EXT = {"python": ".py", "javascript": ".js", "typescript": ".ts", "html": ".html", "html5": ".html",
            "css": ".css", "css3": ".css", "java": ".java", "c++": ".cpp", "c#": ".cs", "go": ".go",
            "rust": ".rs", "php": ".php", "ruby": ".rb", "sql": ".sql", "shell": ".sh", "bash": ".sh",
            "json": ".json", "markdown": ".md", "kotlin": ".kt", "swift": ".swift", "dart": ".dart"}


def verify_technologies(markdown: str, corpus: str) -> List[dict]:
    """Heuristic grounding check: is each technology the model listed visible in the repo?"""
    m = re.search(r"##\s*Main Technologies\s*\n(.*?)(?=\n##|\Z)", markdown, re.S | re.I)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not re.match(r"^[-*•]\s+", line):
            continue
        name = re.sub(r"^[-*•]\s+", "", line)
        name = re.sub(r"\*\*|`", "", name)
        name = re.split(r"\s[—–-]\s|:|\(|,", name)[0].strip()
        if not name:
            continue
        toks = []
        for t in re.findall(r"[a-z0-9.+#]+", name.lower()):
            t = t[:-3] if t.endswith(".js") and len(t) > 3 else t
            if t not in GENERIC and len(t) > 1:
                toks.append(t)
        found = False
        for t in toks:
            if t in LANG_EXT:
                found = LANG_EXT[t] in corpus or (t == "python" and ".ipynb" in corpus)
            else:
                found = t in corpus
            if found:
                break
        out.append({"name": name, "found": bool(toks) and found})
        if len(out) >= 12:
            break
    return out
