"""HTML builders. Everything is returned as ONE line (no blank lines / indentation) so
Streamlit's markdown parser never turns it into a code block."""
from __future__ import annotations

import html

APP_NAME = "CodeAtlas"
TAGLINE = "Map any repository in plain English."
STAGES = ["GitHub Repo", "Code Processing", "Local LLM", "Backend", "Frontend", "Explanation"]

PALETTES = {
    "Aurora": dict(a1="#38bdf8", a2="#8b5cf6", a3="#f472b6", bg=("#060816", "#0b1030", "#140a2e", "#06121f")),
    "Emerald": dict(a1="#34d399", a2="#22d3ee", a3="#a3e635", bg=("#03100d", "#06201b", "#07202a", "#031018")),
    "Sunset": dict(a1="#fb923c", a2="#f43f5e", a3="#a855f7", bg=("#130609", "#240b16", "#1a0b2a", "#0c0613")),
}
INV_ICONS = {"Source code": "💻", "Notebooks": "📓", "Docs": "📚", "Config": "⚙️", "Data": "🗂️",
             "ML models": "🧠", "Images & media": "🖼️", "Other": "📦"}

LANG_COLORS = {
    "Python": "#3572A5", "JavaScript": "#f1e05a", "TypeScript": "#3178c6", "HTML": "#e34c26",
    "CSS": "#9b6bff", "Java": "#b07219", "C++": "#f34b7d", "C": "#8a8a8a", "Go": "#00ADD8",
    "Rust": "#dea584", "Jupyter Notebook": "#DA5B0B", "Shell": "#89e051", "PHP": "#4F5D95",
    "Ruby": "#CC342D", "Kotlin": "#A97BFF", "Swift": "#F05138", "Dart": "#00B4AB", "C#": "#178600",
}
FALLBACK = ["#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#60a5fa"]

PHASES = {
    "queued": ("Waiting for its turn", "#6c72a3"),
    "clone": ("Cloning repository", "#fbbf24"),
    "extract": ("Analysing code", "#fbbf24"),
    "loading": ("Loading model into memory", "#60a5fa"),
    "reading": ("Reading the prompt", "#a78bfa"),
    "generating": ("Generating", "#34d399"),
    "done": ("Finished", "#34d399"),
    "error": ("Failed", "#f87171"),
}

esc = html.escape


def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


def theme_css(name: str) -> str:
    p = PALETTES.get(name, PALETTES["Aurora"])
    return (":root{" + f"--a1:{p['a1']};--a2:{p['a2']};--a3:{p['a3']};"
            f"--a1-rgb:{_hex_rgb(p['a1'])};--a2-rgb:{_hex_rgb(p['a2'])};--a3-rgb:{_hex_rgb(p['a3'])};"
            f"--bg0:{p['bg'][0]};--bg1:{p['bg'][1]};--bg2:{p['bg'][2]};--bg3:{p['bg'][3]};"
            "--ink:#e8ebff;--muted:#9aa1cf;--card:rgba(10,13,36,.80);" + "}")


def logo_svg(size: int = 56, uid: str = "cal") -> str:
    g = f'url(#{uid})'
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="{uid}" x1="6" y1="6" x2="58" y2="58" gradientUnits="userSpaceOnUse">'
        '<stop offset="0" style="stop-color:var(--a1)"/><stop offset=".55" style="stop-color:var(--a2)"/>'
        '<stop offset="1" style="stop-color:var(--a3)"/></linearGradient></defs>'
        f'<circle cx="32" cy="32" r="26" stroke="{g}" stroke-width="3"/>'
        f'<ellipse cx="32" cy="32" rx="11" ry="26" stroke="{g}" stroke-width="2" opacity=".5"/>'
        f'<path d="M6 32h52M10 20h44M10 44h44" stroke="{g}" stroke-width="2" opacity=".3"/>'
        f'<path d="M19 41l13-17 15 11" stroke="{g}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="19" cy="41" r="4.5" fill="{g}"/><circle cx="32" cy="24" r="4.5" fill="{g}"/>'
        f'<circle cx="47" cy="35" r="4.5" fill="{g}"/></svg>'
    )


def hero_html() -> str:
    return (
        f'<div class="hero"><div class="hero-logo">{logo_svg(64, "heroLogo")}</div>'
        '<span class="badge">100% local · private · open-source models</span>'
        f'<h1>{APP_NAME}</h1><p class="tag">{TAGLINE}</p>'
        '<p class="sub">Paste a GitHub link or upload a project. A local AI reads it live, then tells you '
        "what's inside, what it's about and how it works.</p>"
        '<div class="pills"><span>🔗 GitHub links</span><span>📦 Zip uploads</span><span>🧠 Live model view</span>'
        "<span>🗺️ Architecture map</span><span>💬 Ask the repo</span></div></div>"
    )


def brand_html() -> str:
    return f'<div class="brand">{logo_svg(36, "sideLogo")}<div><b>{APP_NAME}</b><small>{TAGLINE}</small></div></div>'


def pipeline_html(active: int = -1, done_upto: int = -1, first: str = "GitHub Repo") -> str:
    parts = []
    for i, name in enumerate([first] + STAGES[1:]):
        cls = "active" if i == active else ("done" if i <= done_upto else "")
        parts.append(f'<div class="node {cls}">{esc(name)}</div>')
    return '<div class="pipe">' + '<span class="arrow">➜</span>'.join(parts) + "</div>"


def fmt_num(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def preview_card(p: dict) -> str:
    if p.get("note"):
        return (
            '<div class="glass"><div class="live"><i></i> LIVE PREVIEW · BASIC</div>'
            f'<div class="pv-head"><img src="{esc(p["avatar"])}" alt="avatar"/>'
            f'<div class="pv-title"><a href="{esc(p["html_url"])}" target="_blank"><span>{esc(p["owner"])} /</span> {esc(p["name"])}</a></div></div>'
            f'<div class="pv-desc">{esc(p["note"])}</div></div>'
        )
    bar, legend = "", ""
    for i, (lang, pct) in enumerate(sorted(p["languages"].items(), key=lambda x: -x[1])[:6]):
        color = LANG_COLORS.get(lang, FALLBACK[i % len(FALLBACK)])
        bar += f'<div style="width:{pct}%;background:{color}"></div>'
        legend += f'<span><i class="dot" style="background:{color}"></i>{esc(lang)} {pct}%</span>'
    chips = f'<span class="chip">⭐ {fmt_num(p["stars"])}</span><span class="chip">🍴 {fmt_num(p["forks"])}</span>'
    chips += f'<span class="chip">🐞 {fmt_num(p["open_issues"])} issues</span>'
    if p.get("license"):
        chips += f'<span class="chip">⚖️ {esc(p["license"])}</span>'
    chips += f'<span class="chip">🌿 {esc(p["default_branch"])}</span>'
    chips += "".join(f'<span class="chip topic">#{esc(t)}</span>' for t in p["topics"][:6])
    files = "".join(f'<div>{"📁" if e["type"] == "dir" else "📄"} {esc(e["name"])}</div>' for e in p["root"][:24])
    desc = esc(p["description"]) if p.get("description") else "<i>No description provided.</i>"
    return (
        '<div class="glass"><div class="live"><i></i> LIVE PREVIEW</div>'
        f'<div class="pv-head"><img src="{esc(p["avatar"])}" alt="avatar"/>'
        f'<div class="pv-title"><a href="{esc(p["html_url"])}" target="_blank"><span>{esc(p["owner"])} /</span> {esc(p["name"])}</a></div></div>'
        f'<div class="pv-desc">{desc}</div><div class="chips">{chips}</div>'
        f'<div class="langbar">{bar}</div><div class="legend">{legend}</div>'
        f'<div class="tree-title">Project structure</div><div class="files">{files}</div></div>'
    )


def inventory_html(inv: dict) -> str:
    return '<div class="chips">' + "".join(
        f'<span class="chip">{INV_ICONS.get(k, "📦")} {v} {esc(k.lower())}</span>' for k, v in inv.items()) + "</div>"


def structure_html(st: dict) -> str:
    dirs = "".join(f'<div>📁 {esc(d["name"])} <em>{d["files"]} files · {esc(d["kind"])}</em></div>' for d in st.get("dirs", []))
    root = "".join(f'<div>📄 {esc(f)}</div>' for f in st.get("root_files", []))
    return f'<div class="files">{dirs}{root}</div>'


def _lang_bar(langs: dict) -> str:
    total = sum(langs.values()) or 1
    bar, legend = "", ""
    for i, (lang, n) in enumerate(list(langs.items())[:6]):
        pct = round(n * 100 / total, 1)
        color = LANG_COLORS.get(lang.split(" (")[0], FALLBACK[i % len(FALLBACK)])
        bar += f'<div style="width:{pct}%;background:{color}"></div>'
        legend += f'<span><i class="dot" style="background:{color}"></i>{esc(lang)} {pct}%</span>'
    return f'<div class="langbar">{bar}</div><div class="legend">{legend}</div>' if bar else ""


def upload_card(p: dict) -> str:
    f = p["facts"]
    techs = "".join(f'<span class="chip topic">{esc(t["name"])}</span>' for t in f["technologies"][:10])
    summary = f["readme"]["summary"]
    desc = esc(summary) if summary else "No README summary found — the AI will work it out from the files."
    return (
        '<div class="glass"><div class="live"><i></i> UPLOAD READY · LIVE PREVIEW</div>'
        f'<div class="pv-head"><div class="pv-icon">📦</div><div class="pv-title">{esc(p["name"])}</div></div>'
        f'<div class="pv-desc">{desc}</div>'
        f'<div class="chips"><span class="chip type">{esc(f["project_type"])}</span>'
        f'<span class="chip">{p["stats"]["total_files"]} files</span>{techs}</div>'
        f'{inventory_html(f["inventory"])}{_lang_bar(f["languages"])}'
        f'<div class="tree-title">Project structure</div>{structure_html(f["structure"])}</div>'
    )


def repo_header(name: str, facts: dict, stats: dict, icon: str = "📦") -> str:
    techs = "".join(f'<span class="chip topic">{esc(t["name"])}</span>' for t in facts["technologies"][:8])
    return (
        f'<div class="rhead"><div class="rh-title">{icon} {esc(name)}'
        f'<span class="chip type">{esc(facts["project_type"])}</span></div>'
        f'<div class="chips"><span class="chip">{stats.get("total_files", "?")} files</span>'
        f'{"".join(f"<span class=chip>{INV_ICONS.get(k, chr(128230))} {v} {esc(k.lower())}</span>" for k, v in list(facts["inventory"].items())[:5])}'
        f'{techs}</div></div>'
    )


def model_card(m: dict) -> str:
    return (f'<div class="mcard"><b>{m["icon"]} {esc(m["name"])}</b> <span>· {esc(m["vendor"])}'
            f'{(" · " + esc(m["size"])) if m.get("size") else ""}</span><br>{esc(m["blurb"])}</div>')


def console_html(s: dict) -> str:
    phase = s["phase"]
    label, color = PHASES[phase]
    if phase == "generating" and s["current_file"]:
        label = f"Analysing file {s['files_done'] + 1}/{s['files_total']}"
    elif phase == "generating" and s["mode"] == "deep":
        label = "Writing the final explanation"
    running = phase in ("clone", "extract", "loading", "reading", "generating")
    gen = phase == "generating"
    eq = f'<span class="eq{"" if gen else " off"}"><i></i><i></i><i></i><i></i><i></i></span>'

    fs = s["final_stats"]
    tps = fs["tps"] if (fs and phase == "done") else s["tps"]
    prompt_tok = fs["prompt_eval_count"] if fs else s["prompt_tokens"]
    tiles = (
        f'<div class="tile"><b>{s["elapsed"]:.1f}s</b><span>Elapsed</span></div>'
        f'<div class="tile"><b>{s["tokens"]}</b><span>Tokens generated</span></div>'
        f'<div class="tile"><b>{(f"{tps:.1f}" if tps else "—")}</b><span>Tokens / second</span></div>'
        f'<div class="tile"><b>{(f"{prompt_tok:,}" if prompt_tok else "—")}</b><span>Prompt tokens</span></div>'
    )

    prog = ""
    if s["mode"] == "deep" and s["files_total"]:
        pct = int(100 * s["files_done"] / max(s["files_total"], 1))
        prog = (f'<div class="plabel"><span>File-by-file reading</span><span>{s["files_done"]}/{s["files_total"]}</span></div>'
                f'<div class="pbar"><div style="width:{pct}%"></div></div>')
    cur = f'<div class="curfile">📄 reading {esc(s["current_file"])}</div>' if s["current_file"] else ""

    recent = list(s["recent"])
    toks = ""
    for i, tk in enumerate(recent):
        shown = tk.replace("\n", "↵") if tk.strip() or "\n" in tk else "␠"
        toks += f'<span class="tk{" new" if i == len(recent) - 1 and running else ""}">{esc(shown)}</span>'
    toks_html = f'<div class="toks">{toks}</div>' if recent else ""

    lines = "".join(f'<div><span class="ts">[{t:5.1f}s]</span> {esc(txt)}</div>' for t, txt in s["log"][-9:])
    if running:
        lines += '<div><span class="cur">▌</span></div>'
    term = f'<div class="term">{lines or "<div>waiting…</div>"}</div>'

    return (
        f'<div class="console"><div class="con-head">'
        f'<span class="model-badge">{s["icon"]} {esc(s["model_name"])} <small>{esc(s["vendor"])} · local</small></span>'
        f'<span class="state {"run" if running else ""}" style="--c:{color}"><i></i>{esc(label)}{eq if running else ""}</span></div>'
        f'<div class="tiles">{tiles}</div>{prog}{cur}{toks_html}{term}</div>'
    )


def files_html(s: dict) -> str:
    if not s["file_summaries"]:
        return ""
    cards = ""
    for f in s["file_summaries"]:
        live = not f["done"]
        meta = "" if live else f'<em>{f["tokens"]} tok{(" · " + str(f["seconds"]) + "s") if f["seconds"] else ""}</em>'
        body = esc(f["summary"]) + ('<span class="cur"> ▌</span>' if live else "")
        cards += (f'<div class="fcard{" cur" if live else ""}"><code>{esc(f["path"])}</code> {meta}'
                  f'<p>{body or "…"}</p></div>')
    return f'<div class="ftitle">What the model understood from each file</div><div class="fcards">{cards}</div>'


def tech_html(techs: list) -> str:
    out, last = "", None
    for t in techs:
        if t["category"] != last:
            out += f'</div><div class="cat">{esc(t["category"])}</div><div class="chips">' if last else f'<div class="cat">{esc(t["category"])}</div><div class="chips">'
            last = t["category"]
        out += f'<span class="chip">{esc(t["name"])}</span>'
    return out + "</div>" if out else ""


def verify_html(v: dict) -> str:
    chips = "".join(
        f'<span class="chip {"ok" if i["found"] else "warn"}">{"✅" if i["found"] else "⚠️"} {esc(i["name"])}</span>'
        for i in v["items"])
    return f'<div class="chips">{chips}</div>'
