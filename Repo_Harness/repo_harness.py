#!/usr/bin/env python3
"""
repo_harness.py — Understand a codebase thoroughly using OpenAI APIs.

Three-pass analysis:
  Pass 1 — Per-file:  purpose, key symbols, imports, complexity notes
  Pass 2 — Per-dir:   module role, internal API surface, coupling
  Pass 3 — Global:    architecture, data flows, patterns, entry points

Produces a self-contained HTML report with Mermaid dependency diagrams.

Usage:
  python repo_harness.py --path /path/to/repo --output report.html
  python repo_harness.py --path . --model gpt-4o-mini --no-cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pathspec
import tiktoken
from openai import OpenAI, RateLimitError, APIError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o"
MAX_FILE_TOKENS = 6_000        # tokens sent per file in Pass 1
MAX_CHUNK_TOKENS = 4_000       # fallback chunk size for oversized files
CACHE_DIR_NAME = ".repo_harness_cache"

SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt",
    ".cs", ".cpp", ".c", ".h", ".hpp", ".rb", ".php", ".swift", ".scala",
    ".sh", ".bash", ".zsh", ".ps1", ".lua", ".r", ".jl", ".ex", ".exs",
    ".clj", ".hs", ".ml", ".fs", ".dart", ".vue", ".svelte",
    ".json", ".yaml", ".yml", ".toml", ".env.example",
    ".md", ".rst", ".sql",
}

ALWAYS_SKIP = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "target", "vendor",
    ".mypy_cache", ".pytest_cache", ".tox", "coverage", ".nyc_output",
    "*.min.js", "*.min.css", "*.lock", "package-lock.json",
    "yarn.lock", "poetry.lock", "Cargo.lock", "Gemfile.lock",
}

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileSummary:
    path: str           # relative to repo root
    language: str
    purpose: str
    exports: list[str]
    imports: list[str]
    key_functions: list[str]
    complexity_notes: str
    token_count: int
    chunked: bool = False


@dataclass
class ModuleSummary:
    directory: str
    role: str
    files: list[str]
    public_api: list[str]
    dependencies: list[str]


@dataclass
class ArchitectureSummary:
    overview: str
    entry_points: list[str]
    layers: list[dict]          # [{name, description, modules}]
    data_flows: list[str]
    patterns: list[str]
    tech_stack: list[str]
    dependency_graph: dict      # {module: [dep, ...]} for Mermaid


@dataclass
class RepoAnalysis:
    repo_path: str
    model: str
    files: list[FileSummary] = field(default_factory=list)
    modules: list[ModuleSummary] = field(default_factory=list)
    architecture: ArchitectureSummary | None = None


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def load_gitignore(repo_root: Path) -> pathspec.PathSpec:
    patterns: list[str] = []
    for gitignore in repo_root.rglob(".gitignore"):
        try:
            patterns += gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            pass
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def should_skip(path: Path, repo_root: Path) -> bool:
    parts = set(path.relative_to(repo_root).parts)
    for skip in ALWAYS_SKIP:
        if skip.startswith("*."):
            if path.suffix == skip[1:]:
                return True
        elif skip in parts or path.name == skip:
            return True
    return False


def discover_files(
    repo_root: Path,
    extensions: set[str],
    gitignore: pathspec.PathSpec,
    max_files: int,
) -> list[Path]:
    files: list[Path] = []
    for p in sorted(repo_root.rglob("*")):
        if not p.is_file():
            continue
        if should_skip(p, repo_root):
            continue
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        if gitignore.match_file(rel):
            continue
        if p.suffix.lower() not in extensions:
            continue
        files.append(p)
        if len(files) >= max_files:
            console.print(f"[yellow]File cap reached ({max_files}). Use --max-files to increase.[/]")
            break
    return files


# ---------------------------------------------------------------------------
# Token counting + chunking
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text, disallowed_special=()))


def chunk_text(text: str, model: str, max_tokens: int) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        lt = count_tokens(line, model)
        if current_tokens + lt > max_tokens and current:
            chunks.append("".join(current))
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += lt
    if current:
        chunks.append("".join(current))
    return chunks or [""]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class Cache:
    def __init__(self, repo_root: Path, enabled: bool):
        self.enabled = enabled
        self.cache_dir = repo_root / CACHE_DIR_NAME
        if enabled:
            self.cache_dir.mkdir(exist_ok=True)

    def _key(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, content: str) -> dict | None:
        if not self.enabled:
            return None
        p = self.cache_dir / (self._key(content) + ".json")
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def put(self, content: str, value: dict) -> None:
        if not self.enabled:
            return
        p = self.cache_dir / (self._key(content) + ".json")
        p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# OpenAI wrapper with retry
# ---------------------------------------------------------------------------

def call_openai(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_retries: int = 4,
) -> str:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            console.print(f"[yellow]API error (retry {attempt+1}): {e}[/]")
            time.sleep(delay)
            delay *= 2
    return ""


def parse_json_response(text: str) -> dict:
    """Extract JSON from a response that may have markdown fences."""
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# Pass 1 — File analysis
# ---------------------------------------------------------------------------

FILE_SYSTEM_PROMPT = """You are a senior software engineer analyzing source code.
Return ONLY valid JSON — no prose, no markdown fences."""

FILE_USER_TEMPLATE = """Analyze this file and return JSON with exactly these keys:
{{
  "language": "<programming language>",
  "purpose": "<one sentence: what this file does>",
  "exports": ["<exported symbol>", ...],
  "imports": ["<imported module or package>", ...],
  "key_functions": ["<name: one-line description>", ...],
  "complexity_notes": "<any notable complexity, patterns, or gotchas; empty string if none>"
}}

File: {path}
```
{content}
```"""


def analyze_file(
    client: OpenAI,
    model: str,
    file_path: Path,
    repo_root: Path,
    cache: Cache,
) -> FileSummary:
    rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return FileSummary(
            path=rel, language="unknown", purpose=f"Could not read: {e}",
            exports=[], imports=[], key_functions=[], complexity_notes="",
            token_count=0,
        )

    token_count = count_tokens(content, model)
    chunked = token_count > MAX_FILE_TOKENS

    if chunked:
        chunks = chunk_text(content, model, MAX_CHUNK_TOKENS)
        per_chunk: list[dict] = []
        for i, chunk in enumerate(chunks):
            prompt = FILE_USER_TEMPLATE.format(
                path=f"{rel} (chunk {i+1}/{len(chunks)})", content=chunk
            )
            cached = cache.get(prompt)
            if cached:
                per_chunk.append(cached)
                continue
            raw = call_openai(client, model, FILE_SYSTEM_PROMPT, prompt)
            parsed = parse_json_response(raw)
            cache.put(prompt, parsed)
            per_chunk.append(parsed)

        # Merge chunks
        merged: dict[str, Any] = {
            "language": per_chunk[0].get("language", "unknown"),
            "purpose": per_chunk[0].get("purpose", ""),
            "exports": [],
            "imports": [],
            "key_functions": [],
            "complexity_notes": "",
        }
        for c in per_chunk:
            merged["exports"] += c.get("exports", [])
            merged["imports"] += c.get("imports", [])
            merged["key_functions"] += c.get("key_functions", [])
            if c.get("complexity_notes"):
                merged["complexity_notes"] += (" " + c["complexity_notes"]).strip()
        # Deduplicate
        merged["exports"] = list(dict.fromkeys(merged["exports"]))
        merged["imports"] = list(dict.fromkeys(merged["imports"]))
        data = merged
    else:
        prompt = FILE_USER_TEMPLATE.format(path=rel, content=content)
        cached = cache.get(prompt)
        if cached:
            data = cached
        else:
            raw = call_openai(client, model, FILE_SYSTEM_PROMPT, prompt)
            data = parse_json_response(raw)
            cache.put(prompt, data)

    return FileSummary(
        path=rel,
        language=data.get("language", "unknown"),
        purpose=data.get("purpose", ""),
        exports=data.get("exports", []),
        imports=data.get("imports", []),
        key_functions=data.get("key_functions", []),
        complexity_notes=data.get("complexity_notes", ""),
        token_count=token_count,
        chunked=chunked,
    )


# ---------------------------------------------------------------------------
# Pass 2 — Module / directory synthesis
# ---------------------------------------------------------------------------

MODULE_SYSTEM_PROMPT = """You are a software architect synthesizing module-level understanding.
Return ONLY valid JSON — no prose, no markdown fences."""

MODULE_USER_TEMPLATE = """Given these file summaries for directory '{directory}', return JSON:
{{
  "role": "<one sentence: purpose of this module/directory>",
  "public_api": ["<symbol or concept exposed to the rest of the codebase>", ...],
  "dependencies": ["<other module or directory this depends on>", ...]
}}

Files:
{files_json}"""


def synthesize_modules(
    client: OpenAI,
    model: str,
    file_summaries: list[FileSummary],
    cache: Cache,
) -> list[ModuleSummary]:
    # Group files by top-level directory (or "." for root files)
    groups: dict[str, list[FileSummary]] = {}
    for fs in file_summaries:
        parts = Path(fs.path).parts
        top = parts[0] if len(parts) > 1 else "."
        groups.setdefault(top, []).append(fs)

    modules: list[ModuleSummary] = []
    for directory, summaries in sorted(groups.items()):
        files_json = json.dumps(
            [{"path": s.path, "purpose": s.purpose, "exports": s.exports,
              "imports": s.imports} for s in summaries],
            indent=2
        )
        prompt = MODULE_USER_TEMPLATE.format(directory=directory, files_json=files_json)
        cached = cache.get(prompt)
        if cached:
            data = cached
        else:
            raw = call_openai(client, model, MODULE_SYSTEM_PROMPT, prompt)
            data = parse_json_response(raw)
            cache.put(prompt, data)

        modules.append(ModuleSummary(
            directory=directory,
            role=data.get("role", ""),
            files=[s.path for s in summaries],
            public_api=data.get("public_api", []),
            dependencies=data.get("dependencies", []),
        ))
    return modules


# ---------------------------------------------------------------------------
# Pass 3 — Global architecture
# ---------------------------------------------------------------------------

ARCH_SYSTEM_PROMPT = """You are a senior software architect producing a comprehensive
architectural overview of an entire codebase. Return ONLY valid JSON."""

ARCH_USER_TEMPLATE = """Given these module summaries, produce a global architectural analysis.
Return JSON with exactly these keys:
{{
  "overview": "<2-4 sentence architectural overview of the whole system>",
  "entry_points": ["<file or module that is the main entry point>", ...],
  "layers": [
    {{"name": "<layer name>", "description": "<what this layer does>", "modules": ["<dir>", ...]}},
    ...
  ],
  "data_flows": ["<describe a key data flow in one sentence>", ...],
  "patterns": ["<design pattern or architectural pattern observed>", ...],
  "tech_stack": ["<technology, framework, or library>", ...],
  "dependency_graph": {{
    "<module>": ["<depends-on-module>", ...]
  }}
}}

Modules:
{modules_json}"""


def synthesize_architecture(
    client: OpenAI,
    model: str,
    modules: list[ModuleSummary],
    cache: Cache,
) -> ArchitectureSummary:
    modules_json = json.dumps(
        [{"directory": m.directory, "role": m.role, "public_api": m.public_api,
          "dependencies": m.dependencies} for m in modules],
        indent=2
    )
    prompt = ARCH_USER_TEMPLATE.format(modules_json=modules_json)
    cached = cache.get(prompt)
    if cached:
        data = cached
    else:
        raw = call_openai(client, model, ARCH_SYSTEM_PROMPT, prompt)
        data = parse_json_response(raw)
        cache.put(prompt, data)

    return ArchitectureSummary(
        overview=data.get("overview", ""),
        entry_points=data.get("entry_points", []),
        layers=data.get("layers", []),
        data_flows=data.get("data_flows", []),
        patterns=data.get("patterns", []),
        tech_stack=data.get("tech_stack", []),
        dependency_graph=data.get("dependency_graph", {}),
    )


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def build_mermaid_graph(dependency_graph: dict) -> str:
    if not dependency_graph:
        return "graph LR\n    A[No dependency data]"
    lines = ["graph LR"]
    node_ids: dict[str, str] = {}

    def node_id(name: str) -> str:
        if name not in node_ids:
            safe = re.sub(r"[^a-zA-Z0-9]", "_", name)
            node_ids[name] = f"n{len(node_ids)}_{safe}"
        return node_ids[name]

    all_nodes: set[str] = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        all_nodes.update(deps)

    for node in sorted(all_nodes):
        nid = node_id(node)
        label = node.replace('"', "'")
        lines.append(f'    {nid}["{label}"]')

    for source, deps in sorted(dependency_graph.items()):
        for dep in deps:
            lines.append(f"    {node_id(source)} --> {node_id(dep)}")

    return "\n".join(lines)


def escape_html(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def render_list(items: list[str], cls: str = "") -> str:
    if not items:
        return "<em>none</em>"
    class_attr = f' class="{cls}"' if cls else ""
    lis = "".join(f"<li>{escape_html(i)}</li>" for i in items)
    return f"<ul{class_attr}>{lis}</ul>"


def build_html_report(analysis: RepoAnalysis) -> str:
    arch = analysis.architecture
    mermaid_src = build_mermaid_graph(arch.dependency_graph if arch else {})

    # File table rows
    file_rows = ""
    for fs in sorted(analysis.files, key=lambda f: f.path):
        chunked_badge = ' <span class="badge">chunked</span>' if fs.chunked else ""
        purpose = escape_html(fs.purpose)
        lang = escape_html(fs.language)
        tokens = f"{fs.token_count:,}"
        file_rows += f"""
        <tr>
          <td class="file-path">{escape_html(fs.path)}{chunked_badge}</td>
          <td><span class="lang-badge">{lang}</span></td>
          <td class="purpose-cell">{purpose}</td>
          <td class="token-count">{tokens}</td>
        </tr>"""

    # Module cards
    module_cards = ""
    for mod in sorted(analysis.modules, key=lambda m: m.directory):
        file_list = "".join(
            f'<li class="file-item">{escape_html(f)}</li>'
            for f in mod.files
        )
        api_list = render_list(mod.public_api, "api-list")
        dep_list = render_list(mod.dependencies, "dep-list")
        module_cards += f"""
        <div class="module-card">
          <h3 class="module-title">{escape_html(mod.directory)}</h3>
          <p class="module-role">{escape_html(mod.role)}</p>
          <div class="module-meta">
            <div class="meta-section">
              <h4>Public API</h4>{api_list}
            </div>
            <div class="meta-section">
              <h4>Dependencies</h4>{dep_list}
            </div>
          </div>
          <details class="file-list-details">
            <summary>{len(mod.files)} file(s)</summary>
            <ul class="file-list">{file_list}</ul>
          </details>
        </div>"""

    # Architecture sections
    arch_overview = escape_html(arch.overview) if arch else ""
    entry_points_html = render_list(arch.entry_points if arch else [])
    patterns_html = render_list(arch.patterns if arch else [])
    stack_html = render_list(arch.tech_stack if arch else [])

    data_flows_html = ""
    if arch and arch.data_flows:
        data_flows_html = "".join(
            f'<li class="flow-item">{escape_html(f)}</li>'
            for f in arch.data_flows
        )
        data_flows_html = f'<ol class="flow-list">{data_flows_html}</ol>'

    layers_html = ""
    if arch and arch.layers:
        for layer in arch.layers:
            name = escape_html(layer.get("name", ""))
            desc = escape_html(layer.get("description", ""))
            mods = render_list(layer.get("modules", []))
            layers_html += f"""
            <div class="layer-card">
              <span class="layer-name">{name}</span>
              <p class="layer-desc">{desc}</p>
              <div class="layer-mods">{mods}</div>
            </div>"""

    total_files = len(analysis.files)
    total_tokens = sum(f.token_count for f in analysis.files)
    languages = sorted({f.language for f in analysis.files if f.language != "unknown"})
    lang_badges = "".join(f'<span class="lang-badge">{escape_html(l)}</span>' for l in languages)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Codebase Analysis — {escape_html(analysis.repo_path)}</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    :root {{
      --bg: #0f1117;
      --surface: #1a1d27;
      --surface2: #252836;
      --border: #2e3148;
      --accent: #6c8aff;
      --accent2: #a78bfa;
      --green: #4ade80;
      --yellow: #fbbf24;
      --red: #f87171;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --font: 'Inter', system-ui, sans-serif;
      --mono: 'JetBrains Mono', 'Fira Code', monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: var(--font);
            font-size: 14px; line-height: 1.6; }}
    a {{ color: var(--accent); text-decoration: none; }}
    h1, h2, h3, h4 {{ font-weight: 600; }}

    /* Layout */
    .sidebar {{
      position: fixed; top: 0; left: 0; width: 220px; height: 100vh;
      background: var(--surface); border-right: 1px solid var(--border);
      padding: 24px 0; overflow-y: auto; z-index: 100;
    }}
    .sidebar-logo {{ padding: 0 20px 16px; font-size: 13px; font-weight: 700;
                     color: var(--accent); letter-spacing: 0.05em; text-transform: uppercase; }}
    .sidebar nav a {{
      display: block; padding: 8px 20px; color: var(--text-muted); font-size: 13px;
      border-left: 2px solid transparent; transition: all 0.15s;
    }}
    .sidebar nav a:hover {{ color: var(--text); border-left-color: var(--accent);
                            background: var(--surface2); }}
    .main {{ margin-left: 220px; padding: 48px 48px 80px; max-width: 1200px; }}

    /* Header */
    .page-header {{ margin-bottom: 40px; }}
    .page-header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
    .page-header .repo-path {{ color: var(--text-muted); font-family: var(--mono);
                                font-size: 12px; }}
    .stats-row {{ display: flex; gap: 24px; margin-top: 20px; flex-wrap: wrap; }}
    .stat-card {{ background: var(--surface); border: 1px solid var(--border);
                  border-radius: 8px; padding: 16px 20px; min-width: 120px; }}
    .stat-card .stat-value {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
    .stat-card .stat-label {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}

    /* Section */
    .section {{ margin-bottom: 56px; }}
    .section-title {{ font-size: 20px; font-weight: 700; margin-bottom: 20px;
                      padding-bottom: 10px; border-bottom: 1px solid var(--border);
                      display: flex; align-items: center; gap: 10px; }}
    .section-title .icon {{ font-size: 18px; }}

    /* Architecture */
    .arch-overview {{ background: var(--surface); border: 1px solid var(--border);
                      border-radius: 8px; padding: 20px; margin-bottom: 24px;
                      line-height: 1.8; }}
    .arch-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                  gap: 16px; margin-bottom: 24px; }}
    .arch-card {{ background: var(--surface); border: 1px solid var(--border);
                  border-radius: 8px; padding: 16px 20px; }}
    .arch-card h4 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
                     color: var(--text-muted); margin-bottom: 10px; }}

    /* Layers */
    .layers-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
    .layer-card {{ background: var(--surface2); border: 1px solid var(--border);
                   border-radius: 8px; padding: 14px 16px; flex: 1; min-width: 180px; }}
    .layer-name {{ display: inline-block; background: var(--accent); color: #fff;
                   font-size: 11px; font-weight: 700; padding: 2px 8px;
                   border-radius: 4px; margin-bottom: 8px; letter-spacing: 0.04em; }}
    .layer-desc {{ font-size: 13px; color: var(--text-muted); margin-bottom: 8px; }}
    .layer-mods ul {{ list-style: none; }}
    .layer-mods li {{ font-family: var(--mono); font-size: 12px; color: var(--accent2); }}

    /* Mermaid */
    .diagram-container {{ background: var(--surface); border: 1px solid var(--border);
                          border-radius: 8px; padding: 24px; overflow: auto; }}
    .mermaid {{ text-align: center; }}

    /* Module cards */
    .modules-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
                     gap: 16px; }}
    .module-card {{ background: var(--surface); border: 1px solid var(--border);
                    border-radius: 8px; padding: 18px 20px; }}
    .module-title {{ font-family: var(--mono); font-size: 14px; color: var(--accent);
                     margin-bottom: 6px; }}
    .module-role {{ font-size: 13px; color: var(--text-muted); margin-bottom: 14px;
                    line-height: 1.6; }}
    .module-meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
                    margin-bottom: 12px; }}
    .meta-section h4 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
                        color: var(--text-muted); margin-bottom: 6px; }}
    .api-list li, .dep-list li {{ font-family: var(--mono); font-size: 11px;
                                   color: var(--green); }}
    .dep-list li {{ color: var(--yellow); }}

    /* File table */
    .file-table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    thead th {{ text-align: left; font-size: 11px; text-transform: uppercase;
                letter-spacing: 0.08em; color: var(--text-muted); padding: 10px 12px;
                border-bottom: 1px solid var(--border); position: sticky; top: 0;
                background: var(--bg); }}
    tbody tr {{ border-bottom: 1px solid var(--border); }}
    tbody tr:hover {{ background: var(--surface); }}
    td {{ padding: 10px 12px; vertical-align: top; }}
    .file-path {{ font-family: var(--mono); font-size: 12px; color: var(--accent2);
                  white-space: nowrap; }}
    .purpose-cell {{ font-size: 13px; color: var(--text-muted); }}
    .token-count {{ font-family: var(--mono); font-size: 12px; color: var(--text-muted);
                    text-align: right; white-space: nowrap; }}

    /* Misc */
    .lang-badge {{ background: var(--surface2); border: 1px solid var(--border);
                   border-radius: 4px; padding: 2px 8px; font-size: 11px;
                   font-family: var(--mono); color: var(--accent); white-space: nowrap; }}
    .badge {{ background: var(--yellow); color: #000; border-radius: 4px;
              font-size: 10px; padding: 1px 5px; font-weight: 600; }}
    .lang-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}

    .file-list-details summary {{ cursor: pointer; font-size: 12px; color: var(--text-muted);
                                   user-select: none; }}
    .file-list {{ list-style: none; margin-top: 8px; }}
    .file-item {{ font-family: var(--mono); font-size: 11px; color: var(--text-muted);
                  padding: 2px 0; }}
    .flow-list {{ padding-left: 20px; }}
    .flow-item {{ margin-bottom: 8px; font-size: 13px; color: var(--text); }}

    ul {{ padding-left: 16px; }}
    ul li {{ margin-bottom: 4px; font-size: 13px; }}

    @media (max-width: 900px) {{
      .sidebar {{ display: none; }}
      .main {{ margin-left: 0; padding: 24px; }}
    }}
  </style>
</head>
<body>
  <div class="sidebar">
    <div class="sidebar-logo">Repo Harness</div>
    <nav>
      <a href="#overview">Overview</a>
      <a href="#architecture">Architecture</a>
      <a href="#dependency-graph">Dependency Graph</a>
      <a href="#modules">Modules</a>
      <a href="#files">All Files</a>
    </nav>
  </div>

  <div class="main">
    <div class="page-header" id="overview">
      <h1>Codebase Analysis</h1>
      <div class="repo-path">{escape_html(analysis.repo_path)}</div>
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-value">{total_files}</div>
          <div class="stat-label">Files Analyzed</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{len(analysis.modules)}</div>
          <div class="stat-label">Modules</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{total_tokens:,}</div>
          <div class="stat-label">Total Tokens</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{escape_html(analysis.model)}</div>
          <div class="stat-label">Model</div>
        </div>
      </div>
      <div class="lang-row" style="margin-top: 16px;">{lang_badges}</div>
    </div>

    <!-- Architecture -->
    <div class="section" id="architecture">
      <h2 class="section-title"><span class="icon">🏛</span> Architecture</h2>
      <div class="arch-overview">{arch_overview}</div>

      <div class="arch-grid">
        <div class="arch-card">
          <h4>Entry Points</h4>
          {entry_points_html}
        </div>
        <div class="arch-card">
          <h4>Patterns</h4>
          {patterns_html}
        </div>
        <div class="arch-card">
          <h4>Tech Stack</h4>
          {stack_html}
        </div>
      </div>

      {'<h3 style="margin-bottom:14px;font-size:16px;">Layers</h3><div class="layers-row">' + layers_html + '</div>' if layers_html else ''}

      {'<h3 style="margin-bottom:14px;font-size:16px;">Data Flows</h3>' + data_flows_html if data_flows_html else ''}
    </div>

    <!-- Dependency Graph -->
    <div class="section" id="dependency-graph">
      <h2 class="section-title"><span class="icon">🔗</span> Dependency Graph</h2>
      <div class="diagram-container">
        <div class="mermaid">{escape_html(mermaid_src)}</div>
      </div>
    </div>

    <!-- Modules -->
    <div class="section" id="modules">
      <h2 class="section-title"><span class="icon">📦</span> Modules</h2>
      <div class="modules-grid">{module_cards}</div>
    </div>

    <!-- Files -->
    <div class="section" id="files">
      <h2 class="section-title"><span class="icon">📄</span> All Files</h2>
      <div class="file-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Path</th><th>Language</th><th>Purpose</th><th>Tokens</th>
            </tr>
          </thead>
          <tbody>{file_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    mermaid.initialize({{ startOnLoad: true, theme: 'dark',
      themeVariables: {{ darkMode: true, background: '#1a1d27', primaryColor: '#6c8aff' }}
    }});
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    repo_root = Path(args.path).resolve()
    if not repo_root.is_dir():
        console.print(f"[red]Error: {repo_root} is not a directory[/]")
        sys.exit(1)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]Error: provide --api-key or set OPENAI_API_KEY[/]")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    cache = Cache(repo_root, enabled=not args.no_cache)
    extensions = SOURCE_EXTENSIONS if not args.extensions else {
        e if e.startswith(".") else f".{e}" for e in args.extensions.split(",")
    }

    console.rule("[bold blue]Repo Harness[/]")
    console.print(f"  Repo:   [cyan]{repo_root}[/]")
    console.print(f"  Model:  [cyan]{args.model}[/]")
    console.print(f"  Output: [cyan]{args.output}[/]")
    console.print(f"  Cache:  [cyan]{'disabled' if args.no_cache else str(repo_root / CACHE_DIR_NAME)}[/]\n")

    # Discover files
    gitignore = load_gitignore(repo_root)
    files = discover_files(repo_root, extensions, gitignore, args.max_files)
    console.print(f"[green]Found {len(files)} source files[/]\n")

    analysis = RepoAnalysis(repo_path=str(repo_root), model=args.model)

    # Pass 1 — file analysis
    console.rule("Pass 1 — File Analysis")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing files...", total=len(files))
        for f in files:
            rel = str(f.relative_to(repo_root)).replace("\\", "/")
            progress.update(task, description=f"[dim]{rel[:60]}[/]")
            summary = analyze_file(client, args.model, f, repo_root, cache)
            analysis.files.append(summary)
            progress.advance(task)

    # Pass 2 — module synthesis
    console.rule("Pass 2 — Module Synthesis")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Synthesizing modules...")
        analysis.modules = synthesize_modules(client, args.model, analysis.files, cache)
    console.print(f"[green]Synthesized {len(analysis.modules)} module(s)[/]\n")

    # Pass 3 — architecture
    console.rule("Pass 3 — Architecture")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Building architecture overview...")
        analysis.architecture = synthesize_architecture(client, args.model, analysis.modules, cache)
    console.print("[green]Architecture synthesis complete[/]\n")

    # Generate report
    console.rule("Generating Report")
    html = build_html_report(analysis)
    output_path = Path(args.output)
    output_path.write_text(html, encoding="utf-8")
    console.print(f"[bold green]Report written to {output_path}[/]")

    # Also save raw JSON
    if args.save_json:
        json_path = output_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(asdict(analysis), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        console.print(f"[green]JSON data written to {json_path}[/]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a source code repository using OpenAI and produce an HTML report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--path", default=".", help="Path to the repository root (default: .)")
    parser.add_argument("--output", default="repo_analysis.html", help="Output HTML file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default="", help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--extensions", default="",
                        help="Comma-separated file extensions to include, e.g. py,ts,go")
    parser.add_argument("--max-files", type=int, default=500,
                        help="Maximum number of files to analyze (default: 500)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching of API responses")
    parser.add_argument("--save-json", action="store_true",
                        help="Also save raw JSON alongside the HTML report")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
