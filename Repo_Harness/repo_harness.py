#!/usr/bin/env python3
"""
repo_harness.py — Deep codebase analysis with DeepWiki-style HTML wiki output.

Eight analysis passes:
  Pass 1 — Per-file:          purpose, exports, imports, key functions
  Pass 2 — Module synthesis:  role, public API, inter-module coupling
  Pass 3 — Architecture:      overview, layers, entry points, tech stack
  Pass 4 — Components:        class/component hierarchy  (classDiagram)
  Pass 5 — API resources:     endpoint & route tree      (graph LR)
  Pass 6 — Type system:       models, schemas, types     (classDiagram)
  Pass 7 — Request lifecycle: request → response flow    (sequenceDiagram)
  Pass 8 — Data flow:         data transformation paths  (flowchart TD)

Output: self-contained HTML wiki with all Mermaid diagrams rendered inline.

Usage:
  python repo_harness.py --path /path/to/repo --output wiki.html
  python repo_harness.py --path . --provider gemini --model gemini-2.0-flash
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
from openai import OpenAI, RateLimitError, APIError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"

DEFAULT_MODELS = {
    PROVIDER_OPENAI: "gpt-4o",
    PROVIDER_GEMINI: "gemini-2.0-flash",
}

MAX_FILE_TOKENS  = 6_000
MAX_CHUNK_TOKENS = 4_000
DIAGRAM_MAX_CHARS = 22_000   # compact context ceiling for diagram passes
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
    path: str
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
    layers: list[dict]
    data_flows: list[str]
    patterns: list[str]
    tech_stack: list[str]
    dependency_graph: dict


@dataclass
class DiagramResult:
    mermaid: str
    notes: str
    applicable: bool = True


@dataclass
class RepoAnalysis:
    repo_path: str
    model: str
    files: list[FileSummary] = field(default_factory=list)
    modules: list[ModuleSummary] = field(default_factory=list)
    architecture: ArchitectureSummary | None = None
    component_hierarchy: DiagramResult | None = None
    api_resources: DiagramResult | None = None
    type_system: DiagramResult | None = None
    request_lifecycle: DiagramResult | None = None
    data_flow: DiagramResult | None = None


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

def count_tokens(text: str, model: str = "") -> int:
    # ~4 characters per token is a reliable approximation for chunking
    return len(text) // 4


def chunk_text(text: str, model: str, max_tokens: int) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        lt = count_tokens(line)
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
# Provider-agnostic LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str, max_retries: int = 4):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries

        if provider == PROVIDER_OPENAI:
            self._openai = OpenAI(api_key=api_key)
            self._gemini = None
        elif provider == PROVIDER_GEMINI:
            if not HAS_GOOGLE_GENAI:
                console.print("[red]google-genai is not installed. Run: pip install google-genai[/]")
                sys.exit(1)
            self._openai = None
            self._gemini = google_genai.Client(api_key=api_key)
        else:
            console.print(f"[red]Unknown provider: {provider!r}. Choose 'openai' or 'gemini'.[/]")
            sys.exit(1)

    def call(self, system: str, user: str, temperature: float = 0.2) -> str:
        if self.provider == PROVIDER_OPENAI:
            return self._call_openai(system, user, temperature)
        return self._call_gemini(system, user, temperature)

    def _call_openai(self, system: str, user: str, temperature: float) -> str:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._openai.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return resp.choices[0].message.content or ""
            except RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
            except APIError as e:
                if attempt == self.max_retries - 1:
                    raise
                console.print(f"[yellow]OpenAI API error (retry {attempt+1}): {e}[/]")
                time.sleep(delay)
                delay *= 2
        return ""

    def _call_gemini(self, system: str, user: str, temperature: float) -> str:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._gemini.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=google_genai_types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                    ),
                )
                return resp.text or ""
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                console.print(f"[yellow]Gemini API error (retry {attempt+1}): {e}[/]")
                time.sleep(delay)
                delay *= 2
        return ""


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# Compact context builder (used by diagram passes)
# ---------------------------------------------------------------------------

def compact_context(
    files: list[FileSummary],
    modules: list[ModuleSummary],
    arch: ArchitectureSummary | None,
    hint_keywords: list[str] | None = None,
    max_chars: int = DIAGRAM_MAX_CHARS,
) -> str:
    """Build a size-bounded JSON context string for diagram generation prompts.

    hint_keywords, if provided, causes matching files to be ranked first so
    the most relevant files survive truncation.
    """
    if hint_keywords:
        def score(f: FileSummary) -> int:
            text = f"{f.path} {f.purpose}".lower()
            return sum(1 for k in hint_keywords if k in text)
        ranked = sorted(files, key=score, reverse=True)
    else:
        ranked = list(files)

    data: dict[str, Any] = {
        "overview": arch.overview if arch else "",
        "tech_stack": arch.tech_stack if arch else [],
        "patterns": arch.patterns if arch else [],
        "entry_points": arch.entry_points if arch else [],
        "layers": arch.layers if arch else [],
        "modules": [
            {
                "dir": m.directory,
                "role": m.role,
                "api": m.public_api[:8],
                "deps": m.dependencies[:8],
            }
            for m in modules
        ],
        "files": [
            {
                "path": f.path,
                "lang": f.language,
                "purpose": f.purpose,
                "exports": f.exports[:5],
                "imports": f.imports[:5],
                "fns": f.key_functions[:4],
            }
            for f in ranked
        ],
    }

    result = json.dumps(data, indent=2)
    # Trim file list if over budget
    while len(result) > max_chars and data["files"]:
        data["files"] = data["files"][: max(5, len(data["files"]) // 2)]
        result = json.dumps(data, indent=2)
    return result


# ---------------------------------------------------------------------------
# Pass 1 — Per-file analysis
# ---------------------------------------------------------------------------

_FILE_SYSTEM = """You are a senior software engineer analyzing source code.
Return ONLY valid JSON — no prose, no markdown fences."""

_FILE_USER = """Analyze this file and return JSON with exactly these keys:
{{
  "language": "<programming language>",
  "purpose": "<one sentence: what this file does>",
  "exports": ["<exported symbol>", ...],
  "imports": ["<imported module or package>", ...],
  "key_functions": ["<name: one-line description>", ...],
  "complexity_notes": "<notable complexity or gotchas; empty string if none>"
}}

File: {path}
```
{content}
```"""


def analyze_file(
    llm: LLMClient,
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

    token_count = count_tokens(content)
    chunked = token_count > MAX_FILE_TOKENS

    if chunked:
        chunks = chunk_text(content, llm.model, MAX_CHUNK_TOKENS)
        per_chunk: list[dict] = []
        for i, chunk in enumerate(chunks):
            prompt = _FILE_USER.format(path=f"{rel} (chunk {i+1}/{len(chunks)})", content=chunk)
            cached = cache.get(prompt)
            if cached:
                per_chunk.append(cached)
                continue
            raw = llm.call(_FILE_SYSTEM, prompt)
            parsed = parse_json_response(raw)
            cache.put(prompt, parsed)
            per_chunk.append(parsed)

        merged: dict[str, Any] = {
            "language": per_chunk[0].get("language", "unknown"),
            "purpose": per_chunk[0].get("purpose", ""),
            "exports": [], "imports": [], "key_functions": [], "complexity_notes": "",
        }
        for c in per_chunk:
            merged["exports"] += c.get("exports", [])
            merged["imports"] += c.get("imports", [])
            merged["key_functions"] += c.get("key_functions", [])
            if c.get("complexity_notes"):
                merged["complexity_notes"] += (" " + c["complexity_notes"]).strip()
        merged["exports"] = list(dict.fromkeys(merged["exports"]))
        merged["imports"] = list(dict.fromkeys(merged["imports"]))
        data = merged
    else:
        prompt = _FILE_USER.format(path=rel, content=content)
        cached = cache.get(prompt)
        if cached:
            data = cached
        else:
            raw = llm.call(_FILE_SYSTEM, prompt)
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
# Pass 2 — Module synthesis
# ---------------------------------------------------------------------------

_MODULE_SYSTEM = """You are a software architect synthesizing module-level understanding.
Return ONLY valid JSON — no prose, no markdown fences."""

_MODULE_USER = """Given these file summaries for directory '{directory}', return JSON:
{{
  "role": "<one sentence: purpose of this module/directory>",
  "public_api": ["<symbol or concept exposed to the rest of the codebase>", ...],
  "dependencies": ["<other module or directory this depends on>", ...]
}}

Files:
{files_json}"""


def synthesize_modules(
    llm: LLMClient,
    file_summaries: list[FileSummary],
    cache: Cache,
) -> list[ModuleSummary]:
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
            indent=2,
        )
        prompt = _MODULE_USER.format(directory=directory, files_json=files_json)
        cached = cache.get(prompt)
        if cached:
            data = cached
        else:
            raw = llm.call(_MODULE_SYSTEM, prompt)
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

_ARCH_SYSTEM = """You are a senior software architect producing a comprehensive
architectural overview of an entire codebase. Return ONLY valid JSON."""

_ARCH_USER = """Given these module summaries, produce a global architectural analysis.
Return JSON with exactly these keys:
{{
  "overview": "<2-4 sentence architectural overview>",
  "entry_points": ["<file or module that is the main entry point>", ...],
  "layers": [
    {{"name": "<layer>", "description": "<what it does>", "modules": ["<dir>", ...]}},
    ...
  ],
  "data_flows": ["<key data flow in one sentence>", ...],
  "patterns": ["<design or architectural pattern observed>", ...],
  "tech_stack": ["<technology, framework, or library>", ...],
  "dependency_graph": {{
    "<module>": ["<depends-on-module>", ...]
  }}
}}

Modules:
{modules_json}"""


def synthesize_architecture(
    llm: LLMClient,
    modules: list[ModuleSummary],
    cache: Cache,
) -> ArchitectureSummary:
    modules_json = json.dumps(
        [{"directory": m.directory, "role": m.role, "public_api": m.public_api,
          "dependencies": m.dependencies} for m in modules],
        indent=2,
    )
    prompt = _ARCH_USER.format(modules_json=modules_json)
    cached = cache.get(prompt)
    if cached:
        data = cached
    else:
        raw = llm.call(_ARCH_SYSTEM, prompt)
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
# Diagram generation passes (4–8)
# ---------------------------------------------------------------------------

_DIAGRAM_SYSTEM = """You are a software architect producing Mermaid diagrams from codebase analysis.
Return ONLY valid JSON — no prose, no markdown fences outside the JSON.
The "mermaid" field must contain raw Mermaid source with NO surrounding code fences.
Keep diagrams focused: max 25–30 nodes. Use real names from the codebase."""

_FALLBACK_MERMAID = "graph TD\n    A[Not detected in this codebase]"


def _diagram_call(
    llm: LLMClient,
    cache: Cache,
    user_prompt: str,
    fallback_notes: str,
) -> DiagramResult:
    cached = cache.get(user_prompt)
    if cached:
        data = cached
    else:
        raw = llm.call(_DIAGRAM_SYSTEM, user_prompt)
        data = parse_json_response(raw)
        cache.put(user_prompt, data)

    if not data:
        return DiagramResult(mermaid=_FALLBACK_MERMAID, notes=fallback_notes, applicable=False)

    mermaid = data.get("mermaid", "").strip()
    notes   = data.get("notes", "")
    applicable = bool(data.get("applicable", True))

    if not mermaid or not applicable:
        return DiagramResult(mermaid=_FALLBACK_MERMAID, notes=notes or fallback_notes, applicable=False)

    # Strip any accidental fences the model snuck in
    mermaid = re.sub(r"^```[a-z]*\s*", "", mermaid, flags=re.MULTILINE)
    mermaid = re.sub(r"\s*```$", "", mermaid, flags=re.MULTILINE).strip()
    return DiagramResult(mermaid=mermaid, notes=notes, applicable=True)


# Pass 4 — Component / class hierarchy

_COMPONENT_USER = """Produce a Mermaid classDiagram showing the component or class hierarchy of this codebase.

Rules:
- Use --|> for inheritance, --> for association, o-- for aggregation
- List 2–4 key methods per class using: +returnType methodName(params)
- If the codebase is functional (no OOP classes), show service/module groupings instead
- Max 25 nodes

Return JSON:
{{
  "mermaid": "classDiagram\\n    ...",
  "notes": "<2-3 sentences explaining the hierarchy>",
  "applicable": true
}}

Codebase:
{context}"""


def generate_component_hierarchy(
    llm: LLMClient,
    files: list[FileSummary],
    modules: list[ModuleSummary],
    arch: ArchitectureSummary | None,
    cache: Cache,
) -> DiagramResult:
    ctx = compact_context(files, modules, arch)
    prompt = _COMPONENT_USER.format(context=ctx)
    return _diagram_call(llm, cache, prompt, "No clear component hierarchy detected.")


# Pass 5 — API resource organization

_API_USER = """Produce a Mermaid graph showing how API resources and routes are organized.

Rules:
- Use graph LR
- Use subgraph blocks to group related routes
- Label edges with HTTP methods: GET, POST, PUT, DELETE, PATCH
- Show auth/middleware annotations as notes where relevant
- Max 30 nodes
- Set applicable=false if this codebase has no HTTP API

Return JSON:
{{
  "mermaid": "graph LR\\n    ...",
  "notes": "<2-3 sentences explaining the API structure>",
  "applicable": true
}}

Codebase:
{context}"""


def generate_api_resources(
    llm: LLMClient,
    files: list[FileSummary],
    modules: list[ModuleSummary],
    arch: ArchitectureSummary | None,
    cache: Cache,
) -> DiagramResult:
    ctx = compact_context(
        files, modules, arch,
        hint_keywords=["route", "api", "controller", "handler", "endpoint", "router", "rest", "graphql"],
    )
    prompt = _API_USER.format(context=ctx)
    return _diagram_call(llm, cache, prompt, "No HTTP API detected in this codebase.")


# Pass 6 — Type system and validation

_TYPE_SYSTEM_USER = """Produce a Mermaid classDiagram showing the type system, data models, and validation.

Rules:
- Show all major types, interfaces, DTOs, schemas, and enums
- Include field names and types: +string email
- Show cardinality on relationships: "1" --> "*"
- Mark interfaces with <<interface>>, enums with <<enumeration>>, abstract with <<abstract>>
- Add validation constraints as notes where visible in code
- Max 20 types

Return JSON:
{{
  "mermaid": "classDiagram\\n    ...",
  "notes": "<2-3 sentences explaining the type system and validation approach>",
  "applicable": true
}}

Codebase:
{context}"""


def generate_type_system(
    llm: LLMClient,
    files: list[FileSummary],
    modules: list[ModuleSummary],
    arch: ArchitectureSummary | None,
    cache: Cache,
) -> DiagramResult:
    ctx = compact_context(
        files, modules, arch,
        hint_keywords=["model", "type", "schema", "interface", "entity", "dto", "struct", "enum", "validator"],
    )
    prompt = _TYPE_SYSTEM_USER.format(context=ctx)
    return _diagram_call(llm, cache, prompt, "No significant type system detected.")


# Pass 7 — Request / response lifecycle

_LIFECYCLE_USER = """Produce a Mermaid sequenceDiagram showing the full request/response lifecycle.

Rules:
- Start with Client, end with Client
- Show middleware chain in order: auth, rate limiting, validation, logging, etc.
- Show the happy path AND one error path (use alt/else blocks)
- Include: routing layer, service/business logic, data access (DB, cache, external APIs)
- Use ->> for calls, -->> for responses
- Max 10 participants
- Set applicable=false if this is not a server or API codebase

Return JSON:
{{
  "mermaid": "sequenceDiagram\\n    ...",
  "notes": "<2-3 sentences explaining the lifecycle>",
  "applicable": true
}}

Codebase:
{context}"""


def generate_request_lifecycle(
    llm: LLMClient,
    files: list[FileSummary],
    modules: list[ModuleSummary],
    arch: ArchitectureSummary | None,
    cache: Cache,
) -> DiagramResult:
    ctx = compact_context(
        files, modules, arch,
        hint_keywords=["middleware", "auth", "request", "response", "handler", "router", "server", "guard"],
    )
    prompt = _LIFECYCLE_USER.format(context=ctx)
    return _diagram_call(llm, cache, prompt, "No server/request lifecycle detected.")


# Pass 8 — Data flow

_DATAFLOW_USER = """Produce a Mermaid flowchart showing the data flow through the system.

Rules:
- Use flowchart TD
- [Rectangle] for processes/transforms
- {{Diamond}} for decisions/branches
- [(Cylinder)] for databases or caches
- ((Circle)) for start/end
- Label every edge with what flows along it
- Show: data entry points → validation → processing → storage → output
- Max 20 nodes

Return JSON:
{{
  "mermaid": "flowchart TD\\n    ...",
  "notes": "<2-3 sentences explaining the data flow>",
  "applicable": true
}}

Codebase:
{context}"""


def generate_data_flow(
    llm: LLMClient,
    files: list[FileSummary],
    modules: list[ModuleSummary],
    arch: ArchitectureSummary | None,
    cache: Cache,
) -> DiagramResult:
    ctx = compact_context(files, modules, arch)
    prompt = _DATAFLOW_USER.format(context=ctx)
    return _diagram_call(llm, cache, prompt, "Data flow could not be determined.")


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _ul(items: list[str], cls: str = "") -> str:
    if not items:
        return "<span class='none-label'>none</span>"
    ca = f' class="{cls}"' if cls else ""
    return "<ul" + ca + ">" + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ul>"


def _dep_graph_mermaid(dependency_graph: dict) -> str:
    if not dependency_graph:
        return "graph LR\n    A[No inter-module dependencies detected]"
    lines = ["graph LR"]
    ids: dict[str, str] = {}

    def nid(name: str) -> str:
        if name not in ids:
            safe = re.sub(r"[^a-zA-Z0-9]", "_", name)
            ids[name] = f"n{len(ids)}_{safe}"
        return ids[name]

    nodes: set[str] = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        nodes.update(deps)
    for n in sorted(nodes):
        lines.append(f'    {nid(n)}["{n.replace(chr(34), chr(39))}"]')
    for src, deps in sorted(dependency_graph.items()):
        for dep in deps:
            lines.append(f"    {nid(src)} --> {nid(dep)}")
    return "\n".join(lines)


def _diagram_panel(
    anchor: str,
    title: str,
    icon: str,
    result: DiagramResult | None,
) -> str:
    """Render a full diagram section: header + Mermaid panel + notes."""
    section_head = f"""
    <div class="section-header">
      <div class="section-icon">{icon}</div>
      <h2 class="section-title" id="{anchor}">{title}</h2>
    </div>"""

    if result is None or not result.applicable:
        na_msg = _esc(result.notes if result and result.notes else "Not applicable to this codebase.")
        return section_head + f"""
    <div class="diagram-panel not-applicable-panel">
      <div class="na-inner">
        <span class="na-icon">⊘</span>
        <p>{na_msg}</p>
      </div>
    </div>"""

    mermaid_escaped = _esc(result.mermaid)
    notes_html = f'<div class="diagram-notes">{_esc(result.notes)}</div>' if result.notes else ""

    return section_head + f"""
    <div class="diagram-panel">
      <div class="diagram-panel-header">
        <span class="diagram-panel-title">{title}</span>
        <button class="src-btn" onclick="toggleSrc(this)">View source</button>
      </div>
      <div class="diagram-body">
        <div class="mermaid">{mermaid_escaped}</div>
      </div>
      {notes_html}
      <div class="src-block" style="display:none">
        <pre>{mermaid_escaped}</pre>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html_report(analysis: RepoAnalysis) -> str:  # noqa: C901
    arch = analysis.architecture
    total_files  = len(analysis.files)
    total_tokens = sum(f.token_count for f in analysis.files)
    languages    = sorted({f.language for f in analysis.files if f.language != "unknown"})

    # ── Sidebar module list ──────────────────────────────────────────────────
    sidebar_modules = ""
    for mod in sorted(analysis.modules, key=lambda m: m.directory):
        slug = re.sub(r"[^a-z0-9]", "-", mod.directory.lower())
        sidebar_modules += (
            f'<a href="#mod-{slug}" class="sidebar-sub">'
            f'{_esc(mod.directory)}</a>'
        )

    # ── Stats row ────────────────────────────────────────────────────────────
    lang_badges = "".join(
        f'<span class="badge-lang">{_esc(l)}</span>' for l in languages
    )
    stack_badges = ""
    if arch:
        stack_badges = "".join(
            f'<span class="badge-stack">{_esc(s)}</span>' for s in arch.tech_stack
        )

    # ── Architecture layers ──────────────────────────────────────────────────
    layers_html = ""
    if arch and arch.layers:
        for layer in arch.layers:
            name = _esc(layer.get("name", ""))
            desc = _esc(layer.get("description", ""))
            mods = _ul(layer.get("modules", []))
            layers_html += f"""
          <div class="layer-card">
            <span class="layer-chip">{name}</span>
            <p class="layer-desc">{desc}</p>
            <div class="layer-mods">{mods}</div>
          </div>"""

    # ── Data flows ───────────────────────────────────────────────────────────
    flows_html = ""
    if arch and arch.data_flows:
        items = "".join(f"<li>{_esc(f)}</li>" for f in arch.data_flows)
        flows_html = f"<ol class='flow-list'>{items}</ol>"

    # ── Module wiki sections ─────────────────────────────────────────────────
    module_sections = ""
    for mod in sorted(analysis.modules, key=lambda m: m.directory):
        slug = re.sub(r"[^a-z0-9]", "-", mod.directory.lower())
        file_items = "".join(
            f'<tr><td class="mf-path">{_esc(fp)}</td>'
            f'<td class="mf-purpose">{_esc(next((f.purpose for f in analysis.files if f.path == fp), ""))}</td></tr>'
            for fp in mod.files
        )
        module_sections += f"""
        <div class="module-wiki" id="mod-{slug}">
          <div class="module-wiki-header">
            <code class="module-dir">{_esc(mod.directory)}</code>
            <span class="module-file-count">{len(mod.files)} file{'s' if len(mod.files) != 1 else ''}</span>
          </div>
          <p class="module-role-text">{_esc(mod.role)}</p>
          <div class="module-detail-grid">
            <div>
              <h4 class="detail-label">Public API</h4>
              {_ul(mod.public_api, "api-list")}
            </div>
            <div>
              <h4 class="detail-label">Dependencies</h4>
              {_ul(mod.dependencies, "dep-list")}
            </div>
          </div>
          <details class="file-details">
            <summary>Files in this module</summary>
            <table class="module-files-table">
              <thead><tr><th>Path</th><th>Purpose</th></tr></thead>
              <tbody>{file_items}</tbody>
            </table>
          </details>
        </div>"""

    # ── File table rows ──────────────────────────────────────────────────────
    file_rows = ""
    for fs in sorted(analysis.files, key=lambda f: f.path):
        chunked = ' <span class="chunk-badge">chunked</span>' if fs.chunked else ""
        cplx = (
            f'<details class="cplx-details"><summary>notes</summary>{_esc(fs.complexity_notes)}</details>'
            if fs.complexity_notes else ""
        )
        file_rows += f"""
          <tr>
            <td class="ft-path">{_esc(fs.path)}{chunked}</td>
            <td><span class="badge-lang sm">{_esc(fs.language)}</span></td>
            <td class="ft-purpose">{_esc(fs.purpose)}{cplx}</td>
            <td class="ft-tokens">{fs.token_count:,}</td>
          </tr>"""

    # ── Dep graph mermaid ────────────────────────────────────────────────────
    dep_mermaid = _dep_graph_mermaid(arch.dependency_graph if arch else {})
    dep_result = DiagramResult(
        mermaid=dep_mermaid,
        notes="Inter-module dependency graph derived from import analysis.",
        applicable=True,
    )

    # ── Diagram panels ───────────────────────────────────────────────────────
    arch_overview_text = _esc(arch.overview) if arch else "Architecture analysis unavailable."

    panel_dep        = _diagram_panel("dep-graph",    "Module Dependency Graph",        "🔗", dep_result)
    panel_components = _diagram_panel("components",   "Component Hierarchy",            "🧩", analysis.component_hierarchy)
    panel_api        = _diagram_panel("api-resources","API Resource Organization",      "🌐", analysis.api_resources)
    panel_types      = _diagram_panel("type-system",  "Type System &amp; Validation",   "🔷", analysis.type_system)
    panel_lifecycle  = _diagram_panel("lifecycle",    "Request / Response Lifecycle",   "↩", analysis.request_lifecycle)
    panel_dataflow   = _diagram_panel("data-flow",    "Data Flow",                      "⇌", analysis.data_flow)

    # ── Sidebar diagram links ────────────────────────────────────────────────
    diagram_links = ""
    diagram_nav = [
        ("dep-graph",     "Dependency Graph"),
        ("components",    "Component Hierarchy"),
        ("api-resources", "API Resources"),
        ("type-system",   "Type System"),
        ("lifecycle",     "Request Lifecycle"),
        ("data-flow",     "Data Flow"),
    ]
    for anchor, label in diagram_nav:
        diagram_links += f'<a href="#{anchor}" class="sidebar-sub">{label}</a>'

    repo_name = Path(analysis.repo_path).name

    # ════════════════════════════════════════════════════════════════════════
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_esc(repo_name)} — Repo Wiki</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    /* ── Reset & tokens ── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:        #0d0f16;
      --surface:   #161921;
      --surface2:  #1e2130;
      --surface3:  #252a3a;
      --border:    #2a2f45;
      --accent:    #5b7cff;
      --accent2:   #9b8bff;
      --green:     #3dd68c;
      --yellow:    #f5a623;
      --red:       #f87171;
      --teal:      #38bdf8;
      --text:      #dce3f0;
      --muted:     #7a86a0;
      --font:      'Inter', system-ui, -apple-system, sans-serif;
      --mono:      'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
      --sidebar-w: 248px;
      --radius:    10px;
    }}
    html {{ scroll-behavior: smooth; }}
    body {{
      background: var(--bg); color: var(--text);
      font-family: var(--font); font-size: 14px; line-height: 1.65;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ color: var(--accent2); }}
    h1,h2,h3,h4 {{ font-weight: 600; line-height: 1.3; }}
    code {{ font-family: var(--mono); font-size: 0.85em; }}

    /* ── Sidebar ── */
    .sidebar {{
      position: fixed; top: 0; left: 0; bottom: 0; width: var(--sidebar-w);
      background: var(--surface); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; overflow: hidden; z-index: 200;
    }}
    .sidebar-brand {{
      padding: 20px 20px 16px;
      border-bottom: 1px solid var(--border);
    }}
    .sidebar-repo {{
      font-size: 13px; font-weight: 700; color: var(--text);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .sidebar-sub-label {{
      font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
      color: var(--muted); margin-top: 2px;
    }}
    .sidebar-nav {{ flex: 1; overflow-y: auto; padding: 12px 0 24px; }}
    .sidebar-section {{
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: .1em; color: var(--muted);
      padding: 14px 20px 5px;
    }}
    .sidebar-nav a {{
      display: block; padding: 6px 20px;
      color: var(--muted); font-size: 13px;
      border-left: 2px solid transparent;
      transition: color .12s, border-color .12s, background .12s;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .sidebar-nav a:hover {{
      color: var(--text); border-left-color: var(--accent);
      background: var(--surface2);
    }}
    .sidebar-nav a.active {{
      color: var(--accent); border-left-color: var(--accent);
      background: var(--surface2); font-weight: 500;
    }}
    .sidebar-sub {{ padding-left: 32px !important; font-size: 12px !important; }}
    .sidebar-footer {{
      padding: 12px 20px; border-top: 1px solid var(--border);
      font-size: 11px; color: var(--muted);
    }}

    /* ── Main ── */
    .main {{
      margin-left: var(--sidebar-w);
      padding: 0 52px 100px;
      max-width: calc(var(--sidebar-w) + 960px);
    }}

    /* ── Page header ── */
    .page-header {{
      padding: 52px 0 40px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 48px;
    }}
    .page-header h1 {{
      font-size: 30px; font-weight: 800;
      background: linear-gradient(135deg, var(--text) 0%, var(--accent2) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text; margin-bottom: 4px;
    }}
    .page-repo-path {{
      font-family: var(--mono); font-size: 12px; color: var(--muted); margin-bottom: 24px;
    }}
    .stats-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
    .stat-card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 14px 20px; min-width: 110px;
      transition: border-color .15s;
    }}
    .stat-card:hover {{ border-color: var(--accent); }}
    .stat-value {{ font-size: 22px; font-weight: 700; color: var(--accent); }}
    .stat-label {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 16px; }}
    .badge-lang, .badge-stack {{
      padding: 3px 10px; border-radius: 20px; font-size: 11px;
      font-family: var(--mono); border: 1px solid var(--border);
    }}
    .badge-lang {{ background: var(--surface2); color: var(--accent); }}
    .badge-stack {{ background: var(--surface3); color: var(--accent2); }}
    .badge-lang.sm {{ padding: 2px 7px; font-size: 10px; }}

    /* ── Wiki section ── */
    .wiki-section {{ margin-bottom: 64px; }}
    .section-header {{
      display: flex; align-items: center; gap: 14px; margin-bottom: 20px;
    }}
    .section-icon {{
      width: 38px; height: 38px; display: flex; align-items: center;
      justify-content: center; background: var(--surface2);
      border: 1px solid var(--border); border-radius: 9px;
      font-size: 18px; flex-shrink: 0;
    }}
    .section-title {{
      font-size: 20px; font-weight: 700; color: var(--text);
      scroll-margin-top: 24px;
    }}

    /* ── Architecture callout ── */
    .arch-overview {{
      background: var(--surface); border: 1px solid var(--border);
      border-left: 3px solid var(--accent); border-radius: var(--radius);
      padding: 18px 22px; margin-bottom: 24px;
      font-size: 14px; line-height: 1.8; color: var(--text);
    }}
    .arch-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px; margin-bottom: 24px;
    }}
    .arch-card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 16px 18px;
    }}
    .arch-card-label {{
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: .08em; color: var(--muted); margin-bottom: 10px;
    }}
    .layers-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px; }}
    .layer-card {{
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 14px 16px; flex: 1; min-width: 160px;
    }}
    .layer-chip {{
      display: inline-block; background: var(--accent); color: #fff;
      font-size: 11px; font-weight: 700; padding: 2px 10px;
      border-radius: 20px; margin-bottom: 8px;
    }}
    .layer-desc {{ font-size: 12px; color: var(--muted); margin-bottom: 8px; }}
    .layer-mods ul {{ list-style: none; padding: 0; }}
    .layer-mods li {{ font-family: var(--mono); font-size: 11px; color: var(--accent2); }}
    .flow-list {{ padding-left: 20px; }}
    .flow-list li {{ margin-bottom: 6px; font-size: 13px; color: var(--text); }}

    /* ── Diagram panel ── */
    .diagram-panel {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); overflow: hidden; margin-bottom: 8px;
    }}
    .diagram-panel-header {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 18px; background: var(--surface2);
      border-bottom: 1px solid var(--border);
    }}
    .diagram-panel-title {{ font-size: 13px; font-weight: 600; color: var(--text); }}
    .src-btn {{
      background: var(--surface3); border: 1px solid var(--border);
      color: var(--muted); font-size: 11px; padding: 4px 10px;
      border-radius: 5px; cursor: pointer; font-family: var(--mono);
      transition: color .12s, border-color .12s;
    }}
    .src-btn:hover {{ color: var(--text); border-color: var(--accent); }}
    .diagram-body {{ padding: 28px; overflow-x: auto; }}
    .diagram-notes {{
      padding: 14px 18px; border-top: 1px solid var(--border);
      font-size: 13px; color: var(--muted); line-height: 1.7;
      background: var(--surface);
    }}
    .src-block {{
      background: var(--bg); border-top: 1px solid var(--border);
      padding: 16px 20px;
    }}
    .src-block pre {{
      font-family: var(--mono); font-size: 11px; color: var(--muted);
      white-space: pre-wrap; word-break: break-all;
    }}
    .not-applicable-panel {{
      border-style: dashed; border-color: var(--surface3);
    }}
    .na-inner {{
      display: flex; align-items: center; gap: 12px;
      padding: 24px 20px; color: var(--muted); font-size: 13px;
    }}
    .na-icon {{ font-size: 20px; opacity: .4; }}

    /* ── Module wiki ── */
    .module-wiki {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); margin-bottom: 14px; overflow: hidden;
    }}
    .module-wiki-header {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 18px; background: var(--surface2);
    }}
    .module-dir {{
      font-size: 14px; color: var(--accent); background: none;
    }}
    .module-file-count {{
      font-size: 11px; color: var(--muted); background: var(--surface3);
      padding: 2px 8px; border-radius: 20px; border: 1px solid var(--border);
    }}
    .module-role-text {{
      padding: 12px 18px; font-size: 13px; color: var(--muted);
      border-bottom: 1px solid var(--border); line-height: 1.6;
    }}
    .module-detail-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 0;
      border-bottom: 1px solid var(--border);
    }}
    .module-detail-grid > div {{ padding: 14px 18px; }}
    .module-detail-grid > div:first-child {{ border-right: 1px solid var(--border); }}
    .detail-label {{
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: .08em; color: var(--muted); margin-bottom: 8px;
    }}
    .api-list li {{ font-family: var(--mono); font-size: 11px; color: var(--green); margin-bottom: 3px; }}
    .dep-list li {{ font-family: var(--mono); font-size: 11px; color: var(--yellow); margin-bottom: 3px; }}
    .none-label {{ font-size: 12px; color: var(--muted); font-style: italic; }}
    .file-details summary {{
      padding: 10px 18px; cursor: pointer; font-size: 12px; color: var(--muted);
      user-select: none; list-style: none;
    }}
    .file-details summary::before {{ content: '▶ '; font-size: 10px; }}
    .file-details[open] summary::before {{ content: '▼ '; }}
    .module-files-table {{
      width: 100%; border-collapse: collapse; margin: 0;
    }}
    .module-files-table th {{
      font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
      color: var(--muted); padding: 8px 18px; text-align: left;
      border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
      background: var(--bg);
    }}
    .module-files-table td {{ padding: 8px 18px; border-bottom: 1px solid var(--border); }}
    .mf-path {{ font-family: var(--mono); font-size: 11px; color: var(--accent2); white-space: nowrap; }}
    .mf-purpose {{ font-size: 12px; color: var(--muted); }}

    /* ── File table ── */
    .search-bar {{
      width: 100%; padding: 9px 14px;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); color: var(--text);
      font-size: 13px; font-family: var(--font);
      margin-bottom: 16px; outline: none;
      transition: border-color .15s;
    }}
    .search-bar:focus {{ border-color: var(--accent); }}
    .search-bar::placeholder {{ color: var(--muted); }}
    .file-table-wrap {{ overflow-x: auto; }}
    .file-table {{
      width: 100%; border-collapse: collapse;
      border: 1px solid var(--border); border-radius: var(--radius);
      overflow: hidden;
    }}
    .file-table thead th {{
      text-align: left; font-size: 10px; text-transform: uppercase;
      letter-spacing: .08em; color: var(--muted); padding: 10px 14px;
      background: var(--surface2); border-bottom: 1px solid var(--border);
      position: sticky; top: 0;
    }}
    .file-table tbody tr {{ border-bottom: 1px solid var(--border); transition: background .1s; }}
    .file-table tbody tr:hover {{ background: var(--surface2); }}
    .file-table tbody tr:last-child {{ border-bottom: none; }}
    .file-table td {{ padding: 9px 14px; vertical-align: top; }}
    .ft-path {{
      font-family: var(--mono); font-size: 11px; color: var(--accent2);
      white-space: nowrap; max-width: 300px; overflow: hidden; text-overflow: ellipsis;
    }}
    .ft-purpose {{ font-size: 12px; color: var(--muted); }}
    .ft-tokens {{ font-family: var(--mono); font-size: 11px; color: var(--muted); text-align: right; }}
    .chunk-badge {{
      background: var(--yellow); color: #000; border-radius: 3px;
      font-size: 9px; font-weight: 700; padding: 1px 5px; vertical-align: middle;
    }}
    .cplx-details summary {{ font-size: 11px; color: var(--muted); cursor: pointer; margin-top: 3px; }}
    .cplx-details {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}

    /* ── Misc ── */
    ul {{ padding-left: 18px; }}
    ul li {{ margin-bottom: 3px; font-size: 13px; }}
    .divider {{ border: none; border-top: 1px solid var(--border); margin: 48px 0; }}
    #back-to-top {{
      position: fixed; bottom: 28px; right: 28px;
      background: var(--accent); color: #fff; border: none;
      width: 38px; height: 38px; border-radius: 50%; cursor: pointer;
      font-size: 18px; display: none; align-items: center; justify-content: center;
      box-shadow: 0 4px 16px rgba(91,124,255,.4); transition: opacity .2s;
      z-index: 300;
    }}
    #back-to-top.visible {{ display: flex; }}

    /* ── Responsive ── */
    @media (max-width: 860px) {{
      .sidebar {{ display: none; }}
      .main {{ margin-left: 0; padding: 0 20px 60px; }}
      .module-detail-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

<!-- ═══════════════ SIDEBAR ═══════════════ -->
<aside class="sidebar">
  <div class="sidebar-brand">
    <div class="sidebar-repo">{_esc(repo_name)}</div>
    <div class="sidebar-sub-label">Repo Wiki</div>
  </div>
  <nav class="sidebar-nav">
    <div class="sidebar-section">Overview</div>
    <a href="#overview">Home</a>

    <div class="sidebar-section">Architecture</div>
    <a href="#architecture">Architecture</a>

    <div class="sidebar-section">Diagrams</div>
    {diagram_links}

    <div class="sidebar-section">Modules</div>
    {sidebar_modules}

    <div class="sidebar-section">Reference</div>
    <a href="#files">All Files</a>
  </nav>
  <div class="sidebar-footer">
    Generated · {_esc(analysis.model)}
  </div>
</aside>

<!-- ═══════════════ MAIN ═══════════════ -->
<main class="main">

  <!-- ── Page header ── -->
  <header class="page-header" id="overview">
    <h1>{_esc(repo_name)}</h1>
    <div class="page-repo-path">{_esc(analysis.repo_path)}</div>
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-value">{total_files}</div>
        <div class="stat-label">Files</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{len(analysis.modules)}</div>
        <div class="stat-label">Modules</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{total_tokens:,}</div>
        <div class="stat-label">Tokens</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{len(languages)}</div>
        <div class="stat-label">Languages</div>
      </div>
    </div>
    <div class="badge-row">{lang_badges}</div>
    {'<div class="badge-row">' + stack_badges + '</div>' if stack_badges else ''}
  </header>

  <!-- ── Architecture ── -->
  <section class="wiki-section">
    <div class="section-header">
      <div class="section-icon">🏛</div>
      <h2 class="section-title" id="architecture">Architecture</h2>
    </div>
    <div class="arch-overview">{arch_overview_text}</div>

    <div class="arch-grid">
      <div class="arch-card">
        <div class="arch-card-label">Entry Points</div>
        {_ul(arch.entry_points if arch else [])}
      </div>
      <div class="arch-card">
        <div class="arch-card-label">Design Patterns</div>
        {_ul(arch.patterns if arch else [])}
      </div>
      <div class="arch-card">
        <div class="arch-card-label">Tech Stack</div>
        {_ul(arch.tech_stack if arch else [])}
      </div>
    </div>

    {'<h3 style="font-size:15px;margin-bottom:14px;">Architectural Layers</h3><div class="layers-row">' + layers_html + '</div>' if layers_html else ''}
    {'<h3 style="font-size:15px;margin:20px 0 12px;">Key Data Flows</h3>' + flows_html if flows_html else ''}
  </section>

  <hr class="divider">

  <!-- ── Dependency graph ── -->
  <section class="wiki-section">
    {panel_dep}
  </section>

  <!-- ── Component hierarchy ── -->
  <section class="wiki-section">
    {panel_components}
  </section>

  <!-- ── API resources ── -->
  <section class="wiki-section">
    {panel_api}
  </section>

  <!-- ── Type system ── -->
  <section class="wiki-section">
    {panel_types}
  </section>

  <!-- ── Request lifecycle ── -->
  <section class="wiki-section">
    {panel_lifecycle}
  </section>

  <!-- ── Data flow ── -->
  <section class="wiki-section">
    {panel_dataflow}
  </section>

  <hr class="divider">

  <!-- ── Modules ── -->
  <section class="wiki-section">
    <div class="section-header">
      <div class="section-icon">📦</div>
      <h2 class="section-title" id="modules">Modules</h2>
    </div>
    {module_sections}
  </section>

  <hr class="divider">

  <!-- ── Files ── -->
  <section class="wiki-section">
    <div class="section-header">
      <div class="section-icon">📄</div>
      <h2 class="section-title" id="files">All Files</h2>
    </div>
    <input
      id="file-search" class="search-bar"
      type="search" placeholder="Search files by path, language, or purpose…"
      autocomplete="off"
    >
    <div class="file-table-wrap">
      <table class="file-table">
        <thead>
          <tr>
            <th>Path</th><th>Language</th><th>Purpose</th><th>Tokens</th>
          </tr>
        </thead>
        <tbody id="file-tbody">{file_rows}</tbody>
      </table>
    </div>
  </section>

</main>

<button id="back-to-top" title="Back to top" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">↑</button>

<script>
  // Mermaid
  mermaid.initialize({{
    startOnLoad: true, theme: 'dark',
    themeVariables: {{
      darkMode: true, background: '#161921',
      primaryColor: '#5b7cff', primaryTextColor: '#dce3f0',
      lineColor: '#2a2f45', nodeBorder: '#2a2f45',
      clusterBkg: '#1e2130', edgeLabelBackground: '#161921',
    }},
    flowchart: {{ curve: 'basis' }},
    sequence: {{ actorMargin: 60 }},
  }});

  // Toggle Mermaid source
  function toggleSrc(btn) {{
    const panel = btn.closest('.diagram-panel');
    const block = panel.querySelector('.src-block');
    const visible = block.style.display !== 'none';
    block.style.display = visible ? 'none' : 'block';
    btn.textContent = visible ? 'View source' : 'Hide source';
  }}

  // File search
  const search = document.getElementById('file-search');
  if (search) {{
    search.addEventListener('input', function() {{
      const q = this.value.toLowerCase();
      document.querySelectorAll('#file-tbody tr').forEach(tr => {{
        tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }});
  }}

  // Active nav on scroll
  const navLinks = document.querySelectorAll('.sidebar-nav a[href^="#"]');
  const observer = new IntersectionObserver(entries => {{
    entries.forEach(e => {{
      if (e.isIntersecting) {{
        navLinks.forEach(a => a.classList.remove('active'));
        const link = document.querySelector(`.sidebar-nav a[href="#${{e.target.id}}"]`);
        if (link) link.classList.add('active');
      }}
    }});
  }}, {{ rootMargin: '-10% 0px -80% 0px' }});
  document.querySelectorAll('[id]').forEach(el => observer.observe(el));

  // Back to top
  const btt = document.getElementById('back-to-top');
  window.addEventListener('scroll', () => {{
    btt.classList.toggle('visible', window.scrollY > 400);
  }});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:  # noqa: C901
    repo_root = Path(args.path).resolve()
    if not repo_root.is_dir():
        console.print(f"[red]Error: {repo_root} is not a directory[/]")
        sys.exit(1)

    provider = args.provider
    env_var  = "GEMINI_API_KEY" if provider == PROVIDER_GEMINI else "OPENAI_API_KEY"
    cli_key  = args.gemini_api_key if provider == PROVIDER_GEMINI else args.api_key
    api_key  = cli_key or os.environ.get(env_var, "")
    if not api_key:
        flag = "--gemini-api-key" if provider == PROVIDER_GEMINI else "--api-key"
        console.print(f"[red]Error: provide {flag} or set {env_var}[/]")
        sys.exit(1)

    model = args.model or DEFAULT_MODELS[provider]
    llm   = LLMClient(provider=provider, model=model, api_key=api_key)
    cache = Cache(repo_root, enabled=not args.no_cache)
    extensions = SOURCE_EXTENSIONS if not args.extensions else {
        e if e.startswith(".") else f".{e}" for e in args.extensions.split(",")
    }

    console.rule("[bold blue]Repo Harness[/]")
    console.print(f"  Repo:     [cyan]{repo_root}[/]")
    console.print(f"  Provider: [cyan]{provider}[/]")
    console.print(f"  Model:    [cyan]{model}[/]")
    console.print(f"  Output:   [cyan]{args.output}[/]")
    console.print(f"  Cache:    [cyan]{'disabled' if args.no_cache else str(repo_root / CACHE_DIR_NAME)}[/]\n")

    gitignore = load_gitignore(repo_root)
    files = discover_files(repo_root, extensions, gitignore, args.max_files)
    console.print(f"[green]Found {len(files)} source files[/]\n")

    analysis = RepoAnalysis(repo_path=str(repo_root), model=f"{provider}/{model}")

    # ── Pass 1: per-file ──────────────────────────────────────────────────
    console.rule("Pass 1 — File Analysis")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Analyzing files…", total=len(files))
        for f in files:
            rel = str(f.relative_to(repo_root)).replace("\\", "/")
            prog.update(task, description=f"[dim]{rel[:60]}[/]")
            analysis.files.append(analyze_file(llm, f, repo_root, cache))
            prog.advance(task)

    # ── Pass 2: modules ───────────────────────────────────────────────────
    console.rule("Pass 2 — Module Synthesis")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        prog.add_task("Synthesizing modules…")
        analysis.modules = synthesize_modules(llm, analysis.files, cache)
    console.print(f"[green]Synthesized {len(analysis.modules)} module(s)[/]\n")

    # ── Pass 3: architecture ──────────────────────────────────────────────
    console.rule("Pass 3 — Architecture")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        prog.add_task("Building architecture overview…")
        analysis.architecture = synthesize_architecture(llm, analysis.modules, cache)
    console.print("[green]Architecture complete[/]\n")

    # ── Passes 4–8: diagrams ──────────────────────────────────────────────
    diagram_tasks = [
        ("Pass 4 — Component Hierarchy",    generate_component_hierarchy, "component_hierarchy"),
        ("Pass 5 — API Resources",          generate_api_resources,       "api_resources"),
        ("Pass 6 — Type System",            generate_type_system,         "type_system"),
        ("Pass 7 — Request Lifecycle",      generate_request_lifecycle,   "request_lifecycle"),
        ("Pass 8 — Data Flow",              generate_data_flow,           "data_flow"),
    ]
    for title, fn, attr in diagram_tasks:
        console.rule(title)
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
            prog.add_task(f"Generating diagram…")
            result = fn(llm, analysis.files, analysis.modules, analysis.architecture, cache)
        status = "[green]done[/]" if result.applicable else "[yellow]not applicable[/]"
        console.print(f"{title}: {status}\n")
        setattr(analysis, attr, result)

    # ── Report ────────────────────────────────────────────────────────────
    console.rule("Generating Report")
    html = build_html_report(analysis)
    output_path = Path(args.output)
    output_path.write_text(html, encoding="utf-8")
    console.print(f"[bold green]Wiki written to {output_path}[/]")

    if args.save_json:
        jp = output_path.with_suffix(".json")
        jp.write_text(json.dumps(asdict(analysis), indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]JSON written to {jp}[/]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep codebase analysis → DeepWiki-style HTML wiki with Mermaid diagrams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--path",    default=".",               help="Repository root (default: .)")
    parser.add_argument("--output",  default="repo_wiki.html",  help="Output HTML file")
    parser.add_argument(
        "--provider", default=PROVIDER_OPENAI,
        choices=[PROVIDER_OPENAI, PROVIDER_GEMINI],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model", default="",
        help=(
            f"Model name. Defaults: openai={DEFAULT_MODELS[PROVIDER_OPENAI]}, "
            f"gemini={DEFAULT_MODELS[PROVIDER_GEMINI]}"
        ),
    )
    parser.add_argument("--api-key",        default="", help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--gemini-api-key", default="", help="Gemini API key (or set GEMINI_API_KEY)")
    parser.add_argument("--extensions",     default="",
                        help="Comma-separated extensions to include, e.g. py,ts,go")
    parser.add_argument("--max-files", type=int, default=500,
                        help="Max files to analyze (default: 500)")
    parser.add_argument("--no-cache",  action="store_true", help="Disable response cache")
    parser.add_argument("--save-json", action="store_true", help="Also write raw JSON output")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
