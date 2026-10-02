"""Work out what the target codebase actually is, before checking it.

Every check needs the same facts: which files are Python, where the prompts
live, which LLM provider is in use, whether there are tests, whether there is a
database or vector store. Computing that once and passing it around keeps the
checks independent of each other and means the repository is walked a single
time.

Detection is evidence-based: a provider is only reported when an import or an
endpoint for it is actually present, never guessed from a folder name, because a
wrong guess sends every downstream check looking for the wrong thing.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._retrieval_hints import MIN_SCORE as _RETRIEVAL_MIN_SCORE
from ._retrieval_hints import score_source as _score_retrieval

__all__ = ["AgentProject", "PromptSite", "StepSite", "discover"]

#: Directories never worth walking: build output, caches, virtualenvs, vendored
#: dependencies. Skipping them is both a speed and a correctness matter -- a
#: finding inside site-packages is noise nobody can act on.
_SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", ".venv", "venv", "env", "node_modules", "dist", "build",
        ".tox", ".nox", "site-packages", ".idea", ".vscode", ".trisol",
        ".next", ".cache", "htmlcov", ".eggs",
    }
)

_MAX_FILES = 2000
_MAX_FILE_BYTES = 512_000

#: import name -> provider label. Checked against real imports only.
_PROVIDERS = {
    "anthropic": "anthropic",
    "openai": "openai",
    "ollama": "ollama",
    "google.generativeai": "google",
    "google.genai": "google",
    "mistralai": "mistral",
    "cohere": "cohere",
    "litellm": "litellm",
    "langchain": "langchain",
    "langchain_openai": "langchain",
    "langchain_anthropic": "langchain",
    "llama_index": "llamaindex",
    "crewai": "crewai",
    "autogen": "autogen",
}

_VECTOR_STORES = {
    "chromadb": "chroma", "chroma": "chroma", "pinecone": "pinecone",
    "qdrant_client": "qdrant", "weaviate": "weaviate", "faiss": "faiss",
    "lancedb": "lancedb", "pgvector": "pgvector", "milvus": "milvus",
}

_DB_MODULES = {
    "sqlite3": "sqlite", "psycopg": "postgres", "psycopg2": "postgres",
    "asyncpg": "postgres", "pymongo": "mongodb", "redis": "redis",
    "sqlalchemy": "sqlalchemy", "duckdb": "duckdb", "supermemory": "supermemory",
}

#: A local endpoint is as good as an import for spotting a provider, since plenty
#: of agents call Ollama over plain HTTP with no client library at all.
_ENDPOINT_HINTS = (
    (re.compile(r"localhost:11434|127\.0\.0\.1:11434|/api/generate|/api/chat"), "ollama"),
    (re.compile(r"api\.anthropic\.com"), "anthropic"),
    (re.compile(r"api\.openai\.com"), "openai"),
)

#: Names that mean "this string is a prompt". Used for variables and for the
#: keyword arguments of model calls.
_PROMPT_NAMES = re.compile(
    r"(?:^|_)(prompt|system|instruction|instructions|template|persona|preamble)(?:$|_)",
    re.IGNORECASE,
)

#: A prompt worth judging. Below this a string assigned to a prompt-shaped
#: name is a label or a fragment ("TEMPLATE = \"{}\""), not an instruction.
#: Deliberately low: the worst real prompts are the shortest ones, so a floor
#: set to exclude them would hide the defect this tool exists to report.
MIN_PROMPT_CHARS = 15

#: The marker _string_value leaves where an f-string had an expression.
_PLACEHOLDER_RE = re.compile(r"\{\.\.\.\}")


@dataclass
class PromptSite:
    """A literal prompt string found in the source."""

    file: str
    line: int
    name: str
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def is_composition(self) -> bool:
        """True when this string only assembles other values.

        `f"{SYSTEM_PROMPT}
{context}
{question}"` is a prompt *template*, not a
        prompt: its instruction text lives in the variables it interpolates.
        Judging it as a thin prompt is a false positive, because the real
        instruction is reported separately at its own definition site.
        """
        placeholders = len(_PLACEHOLDER_RE.findall(self.text))
        if not placeholders:
            return False
        # Words that survive once the placeholders are removed. Scaffolding like
        # "Context:" and "Question:" is not instruction text, so a template is
        # identified by having few real words *per placeholder* rather than by
        # total length -- a long template is still a template.
        remaining = _PLACEHOLDER_RE.sub(" ", self.text)
        words = [w for w in re.findall(r"[A-Za-z']{2,}", remaining)]
        return len(words) < 4 * placeholders

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "name": self.name,
            "text": self.text,
            "chars": self.chars,
        }


@dataclass
class StepSite:
    """A call that looks like a model invocation -- the unit a lifecycle check
    reasons about."""

    file: str
    line: int
    callee: str
    provider: str = ""
    has_try: bool = False
    has_timeout: bool = False
    has_retry: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "callee": self.callee,
            "provider": self.provider,
            "has_try": self.has_try,
            "has_timeout": self.has_timeout,
            "has_retry": self.has_retry,
        }


@dataclass
class AgentProject:
    root: Path
    python_files: list[Path] = field(default_factory=list)
    prompt_files: list[Path] = field(default_factory=list)
    test_files: list[Path] = field(default_factory=list)
    config_files: list[Path] = field(default_factory=list)
    providers: set[str] = field(default_factory=set)
    vector_stores: set[str] = field(default_factory=set)
    databases: set[str] = field(default_factory=set)
    prompts: list[PromptSite] = field(default_factory=list)
    steps: list[StepSite] = field(default_factory=list)
    agent_files: list[Path] = field(default_factory=list)
    syntax_errors: list[tuple[Path, str]] = field(default_factory=list)
    truncated: bool = False
    #: Evidence that this agent retrieves context before generating, which is
    #: what makes a missing grounding rule a real defect rather than a nit.
    retrieval_evidence: list[str] = field(default_factory=list)

    @property
    def is_agent_project(self) -> bool:
        """Whether this looks like an AI agent codebase at all.

        Reported rather than enforced: the CLI warns and continues, because a
        project can be a real agent while using an SDK this list has never
        heard of.
        """
        return bool(self.providers or self.prompts or self.steps)

    @property
    def has_tests(self) -> bool:
        return bool(self.test_files)

    @property
    def does_retrieval(self) -> bool:
        return bool(self.vector_stores or self.retrieval_evidence)

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(path)

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root.resolve()),
            "python_files": len(self.python_files),
            "prompt_files": len(self.prompt_files),
            "test_files": len(self.test_files),
            "providers": sorted(self.providers),
            "vector_stores": sorted(self.vector_stores),
            "databases": sorted(self.databases),
            "prompts": len(self.prompts),
            "model_calls": len(self.steps),
            "agent_files": [self.relative(p) for p in self.agent_files],
            "syntax_errors": [[self.relative(p), msg] for p, msg in self.syntax_errors],
            "truncated": self.truncated,
            "does_retrieval": self.does_retrieval,
            "retrieval_evidence": self.retrieval_evidence,
            "is_agent_project": self.is_agent_project,
        }


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def discover(root: Path | str) -> AgentProject:
    """Walk ``root`` once and collect everything the checks need."""
    root_path = Path(root).resolve()
    project = AgentProject(root=root_path)

    if root_path.is_file():
        # Auditing a single file is explicitly supported: it is the fastest way
        # to try the tool on one agent module.
        files = [root_path]
        project.root = root_path.parent
    else:
        files = []
        for path in sorted(root_path.rglob("*")):
            if not path.is_file() or _should_skip(path.relative_to(root_path)):
                continue
            files.append(path)
            if len(files) >= _MAX_FILES:
                project.truncated = True
                break

    for path in files:
        suffix = path.suffix.lower()
        name = path.name.lower()
        if suffix == ".py":
            project.python_files.append(path)
            if name.startswith("test_") or name.endswith("_test.py") or "tests" in path.parts:
                project.test_files.append(path)
        elif suffix in (".txt", ".md", ".jinja", ".j2", ".tmpl", ".prompt"):
            if _PROMPT_NAMES.search(path.stem) or "prompt" in str(path).lower():
                project.prompt_files.append(path)
        elif suffix in (".toml", ".yaml", ".yml", ".ini", ".cfg", ".json", ".env"):
            project.config_files.append(path)

    for path in project.python_files:
        _scan_python(project, path)

    return project


def _read(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _scan_python(project: AgentProject, path: Path) -> None:
    source = _read(path)
    if source is None:
        return

    for pattern, provider in _ENDPOINT_HINTS:
        if pattern.search(source):
            project.providers.add(provider)

    # Scored, not any-match: see _retrieval_hints.MIN_SCORE for why one weak
    # signal is not enough to call something a retrieval agent.
    score, labels = _score_retrieval(source)
    if score >= _RETRIEVAL_MIN_SCORE:
        for label in labels:
            evidence = f"{project.relative(path)}: {label}"
            if evidence not in project.retrieval_evidence:
                project.retrieval_evidence.append(evidence)

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        # Recorded, not raised: a file that will not parse is itself a finding,
        # and the rest of the audit should still run.
        project.syntax_errors.append((path, f"line {exc.lineno}: {exc.msg}"))
        return

    targets_model_api = any(pattern.search(source) for pattern, _ in _ENDPOINT_HINTS)
    visitor = _Visitor(project, path, targets_model_api=targets_model_api)
    visitor.visit(tree)
    if visitor.found_agent_shape:
        project.agent_files.append(path)


class _Visitor(ast.NodeVisitor):
    """Single AST pass per file, gathering imports, prompts and model calls."""

    def __init__(self, project: AgentProject, path: Path, targets_model_api: bool = False):
        self.project = project
        self.path = path
        self.rel = project.relative(path)
        # True when the source mentions a model endpoint or imports a provider,
        # which is what licenses treating a raw HTTP POST as a model call.
        self.targets_model_api = targets_model_api
        self.found_agent_shape = False
        # Stack of enclosing Try nodes, so a model call can report whether it is
        # actually guarded -- the question every error-handling check asks.
        self._try_depth = 0

    # -- imports ---------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._note_module(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._note_module(node.module)
        self.generic_visit(node)

    def _note_module(self, module: str) -> None:
        # Match on the dotted prefix so `openai.types.chat` counts as openai
        # while `openai_helpers` (a local module) does not.
        parts = module.split(".")
        for size in range(len(parts), 0, -1):
            prefix = ".".join(parts[:size])
            if prefix in _PROVIDERS:
                self.project.providers.add(_PROVIDERS[prefix])
                self.found_agent_shape = True
            if prefix in _VECTOR_STORES:
                self.project.vector_stores.add(_VECTOR_STORES[prefix])
            if prefix in _DB_MODULES:
                self.project.databases.add(_DB_MODULES[prefix])

    # -- try/except depth -------------------------------------------------

    def visit_Try(self, node: ast.Try) -> None:
        self._try_depth += 1
        self.generic_visit(node)
        self._try_depth -= 1

    # -- prompts ----------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            name = _target_name(target)
            if name and _PROMPT_NAMES.search(name):
                text = _string_value(node.value)
                if text and len(text) >= MIN_PROMPT_CHARS:
                    site = PromptSite(self.rel, node.lineno, name, text)
                    if not site.is_composition:
                        self.project.prompts.append(site)
                    self.found_agent_shape = True
        self.generic_visit(node)

    # -- model calls ------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        callee = _callee_name(node.func)
        # A bare HTTP POST is only a model call when this file actually targets a
        # model endpoint; otherwise every unrelated web request would be flagged.
        is_model = callee is not None and (
            _is_model_call(callee) or (self.targets_model_api and _is_http_call(callee))
        )
        if callee is not None and is_model:
            self.found_agent_shape = True
            self.project.steps.append(
                StepSite(
                    file=self.rel,
                    line=node.lineno,
                    callee=callee,
                    provider=_provider_for(callee),
                    has_try=self._try_depth > 0,
                    has_timeout=any(k.arg == "timeout" for k in node.keywords),
                    has_retry=any(
                        k.arg in ("retries", "max_retries", "retry") for k in node.keywords
                    ),
                )
            )
            # A prompt passed inline to the call is still a prompt.
            for keyword in node.keywords:
                if keyword.arg and _PROMPT_NAMES.search(keyword.arg):
                    text = _string_value(keyword.value)
                    if text and len(text) >= MIN_PROMPT_CHARS:
                        site = PromptSite(self.rel, node.lineno, keyword.arg, text)
                        if not site.is_composition:
                            self.project.prompts.append(site)
        self.generic_visit(node)


_MODEL_CALL_RE = re.compile(
    r"(?:messages\.create|chat\.completions\.create|completions\.create|"
    r"generate_content|\bchat\b|\bcomplete\b|\binvoke\b|\bgenerate\b|"
    r"\bpredict\b|run_sync|\bask\b)",
    re.IGNORECASE,
)

#: Plenty of real agents skip the client libraries and POST to the provider's
#: HTTP endpoint directly, so a raw request call inside a file that talks to a
#: model API counts as a model call too. Without this, the agents that most need
#: timeout/retry findings are exactly the ones that report none.
_HTTP_CALL_RE = re.compile(
    r"(?:urlopen|requests\.post|requests\.request|httpx\.post|"
    r"session\.post|client\.post|aiohttp)",
    re.IGNORECASE,
)


def _is_model_call(callee: str) -> bool:
    return bool(_MODEL_CALL_RE.search(callee))


def _is_http_call(callee: str) -> bool:
    return bool(_HTTP_CALL_RE.search(callee))


def _provider_for(callee: str) -> str:
    lowered = callee.lower()
    for hint, provider in (
        ("anthropic", "anthropic"), ("claude", "anthropic"),
        ("openai", "openai"), ("ollama", "ollama"), ("genai", "google"),
    ):
        if hint in lowered:
            return provider
    return ""


def _target_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _callee_name(node: ast.expr) -> str | None:
    """Dotted name of a call target, e.g. ``client.messages.create``."""
    parts: list[str] = []
    current: ast.expr | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    elif isinstance(current, ast.Call):
        # e.g. get_client().messages.create -- keep the attribute chain.
        pass
    elif current is not None:
        return ".".join(reversed(parts)) or None
    return ".".join(reversed(parts)) or None


def _string_value(node: ast.expr) -> str | None:
    """A literal string, including an implicitly concatenated or f-string one.

    f-strings are included with their placeholders rendered as ``{...}``: the
    static text around the holes is still the prompt being judged.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                out.append(value.value)
            else:
                out.append("{...}")
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_value(node.left)
        right = _string_value(node.right)
        if left is not None and right is not None:
            return left + right
    return None
