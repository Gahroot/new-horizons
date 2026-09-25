"""Load and strictly validate a ``topic.toml`` Topic Template.

The template is the only place that grants permissions (tools, evaluator,
data, budget). Anything the LLM or the literature says can never change it.
Unknown keys are rejected so typos fail loudly instead of silently
disabling a guard.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KNOWN_TOOLS = ("literature", "python_sandbox", "data_query")
KNOWN_SOURCES = ("semantic_scholar", "arxiv", "openalex")
GUARD_OPS = ("==", "!=", "<", "<=", ">", ">=")
SANDBOX_MODES = ("docker", "unsafe-local")
# claude / chatgpt = subscription logins (`horizons login ...`); anthropic / openai = API keys.
LLM_PROVIDERS = ("claude", "chatgpt", "anthropic", "openai", "scripted")
# Pinned by digest so the sandbox can't change underneath a run (pulled 2026-09-25).
DEFAULT_IMAGE = "python:3.13-slim@sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0"


class TemplateError(ValueError):
    """Raised when a topic template is invalid. Message is user-facing."""


@dataclass(frozen=True)
class Runtime:
    """How user evaluators and candidate code are executed."""

    sandbox: str = "docker"
    image: str = DEFAULT_IMAGE
    timeout_s: int = 120
    memory: str = "1g"
    cpus: float = 1.0
    gpus: str | None = None


@dataclass(frozen=True)
class LiteratureCfg:
    sources: tuple[str, ...] = KNOWN_SOURCES
    max_papers: int = 20
    fixture: Path | None = None  # local papers JSON, used offline (demo/tests) and merged online


@dataclass(frozen=True)
class PythonSandboxCfg:
    baseline: Path
    evaluator: Path
    runtime: Runtime


@dataclass(frozen=True)
class DataQueryCfg:
    database: Path
    evaluator: Path
    max_rows: int
    runtime: Runtime


@dataclass(frozen=True)
class Guard:
    metric: str
    op: str
    value: float

    def check(self, metrics: dict[str, Any]) -> bool:
        v = metrics.get(self.metric)
        if isinstance(v, bool):
            v = float(v)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return False
        return {
            "==": v == self.value,
            "!=": v != self.value,
            "<": v < self.value,
            "<=": v <= self.value,
            ">": v > self.value,
            ">=": v >= self.value,
        }[self.op]

    def describe(self) -> str:
        return f"{self.metric} {self.op} {self.value:g}"


@dataclass(frozen=True)
class Validation:
    metric: str
    direction: str  # minimize | maximize
    kind: str = "threshold"  # threshold | significance
    target_relative: float | None = None  # e.g. -0.5 => 50% lower than baseline
    target_absolute: float | None = None
    guards: tuple[Guard, ...] = ()
    replications: int = 3
    alpha: float = 0.05
    # Seeded evaluators: every judged candidate run is paired with a baseline run on the same
    # HORIZONS_SEED, and verdicts use the paired difference instead of comparing raw values.
    paired: bool = False

    def describe(self) -> str:
        if self.target_relative is not None:
            t = f"{self.target_relative:+.0%} vs baseline"
        elif self.target_absolute is not None:
            t = f"{'<=' if self.direction == 'minimize' else '>='} {self.target_absolute:g}"
        else:
            t = "significant improvement vs baseline"
        s = f"{self.metric} ({self.direction}) target {t}"
        if self.kind == "significance":
            s += f", p < {self.alpha} (Holm-corrected)"
        if self.guards:
            s += "; guards: " + ", ".join(g.describe() for g in self.guards)
        s += f"; replications: {self.replications}"
        if self.paired:
            s += " (each paired with a baseline run on the same seed)"
        return s


@dataclass(frozen=True)
class Budget:
    max_iterations: int = 12
    max_llm_calls: int = 200
    max_sandbox_runs: int = 60
    max_wall_clock_min: float = 60
    patience: int = 4
    max_pivots: int = 3
    max_refines: int = 2
    hypotheses_per_round: int = 3


@dataclass(frozen=True)
class LLMCfg:
    provider: str | None = None  # claude | chatgpt | anthropic | openai | scripted; None => auto
    model: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    scripted_file: Path | None = None


@dataclass(frozen=True)
class TopicSpec:
    root: Path
    path: Path
    name: str
    goal: str
    context: str
    allowed_tools: tuple[str, ...]
    validation: Validation
    budget: Budget
    llm: LLMCfg
    literature: LiteratureCfg | None = None
    python_sandbox: PythonSandboxCfg | None = None
    data_query: DataQueryCfg | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def allows(self, tool: str) -> bool:
        return tool in self.allowed_tools

    def runtimes(self) -> list[Runtime]:
        out = []
        if self.python_sandbox:
            out.append(self.python_sandbox.runtime)
        if self.data_query:
            out.append(self.data_query.runtime)
        return out


# ----------------------------------------------------------------------------
# helpers


def _table(d: dict, key: str, where: str, required: bool = False) -> dict:
    v = d.get(key)
    if v is None:
        if required:
            raise TemplateError(f"missing required table [{where}]")
        return {}
    if not isinstance(v, dict):
        raise TemplateError(f"[{where}] must be a table")
    return v


def _no_unknown(d: dict, allowed: set[str], where: str) -> None:
    extra = sorted(set(d) - allowed)
    if extra:
        raise TemplateError(f"unknown key(s) in [{where}]: {', '.join(extra)}")


def _str(d: dict, key: str, where: str, default: str | None = None, required: bool = False) -> str | None:
    v = d.get(key, default)
    if v is None:
        if required:
            raise TemplateError(f"[{where}] {key} is required")
        return None
    if not isinstance(v, str) or (required and not v.strip()):
        raise TemplateError(f"[{where}] {key} must be a non-empty string")
    return v


def _num(d: dict, key: str, where: str, default: float, lo: float, hi: float, integer: bool = False) -> Any:
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TemplateError(f"[{where}] {key} must be a number")
    if integer and not isinstance(v, int):
        raise TemplateError(f"[{where}] {key} must be an integer")
    if not math.isfinite(v) or not lo <= v <= hi:
        raise TemplateError(f"[{where}] {key} must be between {lo:g} and {hi:g}")
    return v


def contained_path(root: Path, value: Any, where: str, must_exist: bool = True) -> Path:
    """Resolve ``value`` relative to ``root`` and refuse anything escaping it.

    Rejects absolute paths, ``..`` components and symlinks whose target
    leaves the topic directory.
    """
    if not isinstance(value, str) or not value.strip():
        raise TemplateError(f"{where} must be a relative file path")
    p = Path(value)
    if p.is_absolute() or ".." in p.parts:
        raise TemplateError(f"{where} must stay inside the topic directory (got {value!r})")
    root_r = root.resolve()
    full = (root_r / p).resolve()
    if full != root_r and root_r not in full.parents:
        raise TemplateError(f"{where} resolves outside the topic directory (got {value!r})")
    if must_exist and not full.is_file():
        raise TemplateError(f"{where}: file not found: {full}")
    return full


_MEM_UNITS = {"k": 1 / 1024, "m": 1, "g": 1024}


def memory_mb(mem: str) -> float:
    m = mem.strip().lower().rstrip("b")
    if not m or m[-1] not in _MEM_UNITS:
        raise TemplateError(f"memory must look like '512m' or '1g' (got {mem!r})")
    try:
        n = float(m[:-1])
    except ValueError:
        raise TemplateError(f"memory must look like '512m' or '1g' (got {mem!r})") from None
    return n * _MEM_UNITS[m[-1]]


def _runtime(d: dict, where: str) -> Runtime:
    mode = _str(d, "sandbox", where, "docker")
    if mode not in SANDBOX_MODES:
        raise TemplateError(f"[{where}] sandbox must be one of {SANDBOX_MODES}")
    mem = _str(d, "memory", where, "1g")
    mb = memory_mb(mem)
    if not 64 <= mb <= 256 * 1024:
        raise TemplateError(f"[{where}] memory must be between 64m and 256g")
    image = _str(d, "image", where, DEFAULT_IMAGE)
    if image.startswith("-") or any(c.isspace() for c in image):
        raise TemplateError(f"[{where}] image is not a valid image reference")
    gpus = _str(d, "gpus", where)
    if gpus is not None and (gpus.startswith("-") or any(c.isspace() for c in gpus)):
        raise TemplateError(f"[{where}] gpus is invalid")
    return Runtime(
        sandbox=mode,
        image=image,
        timeout_s=_num(d, "timeout_s", where, 120, 1, 24 * 3600, integer=True),
        memory=mem,
        cpus=float(_num(d, "cpus", where, 1.0, 0.1, 256)),
        gpus=gpus,
    )


_RUNTIME_KEYS = {"sandbox", "image", "timeout_s", "memory", "cpus", "gpus"}


# ----------------------------------------------------------------------------


def load_topic(path: str | Path) -> TopicSpec:
    path = Path(path)
    if path.is_dir():
        path = path / "topic.toml"
    if not path.is_file():
        raise TemplateError(f"topic template not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise TemplateError(f"{path}: invalid TOML: {e}") from None
    return parse_topic(raw, path.parent.resolve(), path.resolve())


def parse_topic(raw: dict, root: Path, path: Path | None = None) -> TopicSpec:
    _no_unknown(raw, {"topic", "tools", "validation", "budget", "llm"}, "top level")

    t = _table(raw, "topic", "topic", required=True)
    _no_unknown(t, {"name", "goal", "context"}, "topic")
    name = _str(t, "name", "topic", required=True)
    if not all(c.isalnum() or c in "-_." for c in name):
        raise TemplateError("[topic] name may only contain letters, digits, '-', '_' and '.'")
    goal = _str(t, "goal", "topic", required=True)
    context = _str(t, "context", "topic", "") or ""

    tools = _table(raw, "tools", "tools", required=True)
    _no_unknown(tools, {"allowed", *KNOWN_TOOLS}, "tools")
    allowed = tools.get("allowed")
    if not isinstance(allowed, list) or not allowed or not all(isinstance(x, str) for x in allowed):
        raise TemplateError("[tools] allowed must be a non-empty list of tool names")
    bad = [x for x in allowed if x not in KNOWN_TOOLS]
    if bad:
        raise TemplateError(f"[tools] unknown tool(s): {', '.join(bad)}; known: {', '.join(KNOWN_TOOLS)}")
    allowed_t = tuple(dict.fromkeys(allowed))
    for k in KNOWN_TOOLS:
        if k in tools and k not in allowed_t:
            raise TemplateError(f"[tools.{k}] is configured but '{k}' is not in [tools] allowed")

    lit = None
    if "literature" in allowed_t:
        d = _table(tools, "literature", "tools.literature")
        _no_unknown(d, {"sources", "max_papers", "fixture"}, "tools.literature")
        srcs = d.get("sources", list(KNOWN_SOURCES))
        if not isinstance(srcs, list) or any(s not in KNOWN_SOURCES for s in srcs):
            raise TemplateError(f"[tools.literature] sources must be a subset of {KNOWN_SOURCES}")
        fixture = (contained_path(root, d["fixture"], "[tools.literature] fixture") if "fixture" in d else None)
        if not srcs and fixture is None:
            raise TemplateError("[tools.literature] needs at least one source or a fixture")
        lit = LiteratureCfg(tuple(dict.fromkeys(srcs)), _num(d, "max_papers", "tools.literature", 20, 1, 200, True),
                            fixture)

    py = None
    if "python_sandbox" in allowed_t:
        w = "tools.python_sandbox"
        d = _table(tools, "python_sandbox", w, required=True)
        _no_unknown(d, {"baseline", "evaluator", *_RUNTIME_KEYS}, w)
        py = PythonSandboxCfg(
            baseline=contained_path(root, d.get("baseline"), f"[{w}] baseline"),
            evaluator=contained_path(root, d.get("evaluator"), f"[{w}] evaluator"),
            runtime=_runtime(d, w),
        )
        if py.baseline == py.evaluator:
            raise TemplateError(f"[{w}] baseline and evaluator must be different files")

    dq = None
    if "data_query" in allowed_t:
        w = "tools.data_query"
        d = _table(tools, "data_query", w, required=True)
        _no_unknown(d, {"database", "evaluator", "max_rows", *_RUNTIME_KEYS}, w)
        dq = DataQueryCfg(
            database=contained_path(root, d.get("database"), f"[{w}] database"),
            evaluator=contained_path(root, d.get("evaluator"), f"[{w}] evaluator"),
            max_rows=_num(d, "max_rows", w, 10_000, 1, 1_000_000, True),
            runtime=_runtime(d, w),
        )

    v = _table(raw, "validation", "validation", required=True)
    _no_unknown(v, {"metric", "direction", "kind", "target", "guards", "replications", "alpha", "paired"},
                "validation")
    paired = v.get("paired", False)
    if not isinstance(paired, bool):
        raise TemplateError("[validation] paired must be true or false")
    if paired and py is None:
        raise TemplateError("[validation] paired needs python_sandbox (the baseline is rerun on each seed)")
    metric = _str(v, "metric", "validation", required=True)
    direction = _str(v, "direction", "validation", "maximize")
    if direction not in ("minimize", "maximize"):
        raise TemplateError("[validation] direction must be 'minimize' or 'maximize'")
    kind = _str(v, "kind", "validation", "threshold")
    if kind not in ("threshold", "significance"):
        raise TemplateError("[validation] kind must be 'threshold' or 'significance'")
    tgt = v.get("target", {})
    if not isinstance(tgt, dict):
        raise TemplateError("[validation] target must be a table like { absolute = 0.9 }")
    _no_unknown(tgt, {"relative_to_baseline", "absolute"}, "validation.target")
    if len(tgt) > 1:
        raise TemplateError("[validation] target: use either relative_to_baseline or absolute, not both")
    rel = abs_ = None
    if "relative_to_baseline" in tgt:
        rel = float(_num(tgt, "relative_to_baseline", "validation.target", 0, -1e6, 1e6))
        if rel == 0:
            raise TemplateError("[validation] relative_to_baseline must be non-zero")
        if (direction == "minimize") != (rel < 0):
            raise TemplateError("[validation] relative_to_baseline sign must match direction "
                                "(negative for minimize, positive for maximize)")
        if py is None:
            raise TemplateError("[validation] relative_to_baseline needs python_sandbox with a baseline")
    if "absolute" in tgt:
        abs_ = float(_num(tgt, "absolute", "validation.target", 0, -1e300, 1e300))
    if kind == "threshold" and rel is None and abs_ is None:
        raise TemplateError("[validation] threshold validation needs a target")
    if kind == "significance" and py is None:
        raise TemplateError("[validation] significance validation needs python_sandbox (a baseline to compare against)")
    guards_raw = v.get("guards", [])
    if not isinstance(guards_raw, list):
        raise TemplateError("[validation] guards must be a list")
    guards = []
    for i, g in enumerate(guards_raw):
        w = f"validation.guards[{i}]"
        if not isinstance(g, dict):
            raise TemplateError(f"[{w}] must be a table")
        _no_unknown(g, {"metric", "op", "value"}, w)
        op = _str(g, "op", w, required=True)
        if op not in GUARD_OPS:
            raise TemplateError(f"[{w}] op must be one of {GUARD_OPS}")
        guards.append(Guard(_str(g, "metric", w, required=True), op, float(_num(g, "value", w, 0, -1e300, 1e300))))
    validation = Validation(
        metric=metric,
        direction=direction,
        kind=kind,
        target_relative=rel,
        target_absolute=abs_,
        guards=tuple(guards),
        replications=_num(v, "replications", "validation", 3, 1, 50, True),
        alpha=float(_num(v, "alpha", "validation", 0.05, 1e-9, 0.5)),
        paired=paired,
    )
    if kind == "significance":
        # Baseline and candidate each get `replications` samples; the smallest p an exact
        # permutation test can reach is 1 / C(2r, r). Refuse templates that can never succeed.
        r = validation.replications
        if 1 / math.comb(2 * r, r) >= validation.alpha:
            need = next(n for n in range(2, 51) if 1 / math.comb(2 * n, n) < validation.alpha)
            raise TemplateError(f"[validation] with alpha {validation.alpha:g}, significance needs "
                                f"replications >= {need} (got {r}); fewer samples can never reach significance")

    b = _table(raw, "budget", "budget")
    bkeys = set(Budget.__dataclass_fields__)
    _no_unknown(b, bkeys, "budget")
    budget = Budget(
        max_iterations=_num(b, "max_iterations", "budget", 12, 1, 10_000, True),
        max_llm_calls=_num(b, "max_llm_calls", "budget", 200, 1, 100_000, True),
        max_sandbox_runs=_num(b, "max_sandbox_runs", "budget", 60, 1, 100_000, True),
        max_wall_clock_min=float(_num(b, "max_wall_clock_min", "budget", 60, 0.1, 7 * 24 * 60)),
        patience=_num(b, "patience", "budget", 4, 1, 10_000, True),
        max_pivots=_num(b, "max_pivots", "budget", 3, 0, 10_000, True),
        max_refines=_num(b, "max_refines", "budget", 2, 0, 100, True),
        hypotheses_per_round=_num(b, "hypotheses_per_round", "budget", 3, 1, 20, True),
    )

    lraw = _table(raw, "llm", "llm")
    _no_unknown(lraw, {"provider", "model", "base_url", "max_tokens", "scripted_file"}, "llm")
    provider = _str(lraw, "provider", "llm")
    if provider is not None and provider not in LLM_PROVIDERS:
        raise TemplateError(f"[llm] provider must be one of: {', '.join(LLM_PROVIDERS)}")
    base_url = _str(lraw, "base_url", "llm")
    if base_url is not None and not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise TemplateError("[llm] base_url must be https:// (plain http only for localhost)")
    llm = LLMCfg(
        provider=provider,
        model=_str(lraw, "model", "llm"),
        base_url=base_url,
        max_tokens=_num(lraw, "max_tokens", "llm", 4096, 256, 200_000, True),
        scripted_file=(contained_path(root, lraw["scripted_file"], "[llm] scripted_file")
                       if "scripted_file" in lraw else None),
    )

    return TopicSpec(
        root=root,
        path=path or root / "topic.toml",
        name=name,
        goal=goal,
        context=context,
        allowed_tools=allowed_t,
        validation=validation,
        budget=budget,
        llm=llm,
        literature=lit,
        python_sandbox=py,
        data_query=dq,
        raw=raw,
    )
