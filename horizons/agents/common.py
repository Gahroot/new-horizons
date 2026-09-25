"""Shared run context and prompt rendering for the agents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any, Callable

from horizons.budget import BudgetGuard
from horizons.kb.store import KB
from horizons.llm import LLM
from horizons.template import TopicSpec
from horizons.tools.registry import ToolRegistry

TESTED_STATUSES = ("supported", "refuted", "errored", "inconclusive")


def load_prompt(name: str) -> str:
    return resources.files("horizons").joinpath("prompts", f"{name}.md").read_text(encoding="utf-8")


def render(name: str, **values: Any) -> str:
    """Fill a prompt template (``$name`` placeholders; braces in values are harmless)."""
    return Template(load_prompt(name)).substitute({k: _s(v) for k, v in values.items()})


def system_prompt() -> str:
    return load_prompt("_system")


def untrusted(text: str, cap: int = 12000) -> str:
    """Neutralise attempts to close the <untrusted> fence from inside data."""
    text = str(text)[:cap]
    return text.replace("</untrusted", "<\\/untrusted").replace("<untrusted", "<\\untrusted")


def _s(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, indent=1, default=str)


@dataclass
class Ctx:
    spec: TopicSpec
    kb: KB
    llm: LLM
    budget: BudgetGuard
    tools: ToolRegistry
    run_id: str
    run_dir: Path
    sandbox_factory: Callable[[str], Any]  # tool name -> sandbox for that tool's runtime
    log: Callable[[str, str], None] = lambda stage, msg: None
    _sandboxes: dict[str, Any] = field(default_factory=dict)

    def sandbox(self, tool: str) -> Any:
        if tool not in self._sandboxes:
            self._sandboxes[tool] = self.sandbox_factory(tool)
        return self._sandboxes[tool]

    def ask_json(self, task: str, **values: Any) -> Any:
        return self.llm.json(task, system_prompt(), render(task, **values))

    def ask_text(self, task: str, **values: Any) -> str:
        return self.llm.text(task, system_prompt(), render(task, **values))

    def tools_str(self) -> str:
        return ", ".join(self.spec.allowed_tools)
