"""Allowlist-gated tool registry.

The Topic Template's ``[tools] allowed`` list is the only thing that decides
which tools exist for a run. The controller asks the registry by name; a
tool that isn't allowed is refused even if its implementation is present.
"""

from __future__ import annotations

from typing import Any, Callable

from horizons.template import KNOWN_TOOLS, TopicSpec


class ToolNotAllowed(PermissionError):
    pass


class ToolRegistry:
    def __init__(self, spec: TopicSpec, factories: dict[str, Callable[[], Any]]):
        unknown = set(factories) - set(KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"unknown tool factories: {sorted(unknown)}")
        self._allowed = frozenset(spec.allowed_tools)
        self._factories = {k: v for k, v in factories.items() if k in self._allowed}
        self._instances: dict[str, Any] = {}

    @property
    def allowed(self) -> frozenset[str]:
        return self._allowed

    def get(self, name: str) -> Any:
        if name not in self._allowed:
            raise ToolNotAllowed(f"tool {name!r} is not in this topic's allowed tools "
                                 f"({', '.join(sorted(self._allowed))})")
        if name not in self._factories:
            raise ToolNotAllowed(f"tool {name!r} is allowed but has no implementation configured")
        if name not in self._instances:
            self._instances[name] = self._factories[name]()
        return self._instances[name]
