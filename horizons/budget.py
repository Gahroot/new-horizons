"""Fail-closed budget guard.

Adapted from the PRAXIST budget ledger idea: every costly action asks the
guard *before* it happens, and the guard refuses once a cap would be
exceeded. Counters are plain data so the controller can persist them and a
resumed run keeps its spend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from horizons.template import Budget


class BudgetExceeded(RuntimeError):
    def __init__(self, resource: str, used: float, cap: float):
        super().__init__(f"budget exhausted: {resource} ({used:g}/{cap:g})")
        self.resource = resource


@dataclass
class BudgetGuard:
    caps: Budget
    llm_calls: int = 0
    sandbox_runs: int = 0
    iterations: int = 0
    elapsed_before_s: float = 0.0  # wall clock from earlier sessions (resume)
    _started: float = field(default_factory=time.monotonic, repr=False)

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "sandbox_runs": self.sandbox_runs,
            "iterations": self.iterations,
            "elapsed_s": round(self.elapsed_s(), 3),
        }

    @classmethod
    def from_dict(cls, caps: Budget, d: dict | None) -> "BudgetGuard":
        d = d or {}
        return cls(
            caps=caps,
            llm_calls=int(d.get("llm_calls", 0)),
            sandbox_runs=int(d.get("sandbox_runs", 0)),
            iterations=int(d.get("iterations", 0)),
            elapsed_before_s=float(d.get("elapsed_s", 0.0)),
        )

    # -- checks -------------------------------------------------------------
    def elapsed_s(self) -> float:
        return self.elapsed_before_s + (time.monotonic() - self._started)

    def check_clock(self) -> None:
        cap = self.caps.max_wall_clock_min * 60
        if self.elapsed_s() >= cap:
            raise BudgetExceeded("wall clock seconds", self.elapsed_s(), cap)

    def _charge(self, attr: str, cap: int, label: str, n: int = 1) -> None:
        self.check_clock()
        used = getattr(self, attr)
        if used + n > cap:
            raise BudgetExceeded(label, used, cap)
        setattr(self, attr, used + n)

    def charge_llm(self) -> None:
        self._charge("llm_calls", self.caps.max_llm_calls, "llm calls")

    def charge_sandbox(self, n: int = 1) -> None:
        self._charge("sandbox_runs", self.caps.max_sandbox_runs, "sandbox runs", n)

    def charge_iteration(self) -> None:
        self._charge("iterations", self.caps.max_iterations, "iterations")
