"""Shared fixtures for the unittest suite (stdlib only)."""

from __future__ import annotations

import functools
import json
import os
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

from horizons.tools.sandbox import docker_available

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# Tests must never read, write or log out of the user's real subscription logins.
os.environ["HORIZONS_AUTH_FILE"] = str(Path(tempfile.mkdtemp(prefix="horizons-test-auth-")) / "auth.json")
os.environ["HORIZONS_CLAUDE_CLI_VERSION"] = "2.1.999"  # no npm lookups from tests


@functools.cache
def _docker() -> tuple[bool, str]:
    return docker_available()


def requires_docker(obj):
    ok, why = _docker()
    return unittest.skipUnless(ok, f"docker unavailable: {why}")(obj)


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="horizons-test-")).resolve()
        # mkdtemp is 0700; on Linux the sandbox's non-root user must be able to read topic folders.
        os.chmod(self.tmp, 0o755)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, rel: str, text: str) -> Path:
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
        return p

    def write_json(self, rel: str, obj) -> Path:
        return self.write(rel, json.dumps(obj))


# A tiny evaluator: calls candidate.value() and reports it (plus "correct").
TINY_EVALUATOR = """
import json, os, runpy
MARKER = os.environ.pop("HORIZONS_RESULT_MARKER")
ns = runpy.run_path(os.environ["HORIZONS_CANDIDATE"])
v = ns["value"]()
print(MARKER + json.dumps({"score": v, "correct": 1}))
"""

TINY_TOPIC = """
[topic]
name = "tiny"
goal = "Make value() return at least 10"

[tools]
allowed = ["python_sandbox"]

[tools.python_sandbox]
baseline = "baseline.py"
evaluator = "evaluate.py"
timeout_s = 20
memory = "256m"

[validation]
metric = "score"
direction = "maximize"
target = { absolute = 10 }
guards = [{ metric = "correct", op = "==", value = 1 }]
replications = 2

[budget]
max_iterations = 4
max_llm_calls = 40
max_sandbox_runs = 20
max_wall_clock_min = 5
patience = 3
max_pivots = 2

[llm]
scripted_file = "script.json"
"""


def hyp(statement: str, kind: str = "python_sandbox", **kw) -> dict:
    d = {"statement": statement, "rationale": "r", "test_kind": kind, "falsification": "f",
         "expected_effect": "e", "citations": [], "readiness": "testable"}
    d.update(kw)
    return d
