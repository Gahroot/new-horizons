"""Run a user-owned evaluator against agent-written code in a hardened container.

Hardening flags follow OpenSandbox's Docker host config (no-new-privileges,
dropped capabilities, pids limit, memory/CPU limits, explicit network mode),
tightened for untrusted code: ``--network none``, read-only root filesystem,
non-root user, tmpfs ``/tmp``, the topic directory mounted read-only, and only
the attempt directory writable. Every subprocess call is an argv list.

Result contract: the evaluator prints one line
``<HORIZONS_RESULT_MARKER><json object of metrics>``. The marker is random per
execution; only the *last* marked stdout line is parsed. Non-finite or
non-numeric metric values make the result ``invalid``.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from horizons.auth import default_auth_file
from horizons.template import Runtime, memory_mb

OUTPUT_CAP = 64_000
START_GRACE_S = 5  # container start-up allowance on top of the topic's timeout_s
CONTAINER_TOPIC = "/topic"
CONTAINER_WORK = "/work"


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class SandboxResult:
    status: str  # ok | errored | timeout | invalid
    metrics: dict[str, float] = field(default_factory=dict)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    error: str = ""


def parse_metrics(stdout: str, marker: str) -> tuple[dict[str, float] | None, str]:
    lines = [ln for ln in stdout.splitlines() if ln.startswith(marker)]
    if not lines:
        return None, "evaluator printed no result line"
    try:
        data = json.loads(lines[-1][len(marker):])
    except json.JSONDecodeError as e:
        return None, f"result line is not JSON: {e}"
    if not isinstance(data, dict) or not data:
        return None, "result must be a non-empty JSON object"
    out: dict[str, float] = {}
    for k, v in list(data.items())[:100]:
        if not isinstance(k, str) or len(k) > 100:
            return None, "metric names must be short strings"
        if isinstance(v, bool):
            v = float(v)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return None, f"metric {k!r} is not a finite number"
        out[k] = float(v)
    return out, ""


def _tail(s: bytes | str | None) -> str:
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    return s[-OUTPUT_CAP:]


def docker_available() -> tuple[bool, str]:
    exe = shutil.which("docker")
    if not exe:
        return False, "docker CLI not found on PATH"
    try:
        r = subprocess.run([exe, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True,
                           timeout=20)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"docker info failed: {e}"
    if r.returncode != 0:
        return False, (r.stderr.strip().splitlines() or ["docker daemon not reachable"])[-1]
    return True, r.stdout.strip()


class DockerSandbox:
    """Runs ``python <evaluator>`` inside a locked-down container."""

    def __init__(self, runtime: Runtime, topic_root: Path, workspace: Path | None = None):
        ok, msg = docker_available()
        if not ok:
            raise SandboxUnavailable(f"Docker is required for the sandbox but is unavailable: {msg}. "
                                     "Start Docker Desktop / the docker daemon and run `horizons doctor`.")
        self.docker = shutil.which("docker")
        self.rt = runtime
        self.root = topic_root.resolve()
        self.workspace = workspace.resolve() if workspace else None
        # Subscription tokens must never be readable by generated code.
        self.auth_dir = default_auth_file().resolve().parent
        if self.auth_dir == self.root:
            raise SandboxUnavailable(f"the login file {default_auth_file()} is inside the topic folder, which the "
                                     "sandbox can read. Move it (HORIZONS_AUTH_FILE) or use another topic folder.")

    def argv(self, name: str, attempt_dir: Path, evaluator: Path, env: dict[str, str]) -> list[str]:
        rel = evaluator.resolve().relative_to(self.root).as_posix()
        mem = f"{int(memory_mb(self.rt.memory))}m"
        argv = [
            self.docker, "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "256",
            "--memory", mem, "--memory-swap", mem,
            "--cpus", f"{self.rt.cpus:g}",
            "--ulimit", "nofile=256:256",
            "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m",
            "--mount", f"type=bind,source={self.root},target={CONTAINER_TOPIC},readonly",
            "--mount", f"type=bind,source={attempt_dir.resolve()},target={CONTAINER_WORK}",
            "--workdir", CONTAINER_WORK,
        ]
        # Hide the engine's own workspace (KB, logs) if it lives inside the topic dir.
        if self.workspace and (self.workspace == self.root or self.root in self.workspace.parents):
            argv += ["--tmpfs", f"{CONTAINER_TOPIC}/{self.workspace.relative_to(self.root).as_posix()}:ro,size=1k"]
        # Same for the folder holding subscription login tokens.
        if self.root in self.auth_dir.parents and self.auth_dir != self.workspace:
            argv += ["--tmpfs", f"{CONTAINER_TOPIC}/{self.auth_dir.relative_to(self.root).as_posix()}:ro,size=1k"]
        if self.rt.gpus:
            argv += ["--gpus", self.rt.gpus]
        for k, v in env.items():
            argv += ["--env", f"{k}={v}"]
        argv += [self.rt.image, "python", "-I", f"{CONTAINER_TOPIC}/{rel}"]
        return argv

    def run(self, attempt_dir: Path, evaluator: Path, extra_env: dict[str, str] | None = None) -> SandboxResult:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(attempt_dir, 0o777)  # container runs as nobody; only this scratch dir is writable
        marker = f"@@HORIZONS_RESULT_{secrets.token_hex(12)}@@"
        env = {"HORIZONS_RESULT_MARKER": marker, "HORIZONS_WORK": CONTAINER_WORK,
               "HORIZONS_CANDIDATE": f"{CONTAINER_WORK}/candidate.py", "HOME": "/tmp",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", **(extra_env or {})}
        name = f"horizons-{secrets.token_hex(6)}"
        argv = self.argv(name, attempt_dir, evaluator, env)
        t0 = time.monotonic()
        try:
            p = subprocess.run(argv, capture_output=True, timeout=self.rt.timeout_s + START_GRACE_S, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            subprocess.run([self.docker, "kill", name], capture_output=True, timeout=30)
            subprocess.run([self.docker, "rm", "-f", name], capture_output=True, timeout=30)
            return SandboxResult("timeout", stdout=_tail(e.stdout), stderr=_tail(e.stderr),
                                 duration_s=time.monotonic() - t0,
                                 error=f"killed after {self.rt.timeout_s}s timeout "
                                       f"(+{START_GRACE_S}s container start-up allowance)")
        return _finish(p.returncode, _tail(p.stdout), _tail(p.stderr), time.monotonic() - t0, marker)


class UnsafeLocalSandbox:
    """Host subprocess runner. NOT isolated: only for trusted, explicit opt-in use."""

    def __init__(self, runtime: Runtime, topic_root: Path):
        self.rt = runtime
        self.root = topic_root.resolve()

    def run(self, attempt_dir: Path, evaluator: Path, extra_env: dict[str, str] | None = None) -> SandboxResult:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        marker = f"@@HORIZONS_RESULT_{secrets.token_hex(12)}@@"
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HORIZONS_RESULT_MARKER": marker,
               "HORIZONS_WORK": str(attempt_dir.resolve()),
               "HORIZONS_CANDIDATE": str((attempt_dir / "candidate.py").resolve()),
               "HOME": str(attempt_dir.resolve()), "PYTHONDONTWRITEBYTECODE": "1", **(extra_env or {})}
        t0 = time.monotonic()
        try:
            p = subprocess.run([sys.executable, "-I", str(evaluator.resolve())], cwd=attempt_dir, env=env,
                               capture_output=True, timeout=self.rt.timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            return SandboxResult("timeout", stdout=_tail(e.stdout), stderr=_tail(e.stderr),
                                 duration_s=time.monotonic() - t0, error=f"killed after {self.rt.timeout_s}s timeout")
        return _finish(p.returncode, _tail(p.stdout), _tail(p.stderr), time.monotonic() - t0, marker)


def _finish(code: int, out: str, err: str, dur: float, marker: str) -> SandboxResult:
    # Never echo the marker back into stored logs: it would let later candidates forge results.
    shown_out = out.replace(marker, "<result>")
    if code != 0:
        hint = " (killed: likely out of memory)" if code in (137, -9) else ""
        last = (err.strip().splitlines() or ["no stderr"])[-1]
        return SandboxResult("errored", exit_code=code, stdout=shown_out, stderr=err, duration_s=dur,
                             error=f"exit code {code}{hint}: {last[:500]}")
    metrics, why = parse_metrics(out, marker)
    if metrics is None:
        return SandboxResult("invalid", exit_code=code, stdout=shown_out, stderr=err, duration_s=dur, error=why)
    return SandboxResult("ok", metrics=metrics, exit_code=code, stdout=shown_out, stderr=err, duration_s=dur)


def make_sandbox(runtime: Runtime, topic_root: Path, workspace: Path | None, accept_unsafe_local: bool):
    if runtime.sandbox == "unsafe-local":
        if not accept_unsafe_local:
            raise SandboxUnavailable('topic requests sandbox = "unsafe-local", which runs generated code directly '
                                     "on this machine. Re-run with --i-accept-unsafe-local to allow it.")
        return UnsafeLocalSandbox(runtime, topic_root)
    return DockerSandbox(runtime, topic_root, workspace)
