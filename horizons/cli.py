"""Command-line interface: init | doctor | login | logout | run | status | report | memory search."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path

from horizons import __version__
from horizons import auth
from horizons.kb.recall import recall
from horizons.kb.store import KB
from horizons.llm import LLMError, make_client
from horizons.template import DEFAULT_IMAGE, LLM_PROVIDERS, TemplateError, load_topic

DEFAULT_WORKSPACE = ".horizons"

_INIT_TOPIC = '''[topic]
name = "{name}"
goal = "Describe the measurable goal in one sentence"
context = "Background, constraints and known baselines."

[tools]
allowed = ["literature", "python_sandbox"]

[tools.literature]
sources = ["semantic_scholar", "arxiv"]
max_papers = 20

[tools.python_sandbox]
baseline = "baseline.py"     # the program the engine is allowed to rewrite
evaluator = "evaluate.py"    # YOUR measurement; the engine can never change it
image = "{image}"
timeout_s = 120
memory = "1g"
cpus = 1.0

[validation]
metric = "score"
direction = "maximize"
target = {{ relative_to_baseline = 0.2 }}
guards = [{{ metric = "correct", op = "==", value = 1 }}]
replications = 3

[budget]
max_iterations = 10
max_llm_calls = 150
max_sandbox_runs = 50
max_wall_clock_min = 45
patience = 4
max_pivots = 3
'''

_INIT_BASELINE = '''def solve(n):
    """Baseline implementation the engine will try to improve."""
    return sum(i * i for i in range(n))
'''

_INIT_EVALUATOR = '''"""User-owned evaluator. Loads $HORIZONS_CANDIDATE, measures it, prints one marked JSON line."""
import json, os, runpy, time

MARKER = os.environ.pop("HORIZONS_RESULT_MARKER")  # hide it from candidate code
candidate = runpy.run_path(os.environ["HORIZONS_CANDIDATE"])
n = 200_000
t0 = time.perf_counter()
out = candidate["solve"](n)
elapsed = time.perf_counter() - t0
correct = int(out == sum(i * i for i in range(n)))
print(MARKER + json.dumps({"score": 1.0 / max(elapsed, 1e-9), "correct": correct}))
'''


def _resolve_workspace(args: argparse.Namespace) -> Path:
    return Path(args.workspace or os.environ.get("HORIZONS_WORKSPACE") or DEFAULT_WORKSPACE).resolve()


def _kb(args: argparse.Namespace, must_exist: bool = True) -> KB:
    ws = _resolve_workspace(args)
    path = ws / "kb.sqlite"
    if must_exist and not path.exists():
        raise SystemExit(f"no knowledge base at {path}; run `horizons run` first (or pass --workspace)")
    return KB(path)


# -- commands -----------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    d = Path(args.directory)
    d.mkdir(parents=True, exist_ok=True)
    files = {"topic.toml": _INIT_TOPIC.format(name=d.resolve().name, image=DEFAULT_IMAGE), "baseline.py": _INIT_BASELINE,
             "evaluate.py": _INIT_EVALUATOR}
    for name, text in files.items():
        p = d / name
        if p.exists():
            print(f"skip   {p} (exists)")
            continue
        p.write_text(text, encoding="utf-8")
        print(f"create {p}")
    print(f"\nEdit {d / 'topic.toml'}, then: horizons run {d / 'topic.toml'}")
    return 0


def _check(ok: bool | None, label: str, detail: str, hint: str = "") -> bool:
    mark = {True: "ok  ", False: "FAIL", None: "warn"}[ok]
    print(f"[{mark}] {label}: {detail}")
    if hint and ok is not True:
        print(f"       fix: {hint}")
    return ok is not False


def cmd_doctor(args: argparse.Namespace) -> int:
    from horizons.tools.sandbox import docker_available

    good = True
    v = sys.version_info
    good &= _check(v >= (3, 11), "python", f"{v.major}.{v.minor}.{v.micro}", "install Python 3.11+")
    try:
        import sqlite3

        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        con.close()
        good &= _check(True, "sqlite fts5", sqlite3.sqlite_version)
    except Exception as e:  # noqa: BLE001
        good &= _check(False, "sqlite fts5", str(e), "use a Python build whose SQLite includes FTS5")
    ok, msg = docker_available()
    good &= _check(ok, "docker", msg, "install/start Docker Desktop (or the docker daemon)")
    image = args.image
    if ok:
        import subprocess

        r = subprocess.run([shutil.which("docker"), "image", "inspect", "--format", "{{.Id}}", image],
                           capture_output=True, text=True, timeout=30)
        _check(True if r.returncode == 0 else None, "sandbox image", image if r.returncode == 0 else f"{image} not pulled",
               f"docker pull {image}   (otherwise the first run pulls it)")
    logins = _login_status()
    keys = {k: bool(os.environ.get(k)) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    _check(True if any(v.startswith("logged in") for v in logins.values()) else None, "subscription logins",
           ", ".join(f"{p}: {s}" for p, s in logins.items()),
           "horizons login claude   (Claude Pro/Max)   or   horizons login chatgpt   (ChatGPT Plus/Pro)")
    _check(True if any(keys.values()) else None, "llm api keys",
           ", ".join(f"{k}={'present' if p else 'absent'}" for k, p in keys.items()),
           "not needed if you logged in above; otherwise export ANTHROPIC_API_KEY=... or OPENAI_API_KEY=... "
           "(or use --llm scripted for the offline demo, or --llm openai with "
           "HORIZONS_OPENAI_BASE_URL=http://localhost:11434/v1 for Ollama)")
    _check(True if os.environ.get("SEMANTIC_SCHOLAR_API_KEY") else None, "semantic scholar key",
           "present" if os.environ.get("SEMANTIC_SCHOLAR_API_KEY") else "absent (optional; lower rate limits)",
           "request a free key at semanticscholar.org/product/api and export SEMANTIC_SCHOLAR_API_KEY")
    for host in ("api.semanticscholar.org", "export.arxiv.org"):
        try:
            socket.create_connection((host, 443), timeout=5).close()
            _check(True, f"network {host}", "reachable")
        except OSError as e:
            _check(None, f"network {host}", f"unreachable ({e})", "check your connection; runs can use --offline")
    ws = _resolve_workspace(args)
    _check(True, "workspace", str(ws))
    print("\nall required checks passed" if good else "\nsome required checks FAILED")
    return 0 if good else 1


def _login_status() -> dict[str, str]:
    """Per provider: 'logged in (...)' or 'not logged in'. Never shows token values."""
    store = auth.AuthStore()
    out = {}
    for p in auth.PROVIDERS:
        try:
            c = store.get(p)
        except auth.AuthError as e:
            out[p] = f"unreadable ({e})"
            continue
        if c is None:
            out[p] = "not logged in"
        elif c.expires_soon():
            out[p] = "logged in (token refreshes on next use)"
        else:
            out[p] = f"logged in (token valid until {time.strftime('%H:%M', time.localtime(c.expires_at))})"
    return out


def _open_url(no_browser: bool) -> Callable[[str], None]:
    def open_url(url: str) -> None:
        print("Open this link to sign in:\n\n  " + url + "\n")
        if not no_browser:
            try:
                webbrowser.open(url)
            except webbrowser.Error:
                pass
    return open_url


def cmd_login(args: argparse.Namespace) -> int:
    store = auth.AuthStore()
    try:
        if args.provider == "claude":
            creds = auth.login_claude(_open_url(args.no_browser), input)
        else:
            creds = auth.login_chatgpt(_open_url(args.no_browser), input)
        store.put(args.provider, creds)
    except auth.AuthError as e:
        print(f"login failed: {e}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nlogin cancelled", file=sys.stderr)
        return 1
    print(f"\nLogged in to {args.provider}. Saved to {store.path} (private to your user).")
    print(f"Runs now use it automatically, or force it with: horizons run TOPIC --llm {args.provider}")
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    store = auth.AuthStore()
    targets = auth.PROVIDERS if args.provider == "all" else (args.provider,)
    try:
        for p in targets:
            print(f"{p}: {'logged out' if store.delete(p) else 'was not logged in'}")
    except auth.AuthError as e:
        print(f"logout failed: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from horizons.controller import Controller, RunError
    from horizons.tools.sandbox import SandboxUnavailable

    try:
        spec = load_topic(args.topic)
    except TemplateError as e:
        print(f"invalid topic template: {e}", file=sys.stderr)
        return 2
    if any(rt.sandbox == "unsafe-local" for rt in spec.runtimes()) and not args.i_accept_unsafe_local:
        print("this topic sets sandbox = \"unsafe-local\" (no isolation). Re-run with --i-accept-unsafe-local "
              "only if you trust every piece of code the model might write.", file=sys.stderr)
        return 2
    try:
        client = make_client(args.llm or spec.llm.provider, args.model or spec.llm.model, spec.llm.base_url,
                             spec.llm.max_tokens, spec.llm.scripted_file)
    except LLMError as e:
        print(f"LLM setup failed: {e}", file=sys.stderr)
        return 2
    # Scripted responses cite the topic's local fixture papers, so scripted runs never touch the network.
    offline = args.offline or client.name == "scripted"
    ws = _resolve_workspace(args)
    kb = KB(ws / "kb.sqlite")
    ctl = Controller(spec, kb, client, ws, offline=offline, accept_unsafe_local=args.i_accept_unsafe_local,
                     debug=args.debug)
    try:
        run_id = ctl.resume(args.resume) if args.resume else ctl.start()
        status = ctl.run()
    except (RunError, SandboxUnavailable) as e:
        print(f"run failed: {e}", file=sys.stderr)
        return 1
    finally:
        kb.close()
    print(f"\nrun {run_id}: {status}\nreport: {ws / 'runs' / run_id / 'report.md'}")
    return 0 if status == "success" else 3 if status == "stopped" else 1


def cmd_status(args: argparse.Namespace) -> int:
    kb = _kb(args)
    try:
        if args.run_id:
            run = kb.get_run(args.run_id)
            if not run:
                print(f"no run {args.run_id}", file=sys.stderr)
                return 1
            b = run["budget"]
            print(f"{run['id']}  {run['status']}  phase={run['stage']}  topic={run['topic_name']}  llm={run['llm']}")
            print(f"  goal: {run['goal']}")
            print(f"  stop: {run['stop_reason'] or '-'}")
            print(f"  spend: {b}")
            for h in kb.hypotheses(args.run_id):
                print(f"  {h['id']}  it{h['iteration']:<2} {h['status']:<15} {h['statement'][:90]}")
            for e in kb.events(args.run_id)[-args.tail:]:
                print(f"  · [{e['stage']}] {e['message'][:140]}")
        else:
            runs = kb.list_runs()
            if not runs:
                print("no runs yet")
            for r in runs:
                print(f"{r['id']}  {r['status']:<8} {r['stage']:<12} {r['topic_name']:<24} {r['stop_reason'] or ''}"[:160])
    finally:
        kb.close()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from horizons.report import write_report

    kb = _kb(args)
    try:
        run = kb.get_run(args.run_id)
        if not run:
            print(f"no run {args.run_id}", file=sys.stderr)
            return 1
        try:
            spec = load_topic(run["topic_path"])
        except TemplateError as e:
            print(f"cannot reload topic template {run['topic_path']}: {e}", file=sys.stderr)
            return 1
        path = write_report(kb, args.run_id, spec, _resolve_workspace(args) / "runs" / args.run_id)
    finally:
        kb.close()
    print(path.read_text(encoding="utf-8") if args.print else path)
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    kb = _kb(args)
    try:
        kinds = tuple(args.kind) if args.kind else None
        hits = recall(kb, args.query, kinds=kinds, topic=args.topic, top_k=args.k)
        if not hits:
            print("no matches")
        for h in hits:
            arms = " ".join(f"{a}#{r}" for a, r in h.arms.items())
            print(f"{h.score:.4f}  {h.kind:<10} {h.ref_id:<14} [{arms}]  {h.text[:140]}")
    finally:
        kb.close()
    return 0


# -- parser ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="horizons", description="New Horizons active discovery engine")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--workspace", help=f"knowledge base + run artifacts dir (default ./{DEFAULT_WORKSPACE}, "
                                       "or $HORIZONS_WORKSPACE)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="scaffold a new topic directory")
    s.add_argument("directory")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("doctor", help="check prerequisites")
    s.add_argument("--image", default=DEFAULT_IMAGE)
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("login", help="sign in with a Claude or ChatGPT subscription (no API key needed)")
    s.add_argument("provider", choices=auth.PROVIDERS)
    s.add_argument("--no-browser", action="store_true", help="only print the sign-in link")
    s.set_defaults(fn=cmd_login)

    s = sub.add_parser("logout", help="remove a saved subscription login")
    s.add_argument("provider", choices=(*auth.PROVIDERS, "all"))
    s.set_defaults(fn=cmd_logout)

    s = sub.add_parser("run", help="run (or resume) the discovery loop for a topic.toml")
    s.add_argument("topic", help="path to topic.toml")
    s.add_argument("--llm", choices=LLM_PROVIDERS,
                   help="override [llm] provider (claude/chatgpt = subscription login; anthropic/openai = API key)")
    s.add_argument("--model", help="override model name")
    s.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    s.add_argument("--offline", action="store_true", help="no literature network calls (cache/fixture only)")
    s.add_argument("--i-accept-unsafe-local", action="store_true",
                   help="allow sandbox = \"unsafe-local\" (runs generated code on this machine without isolation)")
    s.add_argument("--debug", action="store_true", help="print full tracebacks when a run fails")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="list runs, or show one run")
    s.add_argument("run_id", nargs="?")
    s.add_argument("--tail", type=int, default=15, help="number of recent events to show")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("report", help="(re)write the markdown report for a run")
    s.add_argument("run_id")
    s.add_argument("--print", action="store_true", help="print the report instead of its path")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("memory", help="query the knowledge base")
    msub = s.add_subparsers(dest="memory_cmd", required=True)
    m = msub.add_parser("search", help="hybrid (BM25 + vector, RRF) recall")
    m.add_argument("query")
    m.add_argument("--kind", action="append", choices=("lesson", "hypothesis", "finding", "paper"))
    m.add_argument("--topic")
    m.add_argument("-k", type=int, default=10)
    m.set_defaults(fn=cmd_memory)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
