"""Execute: turn a hypothesis into a measured experiment via an allowed tool.

python_sandbox - the model rewrites the current best program (lineage kept,
  earlier failed attempts fed back as in SkyDiscover); the user-owned
  evaluator measures it in the sandbox.
data_query     - the model writes one read-only SELECT; the user-owned data
  evaluator computes metrics from the exported rows in the sandbox.
literature     - targeted search; the model classifies each paper as
  supports/challenges/unrelated and must quote the abstract; quotes are
  verified in code before an edge is recorded.

The model never produces the metrics: they come from evaluators or from
counting verified evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from horizons.agents.common import Ctx, untrusted
from horizons.agents.deconstructor import format_papers
from horizons.kb import graph
from horizons.llm import extract_code
from horizons.tools.data_query import QueryError

MAX_CODE_BYTES = 200_000


@dataclass
class ExecResult:
    kind: str
    status: str  # ok | errored | timeout | invalid
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""
    experiment_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    log: str = ""


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def _dirs(ctx: Ctx) -> tuple[Path, Path]:
    token = secrets.token_hex(5)
    return ctx.run_dir / "attempts" / token, ctx.run_dir / "logs" / token


def _save_logs(log_dir: Path, files: dict[str, str]) -> None:
    # Logs live outside the container-writable attempt dir so a candidate can't plant symlinks there.
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (log_dir / name).write_text(text, encoding="utf-8")


def run_python(ctx: Ctx, code: str, purpose: str, *, hypothesis_id: str | None = None, attempt: int = 0,
               parent_experiment_id: str | None = None) -> ExecResult:
    """Run ``code`` as candidate.py through the user's evaluator."""
    cfg = ctx.spec.python_sandbox
    ctx.budget.charge_sandbox()
    attempt_dir, log_dir = _dirs(ctx)
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "candidate.py").write_text(code, encoding="utf-8")
    r = ctx.sandbox("python_sandbox").run(attempt_dir, cfg.evaluator)
    _save_logs(log_dir, {"candidate.py": code, "stdout.txt": r.stdout, "stderr.txt": r.stderr})
    eid = ctx.kb.add_experiment(ctx.run_id, purpose, r.status, r.metrics, hypothesis_id=hypothesis_id,
                                attempt=attempt, code_hash=code_hash(code), parent_experiment_id=parent_experiment_id,
                                artifacts_dir=str(log_dir), error=r.error, duration_s=r.duration_s)
    ctx.log("execute", f"{purpose} {eid}: {r.status} {r.error or _fmt(r.metrics)}")
    return ExecResult("python_sandbox", r.status, r.metrics, r.error, eid,
                      {"code": code, "code_hash": code_hash(code)}, (r.stderr or r.stdout)[-3000:])


def _fmt(m: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in list(m.items())[:8])


def _failed_block(failed: list[str]) -> str:
    if not failed:
        return ""
    lines = ["\nPrevious failed attempts for this hypothesis (avoid these errors):"]
    for i, err in enumerate(failed[-3:], 1):
        lines.append(f"Attempt {i}:\n<untrusted>\n{untrusted(err, 1500)}\n</untrusted>")
    return "\n".join(lines) + "\n"


def _hyp_text(h: dict) -> str:
    return (f"{h['statement']}\nRationale: {h.get('rationale', '')}\n"
            f"Falsified if: {h.get('falsification', '')}\nExpected effect: {h.get('expected_effect', '')}")


def execute(ctx: Ctx, hyp: dict, state: dict, failed: list[str], attempt: int) -> ExecResult:
    kind = hyp["test_kind"]
    ctx.tools.get(kind)  # allowlist gate, raises ToolNotAllowed
    if kind == "python_sandbox":
        return _exec_python(ctx, hyp, state, failed, attempt)
    if kind == "data_query":
        return _exec_data(ctx, hyp, failed, attempt)
    if kind == "literature":
        return _exec_literature(ctx, hyp)
    raise ValueError(f"unknown test kind {kind!r}")


def replicate(ctx: Ctx, hyp: dict, first: ExecResult) -> ExecResult:
    """Independent rerun of the same experiment (same code / query / papers)."""
    if first.kind == "python_sandbox":
        return run_python(ctx, first.payload["code"], "replication", hypothesis_id=hyp["id"],
                          parent_experiment_id=first.experiment_id)
    if first.kind == "data_query":
        return _run_query(ctx, hyp, first.payload["sql"], "replication", 0)
    return _classify(ctx, hyp, first.payload["papers"], "replication")


# -- python sandbox ---------------------------------------------------------------


def _exec_python(ctx: Ctx, hyp: dict, state: dict, failed: list[str], attempt: int) -> ExecResult:
    best = state.get("best") or {}
    parent_code = best.get("code") or ctx.spec.python_sandbox.baseline.read_text(encoding="utf-8")
    out = ctx.ask_text(
        "implement", goal=ctx.spec.goal, validation=ctx.spec.validation.describe(),
        hypothesis=untrusted(_hyp_text(hyp)), program=parent_code,
        parent_label=best.get("label", "baseline"), parent_metrics=_fmt(best.get("metrics") or {}) or "(none)",
        failed_attempts=_failed_block(failed), timeout_s=ctx.spec.python_sandbox.runtime.timeout_s)
    code = extract_code(out)
    err = ""
    if len(code.encode()) > MAX_CODE_BYTES:
        err = f"candidate is larger than {MAX_CODE_BYTES} bytes"
    elif code_hash(code) == code_hash(parent_code):
        err = "candidate is identical to its parent program; nothing was changed"
    else:
        try:
            compile(code, "candidate.py", "exec")  # parse only; nothing is executed on the host
        except SyntaxError as e:
            err = f"SyntaxError: {e.msg} (line {e.lineno})"
    if err:
        eid = ctx.kb.add_experiment(ctx.run_id, "candidate", "errored", {}, hypothesis_id=hyp["id"], attempt=attempt,
                                    code_hash=code_hash(code), parent_experiment_id=best.get("experiment_id"),
                                    error=err)
        ctx.log("execute", f"candidate {eid}: rejected before running: {err}")
        return ExecResult("python_sandbox", "errored", {}, err, eid, {"code": code, "code_hash": code_hash(code)})
    return run_python(ctx, code, "candidate", hypothesis_id=hyp["id"], attempt=attempt,
                      parent_experiment_id=best.get("experiment_id"))


# -- data query -------------------------------------------------------------------


def _evaluator_doc(path: Path) -> str:
    try:
        return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or "(no description)"
    except (SyntaxError, OSError):
        return "(no description)"


def _exec_data(ctx: Ctx, hyp: dict, failed: list[str], attempt: int) -> ExecResult:
    tool = ctx.tools.get("data_query")
    out = ctx.ask_json("data_query_sql", goal=ctx.spec.goal, validation=ctx.spec.validation.describe(),
                       hypothesis=untrusted(_hyp_text(hyp)), schema=untrusted(tool.schema(), 8000),
                       evaluator_doc=untrusted(_evaluator_doc(ctx.spec.data_query.evaluator), 4000),
                       failed_attempts=_failed_block(failed), max_rows=ctx.spec.data_query.max_rows)
    sql = str(out.get("sql", "")).strip() if isinstance(out, dict) else ""
    return _run_query(ctx, hyp, sql, "data", attempt)


def _run_query(ctx: Ctx, hyp: dict, sql: str, purpose: str, attempt: int) -> ExecResult:
    tool = ctx.tools.get("data_query")
    try:
        res = tool.query(sql)
    except QueryError as e:
        eid = ctx.kb.add_experiment(ctx.run_id, purpose, "errored", {}, hypothesis_id=hyp["id"], attempt=attempt,
                                    error=f"{e} | SQL: {sql[:500]}")
        ctx.log("execute", f"{purpose} {eid}: query rejected: {e}")
        return ExecResult("data_query", "errored", {}, f"{e}\nSQL: {sql[:1000]}", eid, {"sql": sql})
    ctx.budget.charge_sandbox()
    attempt_dir, log_dir = _dirs(ctx)
    attempt_dir.mkdir(parents=True)
    rows_json = json.dumps({"columns": res.columns, "rows": res.rows, "truncated": res.truncated,
                            "hypothesis": hyp["statement"]}, default=str)
    (attempt_dir / "rows.json").write_text(rows_json, encoding="utf-8")
    # The evaluator reads $HORIZONS_WORK/rows.json.
    r = ctx.sandbox("data_query").run(attempt_dir, ctx.spec.data_query.evaluator)
    _save_logs(log_dir, {"query.sql": sql, "stdout.txt": r.stdout, "stderr.txt": r.stderr})
    eid = ctx.kb.add_experiment(ctx.run_id, purpose, r.status, r.metrics, hypothesis_id=hyp["id"], attempt=attempt,
                                code_hash=code_hash(sql), artifacts_dir=str(log_dir), error=r.error,
                                duration_s=r.duration_s)
    ctx.log("execute", f"{purpose} {eid}: {len(res.rows)} rows -> {r.status} {r.error or _fmt(r.metrics)}")
    return ExecResult("data_query", r.status, r.metrics, r.error, eid, {"sql": sql}, (r.stderr or r.stdout)[-3000:])


# -- literature ---------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_quote(quote: str, paper: dict) -> bool:
    q = _norm(quote).strip(" .\"'")
    return len(q) >= 20 and q in _norm(f"{paper.get('title', '')} {paper.get('abstract', '')}")


def _exec_literature(ctx: Ctx, hyp: dict) -> ExecResult:
    lit = ctx.tools.get("literature")
    found = lit.search(hyp["statement"], min(10, ctx.spec.literature.max_papers))
    for p in found:
        ctx.kb.add_paper(ctx.run_id, p, hyp["statement"][:200])
    cited = set(hyp.get("citations") or [])
    pool = {p["id"]: p for p in found}
    for p in ctx.kb.run_papers(ctx.run_id):
        if p["id"] in cited:
            pool.setdefault(p["id"], p)
    papers = list(pool.values())[:15]
    return _classify(ctx, hyp, papers, "literature")


def _classify(ctx: Ctx, hyp: dict, papers: list[dict], purpose: str) -> ExecResult:
    if not papers:
        eid = ctx.kb.add_experiment(ctx.run_id, purpose, "ok", {"supporting_papers": 0, "challenging_papers": 0,
                                                                "evidence_balance": 0.0}, hypothesis_id=hyp["id"])
        return ExecResult("literature", "ok", {"supporting_papers": 0.0, "challenging_papers": 0.0,
                                               "evidence_balance": 0.0}, "", eid, {"papers": []},
                          "no papers found")
    out = ctx.ask_json("classify_evidence", hypothesis=untrusted(_hyp_text(hyp)),
                       papers=untrusted(format_papers(papers, 1500), 30000))
    by_id = {p["id"]: p for p in papers}
    sup = cha = unverified = 0
    judgements = []
    for j in (out.get("judgements") if isinstance(out, dict) else None) or []:
        if not isinstance(j, dict) or j.get("paper_id") not in by_id:
            continue
        rel = j.get("relation")
        if rel not in ("supports", "challenges"):
            continue
        quote = str(j.get("evidence_quote", ""))[:1000]
        if not verify_quote(quote, by_id[j["paper_id"]]):
            unverified += 1
            continue
        judgements.append((j["paper_id"], rel, quote, j.get("confidence", 0.5)))
    seen = set()
    for pid, rel, quote, conf in judgements:
        if pid in seen:
            continue
        seen.add(pid)
        if purpose == "literature":
            graph.link(ctx.kb, ctx.run_id, pid, hyp["id"], rel, conf, quote)
        sup += rel == "supports"
        cha += rel == "challenges"
    total = sup + cha
    metrics = {"supporting_papers": float(sup), "challenging_papers": float(cha),
               "evidence_balance": (sup - cha) / total if total else 0.0,
               "papers_considered": float(len(papers)), "unverified_quotes": float(unverified)}
    eid = ctx.kb.add_experiment(ctx.run_id, purpose, "ok", metrics, hypothesis_id=hyp["id"])
    ctx.log("execute", f"{purpose} {eid}: {sup} supporting, {cha} challenging, {unverified} unverified quotes dropped")
    return ExecResult("literature", "ok", metrics, "", eid, {"papers": papers},
                      "\n".join(f"{pid} {rel}: {q[:200]}" for pid, rel, q, _ in judgements))
