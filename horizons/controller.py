"""The discovery loop: a checkpointed state machine.

    baseline -> deconstruct -> [hypothesize -> execute(+evaluate, reflect)]* -> done

Decisions follow AutoResearchClaw's explicit REFINE / PIVOT / STOP states:
  errored             -> REFINE the same hypothesis (max ``max_refines``)
  refuted/inconclusive-> PIVOT to a new hypothesis; after ``max_pivots`` pivots
                         without improvement, refresh the boundary map once
  supported           -> STOP (success; replicated breakthrough)
  budget / patience   -> STOP (best-so-far, labelled as not meeting target)

State is written to SQLite after every step so ``--resume`` continues a run.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

from horizons import stats
from horizons.agents import deconstructor, executor, hypothesizer, reflector
from horizons.agents import evaluator as ev
from horizons.agents.common import Ctx
from horizons.budget import BudgetExceeded, BudgetGuard
from horizons.kb import graph
from horizons.kb.store import KB
from horizons.llm import LLM, LLMClient
from horizons.report import write_report
from horizons.template import TopicSpec
from horizons.tools.data_query import DataQueryTool
from horizons.tools.literature import LiteratureTool
from horizons.tools.registry import ToolRegistry
from horizons.tools.sandbox import make_sandbox

MAX_REDECONSTRUCTS = 2


class RunError(RuntimeError):
    pass


class Controller:
    def __init__(self, spec: TopicSpec, kb: KB, client: LLMClient, workspace: Path, *, offline: bool = False,
                 accept_unsafe_local: bool = False, debug: bool = False, out: Callable[[str], None] = print):
        self.spec, self.kb, self.client = spec, kb, client
        self.workspace = workspace.resolve()
        self.offline = offline
        self.accept_unsafe_local = accept_unsafe_local
        self.debug = debug
        self.out = out
        self.run_id: str | None = None
        self.state: dict[str, Any] = {}
        self.budget = BudgetGuard(spec.budget)
        self.ctx: Ctx | None = None

    # -- setup -------------------------------------------------------------------
    def _log(self, stage: str, msg: str) -> None:
        self.out(f"[{stage}] {msg}")
        if self.run_id:
            self.kb.event(self.run_id, stage, msg)

    def _build_ctx(self) -> Ctx:
        spec = self.spec
        run_dir = self.workspace / "runs" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        factories: dict[str, Callable[[], Any]] = {}
        if spec.literature:
            fixture = LiteratureTool.load_fixture(spec.literature.fixture) if spec.literature.fixture else []
            factories["literature"] = lambda: LiteratureTool(self.kb, spec.literature, self.offline, fixture,
                                                             lambda m: self._log("literature", m))
        if spec.python_sandbox:
            factories["python_sandbox"] = lambda: make_sandbox(spec.python_sandbox.runtime, spec.root,
                                                               self.workspace, self.accept_unsafe_local)
        if spec.data_query:
            factories["data_query"] = lambda: DataQueryTool(spec.data_query)
        registry = ToolRegistry(spec, factories)

        def sandbox_for(tool: str) -> Any:
            registry.get(tool)  # allowlist gate
            if tool == "python_sandbox":
                return registry.get("python_sandbox")
            if tool == "data_query":
                return make_sandbox(spec.data_query.runtime, spec.root, self.workspace, self.accept_unsafe_local)
            raise RunError(f"tool {tool!r} has no sandbox")

        llm = LLM(self.client, self.budget,
                  lambda task, system, user, output: self.kb.log_llm(self.run_id, task, len(system) + len(user), output))
        return Ctx(spec, self.kb, llm, self.budget, registry, self.run_id, run_dir, sandbox_for, self._log)

    def _checkpoint(self, **kw: Any) -> None:
        self.kb.save_run(self.run_id, state=self.state, budget=self.budget.to_dict(),
                         stage=self.state.get("phase", "?"), **kw)

    def start(self) -> str:
        self.run_id = self.kb.create_run(self.spec.name, str(self.spec.path), self.spec.goal,
                                         f"{self.client.name}:{self.client.model}")
        self.state = {"phase": "baseline", "iteration": 0, "since_improvement": 0, "pivots_since_improvement": 0,
                      "redeconstructs": 0, "pvalues": [], "best": {}, "baseline": None, "boundary_map": None,
                      "current": None}
        self.ctx = self._build_ctx()
        self._checkpoint()
        self._log("run", f"started {self.run_id} for topic {self.spec.name!r} with {self.client.name}:{self.client.model}")
        return self.run_id

    def resume(self, run_id: str) -> str:
        row = self.kb.get_run(run_id)
        if row is None:
            raise RunError(f"no run {run_id!r} in {self.kb.path}")
        if row["topic_name"] != self.spec.name:
            raise RunError(f"run {run_id} belongs to topic {row['topic_name']!r}, not {self.spec.name!r}")
        if row["status"] == "success":
            raise RunError(f"run {run_id} already succeeded; nothing to resume")
        self.run_id = run_id
        self.state = row["state"]
        self.budget = BudgetGuard.from_dict(self.spec.budget, row["budget"])
        self.ctx = self._build_ctx()
        self.kb.save_run(run_id, status="running")
        self._log("run", f"resuming {run_id} at phase {self.state.get('phase')}")
        return run_id

    # -- main loop -----------------------------------------------------------------
    def run(self) -> str:
        assert self.ctx is not None, "call start() or resume() first"
        status = "running"
        try:
            while self.state["phase"] != "done":
                self.budget.check_clock()
                getattr(self, f"_phase_{self.state['phase']}")()
                self._checkpoint()
            status = self.kb.get_run(self.run_id)["status"]
        except BudgetExceeded as e:
            status = self._finish("stopped", str(e))
        except KeyboardInterrupt:
            status = self._finish("stopped", "interrupted by user", done=False)
        except Exception as e:  # noqa: BLE001 - any failure must be recorded and leave a resumable run
            self._log("error", "".join(traceback.format_exception_only(type(e), e)).strip())
            if self.debug:
                traceback.print_exc()
            status = self._finish("failed", f"{type(e).__name__}: {e}", done=False)
        path = write_report(self.kb, self.run_id, self.spec, self.ctx.run_dir)
        self._log("report", f"written to {path}")
        return status

    def _finish(self, status: str, reason: str, done: bool = True) -> str:
        if done:
            self.state["phase"] = "done"
        self._checkpoint(status=status, stop_reason=reason)
        self._log("run", f"{status}: {reason}")
        return status

    # -- phases ----------------------------------------------------------------------
    def _phase_baseline(self) -> None:
        spec, ctx = self.spec, self.ctx
        # Prerequisites fail closed before any money is spent on the LLM.
        for tool in ("python_sandbox", "data_query"):
            if spec.allows(tool):
                ctx.sandbox(tool)
        if spec.python_sandbox:
            code = spec.python_sandbox.baseline.read_text(encoding="utf-8")
            runs = [executor.run_python(ctx, code, "baseline") for _ in range(spec.validation.replications)]
            bad = [r for r in runs if r.status != "ok"]
            if bad:
                raise RunError(f"the unmodified baseline failed in the evaluator ({bad[0].status}: {bad[0].error}). "
                               f"Fix baseline/evaluator first; logs in {self.ctx.run_dir / 'logs'}")
            samples = [ev.metric_value(spec.validation, r.metrics) for r in runs]
            if any(s is None for s in samples):
                raise RunError(f"evaluator did not report the validation metric {spec.validation.metric!r} "
                               f"(got: {', '.join(runs[0].metrics)})")
            keys = runs[0].metrics.keys()
            avg = {k: stats.mean([r.metrics.get(k, 0.0) for r in runs]) for k in keys}
            ok, why = ev.guards_ok(spec.validation, runs[0].metrics)
            if not ok:
                self._log("baseline", f"warning: baseline itself fails a guard ({why})")
            mean = stats.mean(samples)
            self.state["baseline"] = {"samples": samples, "mean": mean, "metrics": avg,
                                      "experiment_id": runs[0].experiment_id}
            self.state["best"] = {"label": "baseline", "code": code, "metrics": avg, "value": mean,
                                  "experiment_id": runs[0].experiment_id, "hypothesis_id": None,
                                  "summary": f"baseline {spec.validation.metric}={mean:g}"}
            target = ev.target_value(spec.validation, mean)
            self._log("baseline", f"{spec.validation.metric} = {mean:g} over {len(samples)} run(s)"
                      + (f"; target {target:g}" if target is not None else ""))
        self.state["phase"] = "deconstruct"

    def _phase_deconstruct(self) -> None:
        refresh = self.state["redeconstructs"] > 0
        self._log("deconstruct", "refreshing boundary map with lessons so far" if refresh else "mapping the field")
        self.state["boundary_map"] = deconstructor.deconstruct(self.ctx, self.state.get("baseline"), refresh)
        self._log("deconstruct", self.state["boundary_map"]["summary"][:300] or "(no summary)")
        self.state["phase"] = "hypothesize"

    def _phase_hypothesize(self) -> None:
        b = self.spec.budget
        if self.state["since_improvement"] >= b.patience:
            self._finish("stopped", f"stalled: no improvement in {b.patience} consecutive iterations")
            return
        self.budget.charge_iteration()
        self.state["iteration"] += 1
        it = self.state["iteration"]
        hid = hypothesizer.propose(self.ctx, it, self.state)
        if hid is None:
            self.state["since_improvement"] += 1
            self._log("hypothesize", f"iteration {it}: no new testable hypothesis")
            return
        h = self.kb.get_hypothesis(hid)
        self._log("hypothesize", f"iteration {it}: selected {hid} [{h['test_kind']}] {h['statement'][:160]}")
        self.state["current"] = {"hid": hid, "refines": 0, "failed": []}
        self.state["phase"] = "execute"

    def _phase_execute(self) -> None:
        spec, ctx, st = self.spec, self.ctx, self.state
        cur = st["current"]
        hyp = self.kb.get_hypothesis(cur["hid"])
        base = st.get("baseline") or {}
        base_samples = base.get("samples") or []
        base_mean = base.get("mean")
        best = st.get("best") or {}

        res = executor.execute(ctx, hyp, st, cur["failed"], cur["refines"])
        step, reason = ev.screen(spec, res.status, res.metrics, res.error, base_mean)
        runs = [(res.status, res.metrics)]
        if step == "errored":
            verdict = ev.Verdict("errored", reason)
        else:
            if step == "replicate" and spec.validation.replications > 1:
                self._log("evaluate", f"{hyp['id']} passed first run; replicating x{spec.validation.replications - 1}")
                for _ in range(spec.validation.replications - 1):
                    r = executor.replicate(ctx, hyp, res)
                    runs.append((r.status, r.metrics))
            prior = [p["p"] for p in st["pvalues"]]
            verdict = ev.judge(spec, runs, base_samples, best.get("value"), prior)
            if step == "refuted" and verdict.outcome == "supported":  # screen and judge disagree -> be conservative
                verdict.outcome = "refuted"
            if verdict.p_value is not None and spec.validation.kind == "significance":
                st["pvalues"].append({"hid": hyp["id"], "p": verdict.p_value})
        self._log("evaluate", f"{hyp['id']}: {verdict.outcome.upper()} - {verdict.reason}"
                  + (f" ({spec.validation.metric}={verdict.value:g})" if verdict.value is not None else ""))

        improved = verdict.outcome != "errored" and verdict.improved
        if improved:
            fid = self.kb.add_finding(self.run_id, "result",
                                      f"{hyp['statement'][:200]} -> {spec.validation.metric}={verdict.value:g}",
                                      {"verdict": verdict.to_dict(), "metrics": res.metrics},
                                      hypothesis_id=hyp["id"], experiment_id=res.experiment_id,
                                      topic_name=spec.name)
            if best.get("finding_id"):
                graph.link(self.kb, self.run_id, fid, best["finding_id"], "updates", 1.0, "new best result")
            st["best"] = {"label": f"{hyp['id']}", "code": res.payload.get("code") or best.get("code"),
                          "metrics": res.metrics, "value": verdict.value, "experiment_id": res.experiment_id,
                          "hypothesis_id": hyp["id"], "finding_id": fid,
                          "summary": f"{hyp['id']}: {spec.validation.metric}={verdict.value:g} ({hyp['statement'][:120]})"}
            self._log("evaluate", f"new best: {spec.validation.metric}={verdict.value:g}")
        if hyp.get("parent_id"):
            graph.link(self.kb, self.run_id, hyp["id"], hyp["parent_id"], "derived_from", 1.0, "built on best program")

        lessons, expl = reflector.reflect(ctx, hyp, verdict.outcome, verdict.reason, res.log)
        self.kb.update_hypothesis(hyp["id"], status=verdict.outcome,
                                  outcome={**verdict.to_dict(), "experiment_id": res.experiment_id,
                                           "explanation": expl, "attempts": cur["refines"] + 1})
        decision = reflector.decide(verdict.outcome, cur["refines"], spec.budget.max_refines)
        self._log("reflect", f"{decision}" + (f" - lesson: {lessons[0][:160]}" if lessons else ""))

        if decision == "STOP":
            fid = self.kb.add_finding(self.run_id, "breakthrough",
                                      f"{hyp['statement'][:300]} ({verdict.reason})",
                                      {"verdict": verdict.to_dict(), "baseline_mean": base_mean,
                                       "metrics": res.metrics}, hypothesis_id=hyp["id"],
                                      experiment_id=res.experiment_id, topic_name=spec.name)
            graph.link(self.kb, self.run_id, fid, hyp["id"], "supports", 1.0, verdict.reason)
            st["success_hid"] = hyp["id"]
            self._finish("success", f"target met and replicated by {hyp['id']}: {verdict.reason}")
            return
        if decision == "REFINE":
            cur["refines"] += 1
            cur["failed"].append(f"{verdict.reason}\n{res.log[-1200:]}")
            return
        # PIVOT
        if improved:
            st["since_improvement"] = 0
            st["pivots_since_improvement"] = 0
        else:
            st["since_improvement"] += 1
            st["pivots_since_improvement"] += 1
        st["current"] = None
        st["phase"] = "hypothesize"
        b = spec.budget
        if (b.max_pivots and st["pivots_since_improvement"] >= b.max_pivots
                and st["redeconstructs"] < MAX_REDECONSTRUCTS):
            st["redeconstructs"] += 1
            st["pivots_since_improvement"] = 0
            st["phase"] = "deconstruct"
            self._log("reflect", f"{b.max_pivots} pivots without improvement: refreshing the boundary map")
