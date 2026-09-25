"""Markdown run report.

All model- and paper-derived text is escaped for Markdown tables/HTML, and
code is placed in fences longer than any backtick run it contains, so a
hostile abstract or program can't restructure the report.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from horizons import stats
from horizons.kb.store import KB
from horizons.template import TopicSpec


def esc(text: Any, cap: int = 400) -> str:
    s = re.sub(r"\s+", " ", str(text if text is not None else "")).strip()
    if len(s) > cap:
        s = s[: cap - 1] + "…"
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("|", "\\|").replace("`", "\\`").replace("[", "\\[").replace("]", "\\]"))


def fence(code: str, lang: str = "") -> str:
    longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
    f = "`" * max(3, longest + 1)
    return f"{f}{lang}\n{code.rstrip()}\n{f}"


def _num(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float)) else "-"


def build_report(kb: KB, run_id: str, spec: TopicSpec) -> str:
    run = kb.get_run(run_id)
    st = run["state"] or {}
    v = spec.validation
    base = st.get("baseline") or {}
    best = st.get("best") or {}
    hyps = kb.hypotheses(run_id)
    budget = run["budget"] or {}
    L: list[str] = []
    L.append(f"# New Horizons report - {esc(spec.name)}")
    L.append("")
    L.append(f"- **Run:** `{run_id}` · **Status:** {run['status'].upper()} · **LLM:** {esc(run['llm'])}")
    L.append(f"- **Goal:** {esc(spec.goal, 1000)}")
    L.append(f"- **Validation:** {esc(v.describe(), 600)}")
    L.append(f"- **Stop reason:** {esc(run['stop_reason'] or 'still running', 600)}")
    L.append(f"- **Spend:** {budget.get('iterations', 0)}/{spec.budget.max_iterations} iterations · "
             f"{budget.get('llm_calls', 0)}/{spec.budget.max_llm_calls} LLM calls · "
             f"{budget.get('sandbox_runs', 0)}/{spec.budget.max_sandbox_runs} sandbox runs · "
             f"{budget.get('elapsed_s', 0) / 60:.1f} min")
    L.append(f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")

    L.append("## Result")
    L.append("")
    breakthroughs = kb.findings(run_id, "breakthrough")
    if run["status"] == "success" and breakthroughs:
        b = breakthroughs[-1]
        vd = b["data"].get("verdict", {})
        L.append(f"**Target met and replicated.** {esc(b['summary'], 800)}")
        L.append("")
        L.append(f"- {v.metric}: {_num(vd.get('value'))} (runs: {', '.join(_num(x) for x in vd.get('samples', []))})"
                 + (f" vs baseline {_num(base.get('mean'))}" if base else ""))
        if base.get("mean"):
            L.append(f"- Change vs baseline: {(vd.get('value', 0) - base['mean']) / abs(base['mean']):+.1%}")
        if vd.get("p_value") is not None:
            L.append(f"- Permutation test p = {_num(vd['p_value'])}, Holm-adjusted p = {_num(vd.get('p_adjusted'))}")
        if vd.get("ci"):
            L.append(f"- 95% bootstrap CI of the difference vs baseline: [{_num(vd['ci'][0])}, {_num(vd['ci'][1])}]")
    else:
        L.append("**Target NOT met.** The numbers below are the best result so far, not a confirmed discovery.")
        L.append("")
        if best:
            L.append(f"- Best so far: {esc(best.get('summary', ''), 500)}")
    if base:
        L.append(f"- Baseline {v.metric}: {_num(base.get('mean'))} over {len(base.get('samples', []))} run(s)")
    L.append("")

    bmaps = kb.findings(run_id, "boundary_map")
    if bmaps:
        bm = bmaps[-1]["data"]
        L.append("## Boundary map (what is already known)")
        L.append("")
        L.append(esc(bm.get("summary", ""), 2000))
        L.append("")
        for title, key, field in (("Known approaches", "known_approaches", "approach"),
                                  ("Promising directions", "promising_directions", "direction")):
            items = bm.get(key) or []
            if items:
                L.append(f"**{title}**")
                L.append("")
                for it in items:
                    cites = ", ".join(f"`{esc(c, 80)}`" for c in it.get("citations", []))
                    L.append(f"- {esc(it.get(field), 500)}" + (f" ({cites})" if cites else ""))
                L.append("")
        if bm.get("gaps"):
            L.append("**Gaps**")
            L.append("")
            L.extend(f"- {esc(g, 500)}" for g in bm["gaps"])
            L.append("")

    tested = [h for h in hyps if h["status"] in ("supported", "refuted", "errored", "inconclusive")]
    if tested:
        L.append("## Hypotheses tested")
        L.append("")
        L.append(f"| # | ID | Tool | Outcome | {esc(v.metric, 40)} | Reason | Statement |")
        L.append("|---|---|---|---|---|---|---|")
        for h in tested:
            o = h["outcome"] or {}
            L.append(f"| {h['iteration']} | `{h['id']}` | {h['test_kind']} | **{h['status']}** | {_num(o.get('value'))} "
                     f"| {esc(o.get('reason'), 160)} | {esc(h['statement'], 240)} |")
        L.append("")
        ps = [(h["id"], (h["outcome"] or {}).get("p_value")) for h in tested]
        ps = [(i, p) for i, p in ps if p is not None]
        if v.kind == "significance" and ps:
            adj = stats.holm([p for _, p in ps])
            L.append(f"**Multiple-testing check (Holm, all {len(ps)} tests in this run, alpha {v.alpha}):** "
                     + "; ".join(f"`{i}` p={_num(p)} -> {_num(a)}{' ✓' if a < v.alpha else ''}"
                                 for (i, p), a in zip(ps, adj)))
            L.append("")

    needs = [h for h in hyps if h["status"] == "needs_resources"]
    if needs:
        L.append("## Ideas that need resources this topic does not allow")
        L.append("")
        L.extend(f"- {esc(h['statement'], 300)} - *missing:* {esc(h['missing'], 200)}" for h in needs)
        L.append("")

    papers = {p["id"]: p for p in kb.run_papers(run_id)}
    edges = [e for e in kb.edges(run_id) if e["type"] in ("supports", "challenges") and e["src"] in papers]
    if edges:
        L.append("## Literature evidence (quotes verified against abstracts)")
        L.append("")
        for e in edges:
            p = papers.get(e["src"], {})
            L.append(f"- `{esc(e['dst'], 40)}` **{e['type']}** by {esc(p.get('title', e['src']), 200)} "
                     f"({esc(p.get('url', ''), 200)}): \"{esc(e['rationale'], 400)}\"")
        L.append("")

    if best.get("code") and best.get("label") != "baseline":
        L.append(f"## Best program ({esc(best.get('label'))})")
        L.append("")
        L.append(fence(best["code"], "python"))
        L.append("")

    lessons = kb.lessons(run_id=run_id)
    if lessons:
        L.append("## Lessons stored in memory")
        L.append("")
        for cat in ("experiment", "analysis", "literature", "system"):
            items = [l for l in lessons if l["category"] == cat]
            if items:
                L.append(f"**{cat}**")
                L.append("")
                L.extend(f"- {esc(l['text'], 500)}" for l in items)
                L.append("")

    L.append("## Limits of this result")
    L.append("")
    L.append("- Success is decided by code comparing the user-owned evaluator's numbers with the template, "
             "never by the model.")
    L.append("- Candidate code runs in the same container process as the evaluator; a deliberately adversarial "
             "candidate could forge output. Review the best program before trusting it.")
    if run["llm"].startswith("scripted"):
        L.append("- This run used the **scripted** offline LLM: it proves the machinery works, not research quality.")
    L.append(f"- Papers retrieved: {len(kb.run_papers(run_id))}. Abstract-level evidence only; full texts were not read.")
    L.append("")
    return "\n".join(L)


def write_report(kb: KB, run_id: str, spec: TopicSpec, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "report.md"
    path.write_text(build_report(kb, run_id, spec), encoding="utf-8")
    return path
