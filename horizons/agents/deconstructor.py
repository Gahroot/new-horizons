"""Deconstruct: literature queries -> papers -> boundary map with verified citations."""

from __future__ import annotations

from typing import Any

from horizons.agents.common import Ctx, untrusted


def _str_list(v: Any, cap: int, item_cap: int = 500) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x)[:item_cap].strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:cap]


def _cited_items(v: Any, key: str, known: set[str], dropped: list[str], cap: int = 12) -> list[dict]:
    out = []
    for it in v if isinstance(v, list) else []:
        if not isinstance(it, dict) or not str(it.get(key, "")).strip():
            continue
        cites = _str_list(it.get("citations"), 10, 200)
        good = [c for c in cites if c in known]
        dropped.extend(c for c in cites if c not in known)
        out.append({key: str(it[key])[:600], "citations": good})
    return out[:cap]


def format_papers(papers: list[dict], abstract_cap: int = 600) -> str:
    return "\n".join(f"- {p['id']} - {p['title']} ({p.get('year') or 'n.d.'}) - {(p.get('abstract') or '')[:abstract_cap]}"
                     for p in papers) or "(no papers)"


def gather_literature(ctx: Ctx, n_queries: int = 4) -> list[dict]:
    lit = ctx.tools.get("literature")
    out = ctx.ask_json("deconstruct_queries", goal=ctx.spec.goal, context=ctx.spec.context or "(none)",
                       validation=ctx.spec.validation.describe(), n=n_queries)
    queries = _str_list(out.get("queries") if isinstance(out, dict) else out, n_queries, 200)
    if not queries:
        queries = [ctx.spec.goal[:200]]
    per_query = max(3, ctx.spec.literature.max_papers // len(queries))
    total = 0
    for q in queries:
        if total >= ctx.spec.literature.max_papers:
            break
        found = lit.search(q, per_query)
        for p in found[: ctx.spec.literature.max_papers - total]:
            ctx.kb.add_paper(ctx.run_id, p, q)
            total += 1
        ctx.log("deconstruct", f"query {q!r}: {len(found)} papers")
    return ctx.kb.run_papers(ctx.run_id)


def deconstruct(ctx: Ctx, baseline: dict | None, refresh: bool = False) -> dict:
    """Build (or refresh) the boundary map and store it as a finding."""
    papers: list[dict] = []
    if ctx.spec.allows("literature"):
        papers = gather_literature(ctx, n_queries=4 if not refresh else 2)
        if not papers:
            ctx.kb.add_lesson(ctx.run_id, ctx.spec.name, "literature",
                              "Literature search returned no papers; the boundary map relies on the goal text only.",
                              outcome="no_papers")
    papers = papers[: ctx.spec.literature.max_papers] if ctx.spec.literature else []
    known = {p["id"] for p in papers}
    if refresh:
        lessons = ctx.kb.lessons(run_id=ctx.run_id)[-10:]
        extra = "\nLessons so far:\n" + "\n".join(f"- [{l['category']}] {l['text']}" for l in lessons)
    else:
        extra = ""
    out = ctx.ask_json("boundary_map", goal=ctx.spec.goal, validation=ctx.spec.validation.describe(),
                       baseline=_baseline_str(baseline),
                       papers=untrusted(format_papers(papers) + extra, 30000),
                       tools=ctx.tools_str())
    out = out if isinstance(out, dict) else {}
    dropped: list[str] = []
    bmap = {
        "summary": str(out.get("summary", ""))[:2000],
        "known_approaches": _cited_items(out.get("known_approaches"), "approach", known, dropped),
        "gaps": _str_list(out.get("gaps"), 12),
        "promising_directions": _cited_items(out.get("promising_directions"), "direction", known, dropped),
        "papers": len(papers),
        "refresh": refresh,
    }
    if dropped:
        uniq = sorted(set(dropped))
        ctx.log("deconstruct", f"stripped {len(uniq)} citation(s) not found in the knowledge base")
        ctx.kb.add_lesson(ctx.run_id, ctx.spec.name, "literature",
                          f"Model cited {len(uniq)} unknown paper ID(s) in the boundary map; they were removed: "
                          + ", ".join(uniq[:5]), outcome="citation_stripped")
    ctx.kb.add_finding(ctx.run_id, "boundary_map", bmap["summary"] or "boundary map", bmap,
                       topic_name=ctx.spec.name)
    return bmap


def _baseline_str(baseline: dict | None) -> str:
    if not baseline:
        return "(no baseline program for this topic)"
    return f"{ctx_metric_line(baseline)}"


def ctx_metric_line(b: dict) -> str:
    m = b.get("metrics") or {}
    return ", ".join(f"{k}={v:g}" for k, v in m.items()) or "(none)"
