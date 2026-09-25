"""Hypothesize: multi-perspective candidates -> schema check -> dedupe vs memory -> judge -> one pick.

Candidate generation + judge ranking follows AutoResearchClaw's synthesis
stage; the hypothesis record (falsification, alternatives, readiness,
missing resources) follows FAROS research contracts.
"""

from __future__ import annotations

from typing import Any

from horizons.agents.common import TESTED_STATUSES, Ctx, untrusted
from horizons.kb.recall import max_similarity, recall
from horizons.kb.vectors import cosine, embed

DUPLICATE_SIM = 0.9


def _clip(v: Any, cap: int) -> str:
    return str(v or "").strip()[:cap]


def sanitize(raw: Any, allowed_tools: tuple[str, ...], known_papers: set[str]) -> list[dict]:
    """Keep only well-formed hypotheses; drop unknown fields and unknown citations."""
    items = raw.get("hypotheses") if isinstance(raw, dict) else raw
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        statement = _clip(it.get("statement"), 1000)
        if len(statement) < 10:
            continue
        test_kind = _clip(it.get("test_kind"), 40)
        readiness = "needs_resources" if it.get("readiness") == "needs_resources" else "testable"
        missing = _clip(it.get("missing"), 500) or None
        if test_kind not in allowed_tools:
            readiness = "needs_resources"
            missing = missing or f"tool {test_kind or '(none)'!r} is not allowed for this topic"
            test_kind = test_kind if test_kind in ("literature", "python_sandbox", "data_query") else "unknown"
        cites = it.get("citations") if isinstance(it.get("citations"), list) else []
        alts = it.get("alternatives") if isinstance(it.get("alternatives"), list) else []
        out.append({
            "statement": statement,
            "rationale": _clip(it.get("rationale"), 2000),
            "test_kind": test_kind,
            "falsification": _clip(it.get("falsification"), 1000),
            "expected_effect": _clip(it.get("expected_effect"), 500),
            "citations": [str(c) for c in cites if str(c) in known_papers][:10],
            "alternatives": [_clip(a, 300) for a in alts if isinstance(a, str)][:5],
            "readiness": readiness,
            "missing": missing if readiness == "needs_resources" else None,
        })
    return out


def _tested_summary(hyps: list[dict]) -> str:
    rows = [f"- [{h['status']}] {h['statement']}" for h in hyps if h["status"] in TESTED_STATUSES]
    return "\n".join(rows[-20:]) or "(none yet)"


def propose(ctx: Ctx, iteration: int, state: dict) -> str | None:
    """Generate, filter and rank hypotheses; return the selected hypothesis id (or None)."""
    spec = ctx.spec
    run_hyps = ctx.kb.hypotheses(ctx.run_id)
    best = state.get("best") or {}
    lesson_hits = recall(ctx.kb, f"{spec.goal} {best.get('summary', '')}", kinds=("lesson",), topic=spec.name, top_k=8)
    known = ctx.kb.paper_ids(ctx.run_id)
    raw = ctx.ask_json(
        "hypothesize", goal=spec.goal, validation=spec.validation.describe(), tools=ctx.tools_str(),
        boundary_map=untrusted(state.get("boundary_map") or {}),
        best=best.get("summary", "baseline only"),
        tested=untrusted(_tested_summary(run_hyps)),
        lessons=untrusted("\n".join(f"- {h.text}" for h in lesson_hits) or "(none)"),
        paper_ids=", ".join(sorted(known)[:60]) or "(none)",
        n=spec.budget.hypotheses_per_round)
    cands = sanitize(raw, spec.allowed_tools, known)

    # Dedupe against already-tested hypotheses of this run and within the batch.
    tested_ids = [h["id"] for h in run_hyps if h["status"] in TESTED_STATUSES]
    kept: list[dict] = []
    for c in cands:
        sim, sim_id = max_similarity(ctx.kb, c["statement"], tested_ids)
        if sim > DUPLICATE_SIM:
            ctx.log("hypothesize", f"dropped near-duplicate of {sim_id} (sim {sim:.2f}): {c['statement'][:80]}")
            continue
        v = embed(c["statement"])
        if any(cosine(v, embed(k["statement"])) > DUPLICATE_SIM for k in kept):
            continue
        kept.append(c)

    testable: list[tuple[dict, str]] = []
    for c in kept:
        if c["readiness"] == "needs_resources":
            c["status"] = "needs_resources"
            hid = ctx.kb.add_hypothesis(ctx.run_id, iteration, c)
            ctx.log("hypothesize", f"{hid} needs resources ({c['missing']}): {c['statement'][:80]}")
        else:
            testable.append((c, ""))
    if not testable:
        return None

    scores = [5.0] * len(testable)
    if len(testable) > 1:
        listing = "\n".join(f"{i + 1}. [{c['test_kind']}] {c['statement']}\n   falsified if: {c['falsification']}"
                            for i, (c, _) in enumerate(testable))
        ranking = ctx.ask_json("judge", goal=spec.goal, validation=spec.validation.describe(),
                               candidates=untrusted(listing))
        for r in (ranking.get("ranking") if isinstance(ranking, dict) else None) or []:
            if isinstance(r, dict) and isinstance(r.get("index"), int) and 1 <= r["index"] <= len(testable):
                try:
                    scores[r["index"] - 1] = max(0.0, min(10.0, float(r.get("score", 5))))
                except (TypeError, ValueError):
                    pass
    pick = max(range(len(testable)), key=lambda i: (scores[i], -i))
    parent = best.get("hypothesis_id")
    root = (ctx.kb.get_hypothesis(parent) or {}).get("lineage_root") if parent else None
    selected = None
    for i, (c, _) in enumerate(testable):
        c["score"] = scores[i]
        c["status"] = "selected" if i == pick else "not_selected"
        hid = ctx.kb.add_hypothesis(ctx.run_id, iteration, c, parent_id=parent if i == pick else None,
                                    lineage_root=root if i == pick else None)
        if i == pick:
            selected = hid
    return selected
