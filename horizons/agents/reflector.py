"""Reflect: store categorised lessons and decide REFINE / PIVOT / STOP.

Lesson categories and keyword classification follow AutoResearchClaw's
``evolution.py``. The model only explains *why* for refuted/inconclusive
outcomes; categories it returns are whitelisted, and the decision itself
is plain code.
"""

from __future__ import annotations

from horizons.agents.common import Ctx, untrusted

CATEGORIES = ("system", "experiment", "literature", "analysis")
_KEYWORDS = {
    "system": ("timeout", "timed out", "killed", "memory", "docker", "network", "permission", "sandbox",
               "exit code 137", "rate limit"),
    "experiment": ("syntax", "error", "traceback", "exception", "guard failed", "import", "identical", "metric"),
    "literature": ("paper", "citation", "search", "abstract"),
}


def classify_error(text: str) -> str:
    t = text.lower()
    best, score = "experiment", 0
    for cat, kws in _KEYWORDS.items():
        s = sum(kw in t for kw in kws)
        if s > score:
            best, score = cat, s
    return best


def decide(outcome: str, refines_used: int, max_refines: int) -> str:
    if outcome == "supported":
        return "STOP"
    if outcome == "errored" and refines_used < max_refines:
        return "REFINE"
    return "PIVOT"


def reflect(ctx: Ctx, hyp: dict, outcome: str, reason: str, log: str) -> tuple[list[str], str]:
    """Record lessons for this outcome. Returns (lesson texts, model explanation or '')."""
    spec = ctx.spec
    texts: list[tuple[str, str]] = []
    expl = ""
    if outcome == "errored":
        texts.append((classify_error(reason + " " + log[-500:]),
                      f"Experiment for '{hyp['statement'][:150]}' errored: {reason[:300]}"))
    elif outcome == "supported":
        texts.append(("analysis", f"Confirmed: '{hyp['statement'][:200]}' ({reason})"))
    else:
        default_cat = "literature" if hyp["test_kind"] == "literature" else "experiment"
        out = ctx.ask_json("reflect", goal=spec.goal, hypothesis=untrusted(hyp["statement"]), outcome=outcome,
                           evidence=reason, log=untrusted(log, 3000))
        out = out if isinstance(out, dict) else {}
        for l in (out.get("lessons") or [])[:4]:
            if isinstance(l, dict) and str(l.get("text", "")).strip():
                cat = l.get("category") if l.get("category") in CATEGORIES else default_cat
                texts.append((cat, str(l["text"]).strip()[:500]))
        expl = str(out.get("explanation", "")).strip()[:500]
        if not texts:
            texts.append((default_cat, f"'{hyp['statement'][:150]}' was {outcome}: {reason[:200]}. {expl}".strip()))
    for cat, text in texts:
        ctx.kb.add_lesson(ctx.run_id, spec.name, cat, text, hypothesis_id=hyp["id"], outcome=outcome)
    return [t for _, t in texts], expl
