"""Evaluate: controller-side verdicts. The model is never consulted here.

Outcome classes (kept distinct on purpose):
  errored      - the experiment itself failed (crash, timeout, invalid output,
                 broken guard such as wrong results) -> REFINE the experiment
  refuted      - the experiment ran and the metric missed the target -> PIVOT
  inconclusive - promising but not established (did not replicate, not
                 significant after Holm correction) -> PIVOT / gather more
  supported    - target met on every replication (and significant, if the
                 template asks for significance) -> breakthrough, STOP

Paired mode ([validation] paired = true): each candidate run has a baseline run on
the same seed. The target is checked per pair, the significance test uses the
paired differences, and the reported value is seed-adjusted: the baseline mean
times the average candidate/baseline ratio (plain differences when baseline values
are not all positive). This removes the run-to-run noise of a randomised evaluator
from the comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from horizons import stats
from horizons.template import TopicSpec, Validation


@dataclass
class Verdict:
    outcome: str
    reason: str
    value: float | None = None
    samples: list[float] = field(default_factory=list)
    target_met: bool = False
    guards_ok: bool = False
    improved: bool = False
    p_value: float | None = None
    p_adjusted: float | None = None
    ci: tuple[float, float] | None = None
    paired_baselines: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["ci"] = list(self.ci) if self.ci else None
        return d


def metric_value(v: Validation, metrics: dict[str, float]) -> float | None:
    x = metrics.get(v.metric)
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def target_value(v: Validation, baseline_mean: float | None) -> float | None:
    if v.target_relative is not None and baseline_mean is not None:
        return baseline_mean * (1 + v.target_relative)
    return v.target_absolute


def meets_target(v: Validation, value: float, baseline_mean: float | None) -> bool:
    t = target_value(v, baseline_mean)
    if t is None:  # significance-only validation: any improvement vs baseline counts
        return baseline_mean is not None and better(v, value, baseline_mean)
    return value <= t if v.direction == "minimize" else value >= t


def better(v: Validation, a: float, b: float | None) -> bool:
    if b is None:
        return True
    eps = 1e-9 * max(1.0, abs(b))
    return a < b - eps if v.direction == "minimize" else a > b + eps


def guards_ok(v: Validation, metrics: dict[str, float]) -> tuple[bool, str]:
    for g in v.guards:
        if not g.check(metrics):
            return False, f"guard failed: {g.describe()} (got {metrics.get(g.metric)!r})"
    return True, ""


def screen(spec: TopicSpec, status: str, metrics: dict[str, float], error: str,
           baseline_mean: float | None) -> tuple[str, str]:
    """First-pass check of one run. Returns (next_step, reason) where next_step is
    'errored' | 'refuted' | 'replicate'."""
    v = spec.validation
    if status != "ok":
        return "errored", error or status
    ok, why = guards_ok(v, metrics)
    if not ok:
        return "errored", why
    val = metric_value(v, metrics)
    if val is None:
        return "errored", f"evaluator did not report metric {v.metric!r}"
    if v.kind == "significance":
        if baseline_mean is not None and not better(v, val, baseline_mean):
            return "refuted", f"{v.metric}={val:g} is not better than baseline {baseline_mean:g}"
        return "replicate", ""
    if not meets_target(v, val, baseline_mean):
        t = target_value(v, baseline_mean)
        return "refuted", f"{v.metric}={val:g} missed target {t:g}" if t is not None else "missed target"
    return "replicate", ""


def paired_diffs(samples: list[float], refs: list[float], baseline_mean: float | None) -> list[float]:
    """Per-seed effect of the candidate, in the metric's units at the baseline's average level.

    With positive baseline values the effect is relative (candidate / baseline on that seed,
    rescaled to the baseline mean), so a hard seed where every number is large cannot make
    the same percentage gain look bigger. Otherwise it is the plain difference.
    """
    if baseline_mean is not None and baseline_mean > 0 and all(r > 0 for r in refs):
        return [baseline_mean * (s / r - 1.0) for s, r in zip(samples, refs)]
    return [s - r for s, r in zip(samples, refs)]


def judge(spec: TopicSpec, runs: list[tuple[str, dict[str, float]]], baseline_samples: list[float],
          best_value: float | None, prior_pvalues: list[float],
          paired_baselines: list[float | None] | None = None) -> Verdict:
    """Final verdict over the first run + replications. ``runs`` = [(status, metrics)].

    ``paired_baselines`` (paired mode) holds the baseline's metric on the same seed as each run;
    a missing pair makes that run invalid.
    """
    v = spec.validation
    baseline_mean = stats.mean(baseline_samples) if baseline_samples else None
    paired = paired_baselines is not None
    if paired and len(paired_baselines) != len(runs):
        raise ValueError("paired_baselines must have one entry per run")
    samples, refs, all_guards, all_target = [], [], True, True
    for i, (status, m) in enumerate(runs):
        val = metric_value(v, m) if status == "ok" else None
        ref = paired_baselines[i] if paired else baseline_mean
        if val is None or (paired and ref is None):
            all_guards = all_target = False
            continue
        samples.append(val)
        refs.append(ref)
        all_guards &= guards_ok(v, m)[0]
        all_target &= meets_target(v, val, ref)
    diffs = paired_diffs(samples, refs, baseline_mean) if paired else []
    if not samples:
        value = None
    elif paired and baseline_mean is not None:
        value = baseline_mean + stats.mean(diffs)
    else:
        value = stats.mean(samples)
    verdict = Verdict("refuted", "", value=value, samples=samples, guards_ok=all_guards and bool(samples),
                      target_met=all_target and len(samples) == len(runs) and bool(samples),
                      paired_baselines=refs if paired else [])
    verdict.improved = bool(samples) and verdict.guards_ok and better(v, value, best_value)
    alt = "less" if v.direction == "minimize" else "greater"
    if paired:
        if len(diffs) >= 2:
            verdict.ci = stats.bootstrap_ci(diffs)
        if diffs:
            verdict.p_value = stats.paired_permutation_test(diffs, alternative=alt)
            verdict.p_adjusted = stats.holm([*prior_pvalues, verdict.p_value])[-1]
    else:
        if len(samples) >= 2 and len(baseline_samples) >= 2:
            verdict.ci = stats.bootstrap_ci(samples, baseline_samples)
        if baseline_samples and samples:
            verdict.p_value = stats.permutation_test(samples, baseline_samples, alternative=alt)
            verdict.p_adjusted = stats.holm([*prior_pvalues, verdict.p_value])[-1]

    n, k = len(runs), len(samples)
    if not verdict.guards_ok or k < n:
        verdict.outcome = "inconclusive"
        verdict.reason = f"only {k}/{n} runs produced valid results that passed the guards"
        return verdict
    if v.kind == "significance":
        if verdict.p_adjusted is None:
            verdict.outcome, verdict.reason = "inconclusive", "no baseline samples to test against"
        elif not better(v, value, baseline_mean):
            verdict.outcome, verdict.reason = "refuted", "no improvement over baseline across replications"
        elif verdict.p_adjusted >= v.alpha:
            verdict.outcome = "inconclusive"
            verdict.reason = (f"improvement not significant: p={verdict.p_value:.4g}, "
                              f"Holm-adjusted {verdict.p_adjusted:.4g} >= alpha {v.alpha}")
        elif (v.target_relative is not None or v.target_absolute is not None) and not meets_target(
                v, value, baseline_mean):
            verdict.outcome, verdict.reason = "refuted", "significant, but mean missed the target"
        else:
            verdict.outcome = "supported"
            verdict.reason = f"significant: Holm-adjusted p={verdict.p_adjusted:.4g} < {v.alpha} over {n} runs"
        return verdict
    pair_note = " (each against the baseline on the same seed)" if paired else ""
    if verdict.target_met:
        verdict.outcome = "supported"
        verdict.reason = f"target met on all {n} run(s){pair_note}"
    else:
        met = sum(meets_target(v, s, r) for s, r in zip(samples, refs))
        verdict.outcome = "inconclusive" if met else "refuted"
        verdict.reason = (f"target met on {met}/{n} runs{pair_note} - did not replicate" if met
                          else f"target missed{pair_note}")
    return verdict
