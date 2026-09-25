"""Small stdlib statistics toolkit: permutation test, bootstrap CI, Holm correction.

Deterministic given a seed so reports are reproducible.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from typing import Sequence


def mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else math.nan


def permutation_test(treatment: Sequence[float], control: Sequence[float], *, alternative: str = "greater",
                     n_resamples: int = 10_000, seed: int = 0) -> float:
    """p-value for the difference in means ``mean(treatment) - mean(control)``.

    ``alternative``: "greater" (treatment larger), "less", or "two-sided".
    Uses the exact permutation distribution when it is small (<= 20k splits),
    otherwise Monte Carlo with the +1 correction so p is never 0.
    """
    t, c = list(map(float, treatment)), list(map(float, control))
    if not t or not c:
        return 1.0
    pooled = t + c
    n_t = len(t)
    observed = mean(t) - mean(c)

    def extreme(d: float) -> bool:
        eps = 1e-12 * max(1.0, abs(observed))
        if alternative == "greater":
            return d >= observed - eps
        if alternative == "less":
            return d <= observed + eps
        return abs(d) >= abs(observed) - eps

    total = sum(pooled)
    n = len(pooled)
    if math.comb(n, n_t) <= 20_000:
        hits = count = 0
        for idx in itertools.combinations(range(n), n_t):
            s = sum(pooled[i] for i in idx)
            d = s / n_t - (total - s) / (n - n_t)
            hits += extreme(d)
            count += 1
        return hits / count
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_resamples):
        rng.shuffle(pooled)
        s = sum(pooled[:n_t])
        hits += extreme(s / n_t - (total - s) / (n - n_t))
    return (hits + 1) / (n_resamples + 1)


def paired_permutation_test(diffs: Sequence[float], *, alternative: str = "greater",
                            n_resamples: int = 10_000, seed: int = 0) -> float:
    """Sign-flip test for paired differences (treatment - control on the same unit).

    Under the null each difference is equally likely to have either sign. Exact
    when there are at most 2**14 sign patterns, otherwise Monte Carlo with +1.
    """
    d = [float(x) for x in diffs]
    if not d:
        return 1.0
    observed = mean(d)
    eps = 1e-12 * max(1.0, abs(observed))

    def extreme(m: float) -> bool:
        if alternative == "greater":
            return m >= observed - eps
        if alternative == "less":
            return m <= observed + eps
        return abs(m) >= abs(observed) - eps

    n = len(d)
    if n <= 14:
        hits = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            hits += extreme(sum(s * x for s, x in zip(signs, d)) / n)
        return hits / 2 ** n
    rng = random.Random(seed)
    hits = sum(extreme(sum(x if rng.random() < 0.5 else -x for x in d) / n) for _ in range(n_resamples))
    return (hits + 1) / (n_resamples + 1)


def bootstrap_ci(treatment: Sequence[float], control: Sequence[float] | None = None, *, level: float = 0.95,
                 n_resamples: int = 5_000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(treatment) (- mean(control) if given)."""
    t = list(map(float, treatment))
    c = list(map(float, control)) if control is not None else None
    if not t or (c is not None and not c):
        return (math.nan, math.nan)
    rng = random.Random(seed)
    stats_ = []
    for _ in range(n_resamples):
        m = mean([rng.choice(t) for _ in t])
        if c is not None:
            m -= mean([rng.choice(c) for _ in c])
        stats_.append(m)
    stats_.sort()
    lo_i = int(((1 - level) / 2) * (n_resamples - 1))
    hi_i = int((1 - (1 - level) / 2) * (n_resamples - 1))
    return (stats_[lo_i], stats_[hi_i])


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj
