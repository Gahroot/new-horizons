"""User-owned evaluator for the memory-reduction topic.

Loads candidate.py, calls nearest_neighbor_distances(points) on 700 fixed
random 3-D points, and reports:
  peak_memory_mb  - peak Python heap allocated during the call (tracemalloc)
  correct         - 1 if every distance matches an independent reference
  runtime_s       - wall time of the call (informational)

The engine never edits this file; it is mounted read-only in the sandbox.
"""

import json
import math
import os
import random
import runpy
import time
import tracemalloc

N = 700
# Take the per-run result marker out of the environment before any candidate code is loaded,
# so the candidate can't simply read it and print a forged result line.
MARKER = os.environ.pop("HORIZONS_RESULT_MARKER")


def reference(points):
    out = []
    for i, (xi, yi, zi) in enumerate(points):
        best = math.inf
        for j, (xj, yj, zj) in enumerate(points):
            if i != j:
                d = (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2
                if d < best:
                    best = d
        out.append(math.sqrt(best))
    return out


def main():
    rng = random.Random(1234)
    points = [(rng.random(), rng.random(), rng.random()) for _ in range(N)]
    expected = reference(points)
    candidate = runpy.run_path(os.environ["HORIZONS_CANDIDATE"])
    fn = candidate["nearest_neighbor_distances"]

    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(list(points))
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    try:
        out = [float(x) for x in out]
        correct = len(out) == N and all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                                        for a, b in zip(out, expected))
    except (TypeError, ValueError):
        correct = False
    metrics = {"peak_memory_mb": peak / 1e6, "correct": int(correct), "runtime_s": elapsed}
    print(MARKER + json.dumps(metrics))


if __name__ == "__main__":
    main()
