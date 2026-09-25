"""Calibration-aware refinement of a fixed-budget binary factorisation."""

import math
from operator import mul

POWER_ITERS = 12
REFINEMENT_PASSES = 5
MAX_CALIBRATION_ROWS = 160


def compress(W, X_calib, max_bits):
    m = len(W)
    n = len(W[0])
    r = max(0, (max_bits - 16 * (m + n)) // (m + n + 16))

    a = [max(sum(abs(x) for x in row) / n, 1e-4) for row in W]
    U = [[] for _ in range(m)]
    V = [[] for _ in range(n)]
    c = []

    # Construct a feasible factorisation before refining it. Retain the
    # baseline's bit allocation and representation.
    res = [[x / ai for x in row] for row, ai in zip(W, a)]
    for _ in range(r):
        cols = list(zip(*res))
        v = [1.0 + 0.01 * j for j in range(n)]
        for _ in range(POWER_ITERS):
            u = [sum(map(mul, row, v)) for row in res]
            v = [sum(map(mul, col, u)) for col in cols]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            v = [x / norm for x in v]

        u = [sum(map(mul, row, v)) for row in res]
        su = [1 if x >= 0 else -1 for x in u]
        sv = [1 if x >= 0 else -1 for x in v]
        ck = sum(
            su[i] * sum(map(mul, res[i], sv)) for i in range(m)
        ) / (m * n)

        for i in range(m):
            row = res[i]
            f = ck * su[i]
            for j in range(n):
                row[j] -= f * sv[j]
            U[i].append(su[i])
        for j in range(n):
            V[j].append(sv[j])
        c.append(ck)

    if not r:
        return {"U": U, "V": V, "a": a, "b": [1.0] * n, "c": c}

    # Columns of the calibration matrix are input coordinates. If calibration
    # is unavailable or has an unexpected shape, use the Frobenius objective.
    try:
        samples = [list(x) for x in X_calib if len(x) == n]
    except (TypeError, ValueError):
        samples = []
    if not samples:
        samples = [
            [1.0 if j == i else 0.0 for j in range(n)]
            for i in range(n)
        ]
    elif len(samples) > MAX_CALIBRATION_ROWS:
        count = MAX_CALIBRATION_ROWS
        samples = [
            samples[(q * len(samples)) // count] for q in range(count)
        ]

    t = len(samples)
    xcols = [tuple(sample[j] for sample in samples) for j in range(n)]
    xnorm = [sum(x * x for x in col) for col in xcols]
    target = [
        [sum(W[i][j] * samples[s][j] for j in range(n))
         for s in range(t)]
        for i in range(m)
    ]
    z = [
        [sum(V[j][k] * samples[s][j] for j in range(n))
         for s in range(t)]
        for k in range(r)
    ]
    # E is the current error on calibration outputs.
    E = [
        [target[i][s] -
         a[i] * sum(U[i][k] * c[k] * z[k][s] for k in range(r))
         for s in range(t)]
        for i in range(m)
    ]

    for _ in range(REFINEMENT_PASSES):
        changed = False
        for k in range(r):
            ck = c[k]
            zk = z[k]
            znorm = sum(value * value for value in zk)
            if znorm <= 1e-24 or abs(ck) <= 1e-24:
                continue

            # Flip row signs only when doing so reduces calibrated output
            # error, accounting for the row's scale.
            for i in range(m):
                ui = U[i][k]
                ai = a[i]
                row = E[i]
                correlation = sum(row[s] * zk[s] for s in range(t))
                delta = (4.0 * ai * ck * ui * correlation +
                         4.0 * ai * ai * ck * ck * znorm)
                if delta < -1e-12:
                    U[i][k] = -ui
                    shift = 2.0 * ai * ck * ui
                    for s in range(t):
                        row[s] += shift * zk[s]
                    changed = True

            # Aggregate the residual seen by this component. A column-sign
            # flip can then be evaluated without revisiting every output row.
            weighted_signs = [a[i] * U[i][k] for i in range(m)]
            sign_norm = sum(w * w for w in weighted_signs)
            aggregate = [
                sum(weighted_signs[i] * E[i][s] for i in range(m))
                for s in range(t)
            ]
            dz = [0.0] * t
            for j in range(n):
                vj = V[j][k]
                col = xcols[j]
                correlation = sum(aggregate[s] * col[s] for s in range(t))
                delta = (4.0 * ck * vj * correlation +
                         4.0 * ck * ck * sign_norm * xnorm[j])
                if delta < -1e-12:
                    V[j][k] = -vj
                    step = -2.0 * vj
                    aggregate_step = -ck * sign_norm * step
                    for s in range(t):
                        amount = step * col[s]
                        dz[s] += amount
                        aggregate[s] += aggregate_step * amount
                    changed = True

            if any(dz):
                for s in range(t):
                    zk[s] += dz[s]
                for i in range(m):
                    row = E[i]
                    scale = -weighted_signs[i] * ck
                    for s in range(t):
                        row[s] += scale * dz[s]

            # Least-squares component coefficient, with all other
            # components held fixed.
            znorm = sum(value * value for value in zk)
            denominator = sign_norm * znorm
            if denominator > 1e-24:
                numerator = sum(
                    weighted_signs[i] *
                    sum(E[i][s] * zk[s] for s in range(t))
                    for i in range(m)
                )
                new_c = ck + numerator / denominator
                if new_c < 0.0:
                    new_c = -new_c
                    for i in range(m):
                        U[i][k] = -U[i][k]
                difference = new_c - ck
                if difference:
                    for i in range(m):
                        row = E[i]
                        scale = -a[i] * U[i][k] * difference
                        for s in range(t):
                            row[s] += scale * zk[s]
                    c[k] = new_c
                    changed = True

        # Refit the already-budgeted row scales to the calibrated outputs.
        for i in range(m):
            row = E[i]
            predicted = [target[i][s] - row[s] for s in range(t)]
            denominator = sum(value * value for value in predicted)
            if denominator <= 1e-24:
                continue
            ratio = sum(
                target[i][s] * predicted[s] for s in range(t)
            ) / denominator
            new_a = max(1e-8, a[i] * ratio)
            if new_a != a[i]:
                adjustment = 1.0 - new_a / a[i]
                for s in range(t):
                    row[s] += adjustment * predicted[s]
                a[i] = new_a
                changed = True

        if not changed:
            break

    return {"U": U, "V": V, "a": a, "b": [1.0] * n, "c": c}
