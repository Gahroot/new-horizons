"""User-owned evaluator for the sub-1-bit MoE expert compression topic.

Per run it draws 3 random "expert" layers (W, calibration activations, held-out activations) from
HORIZONS_SEED, a fresh random number the engine picks per run and reuses for that run's paired
baseline run (paired = true in topic.toml), so a candidate and the baseline face identical data.
The seed is removed from the environment and all data is drawn before any candidate code loads,
so the candidate cannot regenerate the held-out activations. For each layer it calls
candidate.compress(W, X_calib, max_bits), rebuilds W_hat itself from the fixed low-rank binary
format, counts the bits itself, and measures the output error on held-out activations.

Reports:
  output_error     - mean relative output error on held-out activations (lower is better)
  bits_per_weight  - worst bits per weight over the 3 layers, as charged by this evaluator
  valid            - 1 if every returned representation was well-formed
  runtime_s        - total compress() time (informational)

The engine never edits this file; it is mounted read-only in the sandbox.
"""

import json
import math
import os
import random
import runpy
import struct
import time
from operator import mul

# Remove the result marker and the seed before any candidate code is loaded.
MARKER = os.environ.pop("HORIZONS_RESULT_MARKER")
_SEED = os.environ.pop("HORIZONS_SEED", None)

M = N = 128           # expert matrix shape (out x in)
N_CALIB = 256         # calibration tokens shown to compress()
N_TEST = 256          # held-out tokens used for scoring
N_LAYERS = 3
BITS_PER_WEIGHT = 0.5
SCALE_BITS = 16       # a, b, c are stored as float16


class Invalid(Exception):
    pass


def make_layer(rng):
    """A small stand-in for one MoE expert: weights plus activations with outlier channels."""
    col_scale = [math.exp(rng.gauss(0.0, 0.5)) for _ in range(N)]
    for j in rng.sample(range(N), 3):            # outlier input channels in the weights
        col_scale[j] *= 4.0
    k = 8                                         # shared low-rank structure
    left = [[rng.gauss(0, 1) for _ in range(k)] for _ in range(M)]
    right = [[rng.gauss(0, 1) for _ in range(k)] for _ in range(N)]
    w = [[(0.02 * rng.gauss(0, 1) + 0.01 * sum(map(mul, left[i], right[j])) / math.sqrt(k))
          * col_scale[j] for j in range(N)] for i in range(M)]

    act_scale = [math.exp(rng.gauss(0.0, 0.5)) for _ in range(N)]
    for j in rng.sample(range(N), 4):            # activation outlier channels (as in real LLMs)
        act_scale[j] *= 15.0

    def acts(n):
        return [[rng.gauss(0, 1) * act_scale[j] for j in range(N)] for _ in range(n)]

    return w, acts(N_CALIB), acts(N_TEST)


def to_f16(x, what):
    try:
        return struct.unpack("<e", struct.pack("<e", float(x)))[0]
    except (OverflowError, TypeError, ValueError, struct.error) as exc:
        raise Invalid(f"{what} is not a finite float16 value: {x!r}") from exc


def signs(rows, n_rows, what):
    if not isinstance(rows, list) or len(rows) != n_rows:
        raise Invalid(f"{what} must be a list of {n_rows} rows")
    r = None
    out = []
    for row in rows:
        if not isinstance(row, list):
            raise Invalid(f"{what} rows must be lists")
        if r is None:
            r = len(row)
        if len(row) != r or any(type(v) is not int or v not in (-1, 1) for v in row):
            raise Invalid(f"{what} rows must all have the same length and contain only -1/+1 ints")
        out.append(list(row))
    return out, r or 0


def scales(values, n, what):
    if not isinstance(values, list) or len(values) != n:
        raise Invalid(f"{what} must be a list of {n} floats")
    out = [to_f16(v, what) for v in values]
    if not all(math.isfinite(v) for v in out):
        raise Invalid(f"{what} contains inf/nan")
    return out


def rebuild(rep):
    """W_hat = diag(a) U diag(c) V^T diag(b), with the evaluator's own bit accounting."""
    if not isinstance(rep, dict) or set(rep) != {"U", "V", "a", "b", "c"}:
        raise Invalid("compress() must return a dict with exactly the keys U, V, a, b, c")
    u, r_u = signs(rep["U"], M, "U")
    v, r_v = signs(rep["V"], N, "V")
    if r_u != r_v:
        raise Invalid(f"U has rank {r_u} but V has rank {r_v}")
    r = r_u
    a = scales(rep["a"], M, "a")
    b = scales(rep["b"], N, "b")
    c = scales(rep["c"], r, "c")
    bits = r * (M + N) + SCALE_BITS * (M + N + r)
    vc = [[v[j][k] * c[k] for k in range(r)] for j in range(N)]
    w_hat = [[a[i] * b[j] * sum(map(mul, u[i], vc[j])) for j in range(N)] for i in range(M)]
    return w_hat, bits / (M * N)


def output_energy(x, w):
    """sum over tokens and outputs of (x . w_i)^2."""
    total = 0.0
    for row in x:
        for wi in w:
            y = sum(map(mul, row, wi))
            total += y * y
    return total


def main():
    seed = int(_SEED) if _SEED is not None else int.from_bytes(os.urandom(8), "big")
    rng = random.Random(seed)
    layers = [make_layer(rng) for _ in range(N_LAYERS)]
    compress = runpy.run_path(os.environ["HORIZONS_CANDIDATE"])["compress"]
    max_bits = int(BITS_PER_WEIGHT * M * N)

    errors, bpws, elapsed, valid = [], [], 0.0, 1
    for w, x_calib, x_test in layers:
        t0 = time.perf_counter()
        rep = compress([row[:] for row in w], [row[:] for row in x_calib], max_bits)
        elapsed += time.perf_counter() - t0
        try:
            w_hat, bpw = rebuild(rep)
        except Invalid as exc:
            print(f"invalid representation: {exc}")
            valid = 0
            errors.append(1.0)
            bpws.append(float(M * N))
            continue
        diff = [[w[i][j] - w_hat[i][j] for j in range(N)] for i in range(M)]
        errors.append(output_energy(x_test, diff) / output_energy(x_test, w))
        bpws.append(bpw)

    metrics = {"output_error": sum(errors) / len(errors), "bits_per_weight": max(bpws),
               "valid": valid, "runtime_s": elapsed}
    print(MARKER + json.dumps(metrics))


if __name__ == "__main__":
    main()
