"""Local hashed embeddings (feature hashing) - no model download, deterministic.

Tokens and word bigrams are hashed into ``DIMS`` buckets with a signed hash,
then L2-normalised. Good enough for near-duplicate detection and a semantic-ish
recall arm; swap for provider embeddings later if needed.
"""

from __future__ import annotations

import hashlib
import math
import re
from array import array

DIMS = 512
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "we our can not no than then into using use via per".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _bucket(feature: str) -> tuple[int, float]:
    h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    n = int.from_bytes(h, "little")
    return n % DIMS, (1.0 if (n >> 63) & 1 else -1.0)


def embed(text: str) -> list[float]:
    v = [0.0] * DIMS
    toks = tokens(text)
    feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    for f in feats:
        i, s = _bucket(f)
        v[i] += s
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are L2-normalised


def to_blob(v: list[float]) -> bytes:
    return array("f", v).tobytes()


def from_blob(b: bytes) -> list[float]:
    a = array("f")
    a.frombytes(b)
    return a.tolist()
