"""Hybrid memory recall: FTS5/BM25 arm + hashed-vector arm fused with RRF.

Adapted from Hindsight's ``engine/search/fusion.py``: each arm is capped to
its own top-N before fusion (so one arm can't crowd the other out), then
``score(d) = sum(1 / (k + rank_arm(d)))`` with k = 60.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from horizons.kb import vectors
from horizons.kb.store import KB

RRF_K = 60


@dataclass
class Hit:
    kind: str
    ref_id: str
    text: str
    run_id: str | None
    score: float
    arms: dict[str, int] = field(default_factory=dict)  # arm -> rank


def _filters(kinds: tuple[str, ...] | None, topic: str | None) -> tuple[str, list]:
    clauses, args = [], []
    if kinds:
        clauses.append(f"m.kind IN ({', '.join('?' for _ in kinds)})")
        args.extend(kinds)
    if topic:
        clauses.append("(m.topic_name = ? OR m.topic_name IS NULL)")
        args.append(topic)
    return (" AND " + " AND ".join(clauses)) if clauses else "", args


def bm25_arm(kb: KB, query: str, kinds: tuple[str, ...] | None, topic: str | None, cap: int) -> list[dict]:
    toks = vectors.tokens(query)[:32]
    if not toks:
        return []
    # Quote every token: FTS5 query syntax from untrusted text can't inject operators.
    match = " OR ".join('"' + t.replace('"', "") + '"' for t in toks)
    where, args = _filters(kinds, topic)
    sql = ("SELECT m.id, m.kind, m.ref_id, m.run_id, m.text FROM memory_fts f JOIN memory m ON m.id = f.rowid"
           f" WHERE memory_fts MATCH ?{where} ORDER BY bm25(memory_fts) LIMIT ?")
    return [dict(r) for r in kb.conn.execute(sql, (match, *args, cap)).fetchall()]


def vector_arm(kb: KB, query: str, kinds: tuple[str, ...] | None, topic: str | None, cap: int,
               min_sim: float = 0.05) -> list[dict]:
    q = vectors.embed(query)
    where, args = _filters(kinds, topic)
    scored = []
    for r in kb.conn.execute(f"SELECT m.id, m.kind, m.ref_id, m.run_id, m.text, m.vec FROM memory m WHERE 1=1{where}",
                             args):
        s = vectors.cosine(q, vectors.from_blob(r["vec"]))
        if s >= min_sim:
            d = dict(r)
            d.pop("vec")
            d["sim"] = s
            scored.append(d)
    scored.sort(key=lambda d: d["sim"], reverse=True)
    return scored[:cap]


def rrf(arms: dict[str, list[dict]], k: int = RRF_K) -> list[Hit]:
    hits: dict[int, Hit] = {}
    for name, results in arms.items():
        for rank, r in enumerate(results, start=1):
            h = hits.get(r["id"])
            if h is None:
                h = hits[r["id"]] = Hit(r["kind"], r["ref_id"], r["text"], r.get("run_id"), 0.0)
            h.score += 1.0 / (k + rank)
            h.arms[name] = rank
    return sorted(hits.values(), key=lambda h: h.score, reverse=True)


def recall(kb: KB, query: str, *, kinds: tuple[str, ...] | None = None, topic: str | None = None,
           top_k: int = 8, per_arm_cap: int = 30) -> list[Hit]:
    arms = {
        "vector": vector_arm(kb, query, kinds, topic, per_arm_cap),
        "bm25": bm25_arm(kb, query, kinds, topic, per_arm_cap),
    }
    return rrf(arms)[:top_k]


def max_similarity(kb: KB, text: str, ref_ids: list[str], kind: str = "hypothesis") -> tuple[float, str | None]:
    """Highest cosine similarity between ``text`` and the given memory items."""
    if not ref_ids:
        return 0.0, None
    q = vectors.embed(text)
    best, best_id = 0.0, None
    qs = ", ".join("?" for _ in ref_ids)
    for r in kb.conn.execute(f"SELECT ref_id, vec FROM memory WHERE kind = ? AND ref_id IN ({qs})", (kind, *ref_ids)):
        s = vectors.cosine(q, vectors.from_blob(r["vec"]))
        if s > best:
            best, best_id = s, r["ref_id"]
    return best, best_id
