"""Finding graph with whitelisted edge types and per-node link caps.

Adapted from PRAXIST's ``finding_graph_mvp``: edge types stay in a known
set (model output can't invent new ones), confidence is clamped to [0, 1],
duplicate (src, dst, type) edges are ignored, and each source node has a cap
so a single noisy step can't flood the graph.
"""

from __future__ import annotations

from horizons.kb.store import KB

EDGE_TYPES = frozenset({"supports", "challenges", "derived_from", "updates", "related_to"})
MAX_LINKS_PER_NODE = 20


class EdgeError(ValueError):
    pass


def link(kb: KB, run_id: str, src: str, dst: str, type_: str, confidence: float = 1.0,
         rationale: str = "") -> bool:
    """Insert an edge. Returns False if it was a duplicate or the node is at its cap."""
    if type_ not in EDGE_TYPES:
        raise EdgeError(f"edge type must be one of {sorted(EDGE_TYPES)}, got {type_!r}")
    if src == dst:
        raise EdgeError("self-loops are not allowed")
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.5
    conf = min(1.0, max(0.0, conf))
    if kb.count_edges_from(src) >= MAX_LINKS_PER_NODE:
        kb.event(run_id, "graph", f"edge cap reached for {src}; dropped {type_} -> {dst}")
        return False
    return kb.insert_edge(run_id, src, dst, type_, conf, (rationale or "")[:1000])
