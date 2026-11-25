"""Label-independent topology comparison for split-leakage control.

`fingerprint` is an isomorphism-invariant hash (equal for isomorphic graphs, usually
different otherwise). It is only used to bucket candidates; `is_isomorphic` then decides
exactly by backtracking. A fingerprint match alone is never treated as proof.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

from .graphs import Graph, bits


def refined_colors(g: Graph) -> List[int]:
    """Colour refinement (1-WL) on a directed graph.

    Colours are dense ranks of sorted signatures, so the result does not depend on node
    numbering: an isomorphism always maps a node to a node of equal colour.
    """
    n = g.n
    preds: List[List[int]] = [[] for _ in range(n)]
    succs: List[List[int]] = [bits(a) for a in g.adj]
    for u in range(n):
        for v in succs[u]:
            preds[v].append(u)
    sig0 = [(len(succs[u]), len(preds[u])) for u in range(n)]
    order = sorted(set(sig0))
    colors = [order.index(s) for s in sig0]
    for _ in range(n):
        sigs = [
            (
                colors[u],
                tuple(sorted(colors[v] for v in succs[u])),
                tuple(sorted(colors[v] for v in preds[u])),
            )
            for u in range(n)
        ]
        order2 = sorted(set(sigs))
        new = [order2.index(s) for s in sigs]
        if len(order2) == len(set(colors)):
            colors = new
            break
        colors = new
    return colors


def fingerprint(g: Graph) -> str:
    colors = refined_colors(g)
    edge_cols = sorted((colors[u], colors[v]) for u, v in g.edges)
    payload = repr((g.n, g.edge_count, sorted(colors), edge_cols)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:20]


def find_isomorphism(g1: Graph, g2: Graph) -> Optional[List[int]]:
    """Return a mapping m with m[u] = image of g1's node u in g2, or None."""
    if g1.n != g2.n or g1.edge_count != g2.edge_count:
        return None
    n = g1.n
    c1, c2 = refined_colors(g1), refined_colors(g2)
    if sorted(c1) != sorted(c2):
        return None
    by_color: Dict[int, List[int]] = {}
    for v in range(n):
        by_color.setdefault(c2[v], []).append(v)

    # Order g1's nodes: rare colours first, then nodes touching already-placed ones.
    class_size = {c: len(vs) for c, vs in by_color.items()}
    und = [g1.adj[u] for u in range(n)]
    for u, v in g1.edges:
        und[v] |= 1 << u
    order: List[int] = []
    placed = 0
    remaining = set(range(n))
    while remaining:
        best = min(
            remaining,
            key=lambda u: (0 if und[u] & placed else 1, class_size[c1[u]], u),
        )
        order.append(best)
        placed |= 1 << best
        remaining.discard(best)

    mapping = [-1] * n
    used = [False] * n

    def consistent(u: int, v: int) -> bool:
        for w in range(n):
            mw = mapping[w]
            if mw < 0:
                continue
            if g1.has_edge(u, w) != g2.has_edge(v, mw):
                return False
            if g1.has_edge(w, u) != g2.has_edge(mw, v):
                return False
        return True

    def extend(i: int) -> bool:
        if i == n:
            return True
        u = order[i]
        for v in by_color.get(c1[u], ()):
            if used[v] or not consistent(u, v):
                continue
            mapping[u] = v
            used[v] = True
            if extend(i + 1):
                return True
            mapping[u] = -1
            used[v] = False
        return False

    return list(mapping) if extend(0) else None


def is_isomorphic(g1: Graph, g2: Graph) -> bool:
    return find_isomorphism(g1, g2) is not None


class TopologyRegistry:
    """Remembers graphs by group so that no two groups share a topology."""

    def __init__(self) -> None:
        self._buckets: Dict[str, List[Tuple[str, Graph]]] = {}

    def find_other_group(self, g: Graph, group_id: str) -> Optional[str]:
        """Group id of an isomorphic graph that belongs to a different group, if any."""
        for other_group, other in self._buckets.get(fingerprint(g), ()):
            if other_group != group_id and is_isomorphic(g, other):
                return other_group
        return None

    def add(self, g: Graph, group_id: str) -> None:
        self._buckets.setdefault(fingerprint(g), []).append((group_id, g))

    def __len__(self) -> int:
        return sum(len(v) for v in self._buckets.values())
