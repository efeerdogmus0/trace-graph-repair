"""Small directed simple graphs with bitmask adjacency.

Scope of the first version (see docs/protocol.md): no self-loops, no parallel edges and
no antiparallel pairs (u->v together with v->u), so that every drawn arrow is unambiguous.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

Edge = Tuple[int, int]


def bits(mask: int) -> List[int]:
    out = []
    i = 0
    while mask:
        if mask & 1:
            out.append(i)
        mask >>= 1
        i += 1
    return out


class Graph:
    """Immutable directed graph on nodes 0..n-1."""

    __slots__ = ("n", "adj", "_closure")

    def __init__(self, n: int, edges: Iterable[Edge]) -> None:
        if n < 1:
            raise ValueError("graph needs at least one node")
        adj = [0] * n
        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError("edge endpoint out of range: %r" % ((u, v),))
            if u == v:
                raise ValueError("self-loops are out of scope")
            adj[u] |= 1 << v
        self.n = n
        self.adj: Tuple[int, ...] = tuple(adj)
        self._closure: Optional[Tuple[int, ...]] = None

    # -- structure ---------------------------------------------------------------
    @property
    def edges(self) -> List[Edge]:
        return [(u, v) for u in range(self.n) for v in bits(self.adj[u])]

    @property
    def edge_count(self) -> int:
        return sum(bin(a).count("1") for a in self.adj)

    def has_edge(self, u: int, v: int) -> bool:
        return bool(self.adj[u] >> v & 1)

    def can_add(self, u: int, v: int) -> bool:
        """An edge may be added if it keeps the graph simple and free of antiparallel pairs."""
        return u != v and not self.has_edge(u, v) and not self.has_edge(v, u)

    def has_antiparallel(self) -> bool:
        return any(self.has_edge(v, u) for u, v in self.edges)

    def with_edge(self, u: int, v: int) -> "Graph":
        if self.has_edge(u, v):
            raise ValueError("edge already present")
        return Graph(self.n, self.edges + [(u, v)])

    def without_edge(self, u: int, v: int) -> "Graph":
        if not self.has_edge(u, v):
            raise ValueError("edge not present")
        return Graph(self.n, [e for e in self.edges if e != (u, v)])

    def relabeled(self, perm: Sequence[int]) -> "Graph":
        """Graph with node i renamed to perm[i]."""
        return Graph(self.n, [(perm[u], perm[v]) for u, v in self.edges])

    def in_degrees(self) -> List[int]:
        deg = [0] * self.n
        for _, v in self.edges:
            deg[v] += 1
        return deg

    def out_degrees(self) -> List[int]:
        return [bin(a).count("1") for a in self.adj]

    # -- reachability ------------------------------------------------------------
    def closure(self) -> Tuple[int, ...]:
        """closure()[u] = bitmask of nodes reachable from u using at least one edge."""
        if self._closure is None:
            out = []
            adj = self.adj
            for u in range(self.n):
                seen = 0
                frontier = adj[u]
                while frontier:
                    seen |= frontier
                    nxt = 0
                    f = frontier
                    i = 0
                    while f:
                        if f & 1:
                            nxt |= adj[i]
                        f >>= 1
                        i += 1
                    frontier = nxt & ~seen
                out.append(seen)
            self._closure = tuple(out)
        return self._closure

    def reaches(self, s: int, t: int) -> bool:
        return bool(self.closure()[s] >> t & 1)

    def shortest_path_len(self, s: int, t: int) -> Optional[int]:
        """Number of edges on a shortest s->t path (>= 1), or None if unreachable."""
        seen = 0
        frontier = self.adj[s]
        dist = 1
        while frontier:
            if frontier >> t & 1:
                return dist
            seen |= frontier
            nxt = 0
            for i in bits(frontier):
                nxt |= self.adj[i]
            frontier = nxt & ~seen
            dist += 1
        return None

    def is_weakly_connected(self) -> bool:
        und = list(self.adj)
        for u, v in self.edges:
            und[v] |= 1 << u
        seen = 1
        frontier = 1
        while frontier:
            nxt = 0
            for i in bits(frontier):
                nxt |= und[i]
            frontier = nxt & ~seen
            seen |= frontier
        return seen == (1 << self.n) - 1

    # -- misc --------------------------------------------------------------------
    def key(self) -> Tuple[int, Tuple[int, ...]]:
        return (self.n, self.adj)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Graph) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return "Graph(%d, %r)" % (self.n, self.edges)
