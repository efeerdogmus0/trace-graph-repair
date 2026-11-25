"""Independent reference implementations used only for checking.

Deliberately written differently from `graphs.py` / `semantics.py` (label strings,
Floyd-Warshall, Bellman-Ford style relaxation, plain brute force) so that the dataset
validator and the tests do not simply re-run the generator's own code.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

LEdge = Tuple[str, str]


def closure_fw(nodes: Sequence[str], edges: Sequence[LEdge]) -> Dict[LEdge, bool]:
    """reach[(a, b)] is True iff there is a directed path with at least one edge from a to b."""
    reach = {(a, b): False for a in nodes for b in nodes}
    for a, b in edges:
        reach[(a, b)] = True
    for k in nodes:
        for i in nodes:
            if reach[(i, k)]:
                for j in nodes:
                    if reach[(k, j)]:
                        reach[(i, j)] = True
    return reach


def reachable(nodes: Sequence[str], edges: Sequence[LEdge], s: str, t: str) -> bool:
    return closure_fw(nodes, edges)[(s, t)]


def shortest_len(nodes: Sequence[str], edges: Sequence[LEdge], s: str, t: str) -> Optional[int]:
    """Fewest edges on a walk of length >= 1 from s to t, by repeated relaxation."""
    best: Dict[str, Optional[int]] = {n: None for n in nodes}
    for a, b in edges:
        if a == s:
            best[b] = 1
    for _ in range(len(nodes)):
        changed = False
        for a, b in edges:
            if best[a] is not None and (best[b] is None or best[a] + 1 < best[b]):
                best[b] = best[a] + 1
                changed = True
        if not changed:
            break
    return best[t]


def critical(nodes: Sequence[str], edges: Sequence[LEdge], s: str, t: str, edge: LEdge) -> bool:
    remaining = [e for e in edges if tuple(e) != tuple(edge)]
    return not reachable(nodes, remaining, s, t)


def constraints_hold(
    nodes: Sequence[str], edges: Sequence[LEdge], must_reach: Sequence[LEdge], must_not: Sequence[LEdge]
) -> bool:
    reach = closure_fw(nodes, edges)
    return all(reach[tuple(p)] for p in must_reach) and not any(reach[tuple(p)] for p in must_not)


def repair(
    nodes: Sequence[str],
    edges: Sequence[LEdge],
    must_reach: Sequence[LEdge],
    must_not: Sequence[LEdge],
    additions: Sequence[LEdge],
) -> Tuple[str, List[List[Tuple[str, str, str]]], Optional[int]]:
    """Brute-force one-edit repair. Returns (status, all minimal solutions, minimum edit count)."""
    edge_set: Set[LEdge] = {tuple(e) for e in edges}
    if constraints_hold(nodes, sorted(edge_set), must_reach, must_not):
        return "unchanged", [[]], 0
    solutions: List[List[Tuple[str, str, str]]] = []
    for a, b in sorted(tuple(x) for x in additions):
        if (a, b) in edge_set:
            continue
        if constraints_hold(nodes, sorted(edge_set | {(a, b)}), must_reach, must_not):
            solutions.append([("add_edge", a, b)])
    for a, b in sorted(edge_set):
        if constraints_hold(nodes, sorted(edge_set - {(a, b)}), must_reach, must_not):
            solutions.append([("remove_edge", a, b)])
    if solutions:
        return "edited", solutions, 1
    return "impossible", [], None
