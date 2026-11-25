"""Graph families (generation distributions).

`graph_family` names a whole distribution; it is not the leakage unit. The leakage unit
is `graph_group_id` (one root graph plus all of its derived variants).

Every generator returns a weakly connected graph without antiparallel pairs, with node
ids randomly permuted so that numbering carries no structural information.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Set, Tuple

from .graphs import Edge, Graph
from .rng import Rng


def _finish(n: int, edges: Set[Edge], rng: Rng) -> Graph:
    perm = rng.shuffled(range(n))
    return Graph(n, sorted((perm[u], perm[v]) for u, v in edges))


def _try_add(edges: Set[Edge], u: int, v: int) -> bool:
    if u == v or (u, v) in edges or (v, u) in edges:
        return False
    edges.add((u, v))
    return True


def _components(n: int, edges: Set[Edge]) -> List[List[int]]:
    """Weakly connected components, each sorted, ordered by smallest member."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in sorted(edges):
        parent[find(u)] = find(v)
    groups: Dict[int, List[int]] = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    return sorted(groups.values(), key=lambda g: g[0])


def _fill_random(n: int, edges: Set[Edge], target: int, rng: Rng, tries: int = 400) -> None:
    while len(edges) < target and tries > 0:
        tries -= 1
        _try_add(edges, rng.below(n), rng.below(n))


def layered_dag(n: int, rng: Rng) -> Graph:
    """Nodes in 3-5 layers; edges only go from an earlier layer to a later one."""
    n_layers = rng.randint(3, max(3, min(5, n - 1)))
    sizes = [1] * n_layers
    for _ in range(n - n_layers):
        sizes[rng.below(n_layers)] += 1
    layers: List[List[int]] = []
    k = 0
    for s in sizes:
        layers.append(list(range(k, k + s)))
        k += s
    edges: Set[Edge] = set()
    for i in range(1, n_layers):
        for v in layers[i]:
            edges.add((rng.choice(layers[i - 1]), v))
    has_out = {u for u, _ in edges}
    for i in range(n_layers - 1):
        for u in layers[i]:
            # First-layer nodes have no incoming edge, so they always need an outgoing one.
            if u not in has_out and (rng.chance(0.7) or i == 0):
                edges.add((u, rng.choice(layers[i + 1])))
    target = int(round(n * rng.uniform(1.1, 1.6)))
    tries = 300
    while len(edges) < target and tries > 0:
        tries -= 1
        i = rng.below(n_layers - 1)
        if i + 2 < n_layers and rng.chance(0.25):
            j = rng.randint(i + 2, n_layers - 1)
        else:
            j = i + 1
        edges.add((rng.choice(layers[i]), rng.choice(layers[j])))
    # Parent links alone can leave several separate strands; join them with forward edges.
    layer_of = {u: i for i, nodes in enumerate(layers) for u in nodes}
    while True:
        comps = _components(n, edges)
        if len(comps) == 1:
            break
        first, second = comps[0], comps[1]
        joins = sorted(
            (u, v) if layer_of[u] < layer_of[v] else (v, u)
            for u in first for v in second if layer_of[u] != layer_of[v]
        )
        edges.add(rng.choice(joins))
    return _finish(n, edges, rng)


def tree_plus(n: int, rng: Rng) -> Graph:
    """Random out-tree with zero to three extra edges in any direction."""
    edges: Set[Edge] = set()
    for v in range(1, n):
        if rng.chance(0.5):
            parent = rng.below(v)
        else:
            parent = max(0, v - 1 - rng.below(min(v, 2)))
        edges.add((parent, v))
    extra = rng.randint(0, 3)
    _fill_random(n, edges, len(edges) + extra, rng)
    return _finish(n, edges, rng)


def sparse_random(n: int, rng: Rng) -> Graph:
    """Random spanning tree with random orientations plus random extra edges."""
    edges: Set[Edge] = set()
    order = rng.shuffled(range(n))
    for i in range(1, n):
        a, b = order[i], order[rng.below(i)]
        edges.add((a, b) if rng.chance(0.5) else (b, a))
    target = int(round(n * rng.uniform(1.2, 1.8)))
    _fill_random(n, edges, target, rng)
    return _finish(n, edges, rng)


def ring_chords(n: int, rng: Rng) -> Graph:
    """Mostly-directed ring with a few chords (held out for structure OOD)."""
    edges: Set[Edge] = set()
    for i in range(n):
        a, b = i, (i + 1) % n
        edges.add((b, a) if rng.chance(0.2) else (a, b))
    chords = rng.randint(1, max(2, n // 3))
    _fill_random(n, edges, len(edges) + chords, rng)
    return _finish(n, edges, rng)


def two_cluster(n: int, rng: Rng) -> Graph:
    """Two sparse clusters joined by one or two bridges (held out for structure OOD)."""
    size_a = rng.randint(max(2, n // 2 - 1), max(2, min(n - 2, n // 2 + 1)))
    groups = [list(range(size_a)), list(range(size_a, n))]
    edges: Set[Edge] = set()
    for nodes in groups:
        order = rng.shuffled(nodes)
        for i in range(1, len(order)):
            a, b = order[i], order[rng.below(i)]
            edges.add((a, b) if rng.chance(0.5) else (b, a))
        target = len(edges) + max(0, int(round(len(nodes) * rng.uniform(0.2, 0.7))))
        tries = 200
        while len(edges) < target and tries > 0:
            tries -= 1
            _try_add(edges, rng.choice(nodes), rng.choice(nodes))
    _try_add(edges, rng.choice(groups[0]), rng.choice(groups[1]))
    if rng.chance(0.5):
        if rng.chance(0.5):
            _try_add(edges, rng.choice(groups[0]), rng.choice(groups[1]))
        else:
            _try_add(edges, rng.choice(groups[1]), rng.choice(groups[0]))
    return _finish(n, edges, rng)


FAMILIES: Dict[str, Callable[[int, Rng], Graph]] = {
    "layered_dag": layered_dag,
    "tree_plus": tree_plus,
    "sparse_random": sparse_random,
    "ring_chords": ring_chords,
    "two_cluster": two_cluster,
}

# Smallest node count each family is defined for.
MIN_NODES: Dict[str, int] = {
    "layered_dag": 4,
    "tree_plus": 3,
    "sparse_random": 3,
    "ring_chords": 4,
    "two_cluster": 5,
}


def generate(family: str, n: int, rng: Rng) -> Graph:
    if family not in FAMILIES:
        raise KeyError("unknown graph family: %s" % family)
    if n < MIN_NODES[family]:
        raise ValueError("family %s needs at least %d nodes" % (family, MIN_NODES[family]))
    return FAMILIES[family](n, rng)


def family_invariants_ok(g: Graph) -> Tuple[bool, str]:
    """Checks shared by all families; used by tests and the dataset validator."""
    if g.has_antiparallel():
        return False, "antiparallel edge pair"
    if not g.is_weakly_connected():
        return False, "not weakly connected"
    return True, ""
