"""Risk 1 and 4 of the master plan: graph answers and topology comparison must agree with
independent references (networkx here; `reference.py` as a second, separately written one)."""
import networkx as nx

from trace_vr import families, iso, reference as ref, semantics as sem
from trace_vr.graphs import Graph
from trace_vr.rng import Rng


def random_graphs(count, tag, max_nodes=12):
    for i in range(count):
        rng = Rng(tag, i)
        fam = rng.choice(sorted(families.FAMILIES))
        n = rng.randint(families.MIN_NODES[fam], max_nodes)
        yield fam, families.generate(fam, n, rng.child("g")), rng


def to_nx(g):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from(g.edges)
    return G


def test_families_keep_their_invariants():
    for fam, g, _ in random_graphs(250, "invariants", 16):
        ok, why = families.family_invariants_ok(g)
        assert ok, (fam, why, g)
        assert nx.is_weakly_connected(to_nx(g))
        assert not any(g.has_edge(v, u) for u, v in g.edges)


def test_reachability_and_shortest_paths_match_two_references():
    for _, g, _ in random_graphs(80, "reach"):
        G = to_nx(g)
        nodes = [str(i) for i in range(g.n)]
        edges = [(str(u), str(v)) for u, v in g.edges]
        fw = ref.closure_fw(nodes, edges)
        for s in range(g.n):
            desc = set()
            for v in G.successors(s):
                desc |= {v} | nx.descendants(G, v)
            for t in range(g.n):
                assert g.reaches(s, t) == (t in desc) == fw[(str(s), str(t))]
                if s != t:
                    want = nx.shortest_path_length(G, s, t) if t in desc else None
                    assert g.shortest_path_len(s, t) == want
                    assert ref.shortest_len(nodes, edges, str(s), str(t)) == want


def test_isomorphism_matches_networkx():
    for _, g, rng in random_graphs(150, "iso"):
        perm = rng.shuffled(range(g.n))
        h = g.relabeled(perm)
        assert iso.fingerprint(g) == iso.fingerprint(h)
        mapping = iso.find_isomorphism(g, h)
        assert mapping is not None and all(h.has_edge(mapping[u], mapping[v]) for u, v in g.edges)
        changed = sem.apply_mod(g, rng.choice(sem.single_edge_mods(g))).relabeled(perm)
        assert iso.is_isomorphic(g, changed) == nx.is_isomorphic(to_nx(g), to_nx(changed))


def test_equal_fingerprint_is_not_taken_as_proof():
    one_ring = Graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    two_rings = Graph(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert iso.fingerprint(one_ring) == iso.fingerprint(two_rings)  # colour refinement cannot separate these
    assert not iso.is_isomorphic(one_ring, two_rings)
    reg = iso.TopologyRegistry()
    reg.add(one_ring, "g1")
    assert reg.find_other_group(two_rings, "g2") is None
    assert reg.find_other_group(one_ring.relabeled([3, 4, 5, 0, 1, 2]), "g2") == "g1"
    assert reg.find_other_group(one_ring, "g1") is None  # the same group may repeat a topology


def test_graph_rejects_out_of_scope_edges():
    import pytest

    with pytest.raises(ValueError):
        Graph(3, [(0, 0)])
    with pytest.raises(ValueError):
        Graph(3, [(0, 5)])
    g = Graph(3, [(0, 1)])
    assert not g.can_add(1, 0) and not g.can_add(0, 1) and g.can_add(1, 2)
