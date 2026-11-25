import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("layout_capacity", os.path.join(ROOT, "scripts", "layout_capacity.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scan_cell_deterministic_and_consistent():
    mod = _load()
    a = mod.scan_cell("tree_plus", 13, "a", 3)
    b = mod.scan_cell("tree_plus", 13, "a", 3)
    a.pop("s_per_graph"); b.pop("s_per_graph")
    assert a == b
    assert 0 <= a["pixel_problems"] <= a["accepted"] <= a["draws"] == 3
    assert sum(a["layouts"].values()) == a["accepted"]


def test_larger_budget_never_loses_a_layout_and_edges_are_split():
    mod = _load()
    small = mod.scan_cell("sparse_random", 15, "c", 3, render_check=False, budget=90)
    big = mod.scan_cell("sparse_random", 15, "c", 3, render_check=False, budget=270)
    # proposal seeds are indexed, so the bigger budget tries a superset of proposals
    assert big["accepted"] >= small["accepted"]
    assert small["budget"] == 90 and big["budget"] == 270
    for r in (small, big):
        assert (r["mean_edges_accepted"] is None) == (r["accepted"] == 0)
        assert (r["mean_edges_rejected"] is None) == (r["accepted"] == r["draws"])
