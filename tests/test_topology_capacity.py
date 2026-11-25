import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("topology_capacity", os.path.join(ROOT, "scripts", "topology_capacity.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scan_cell_deterministic_and_bounded():
    mod = _load()
    a = mod.scan_cell("layered_dag", 5, 300, 50)
    b = mod.scan_cell("layered_dag", 5, 300, 50)
    assert a == b
    assert a["singletons"] + 2 * a["doubletons"] <= a["draws"]
    assert a["singletons"] <= a["distinct"] <= a["draws"]
    assert a["chao1"] >= a["distinct"]
    assert 0.0 <= a["dup_rate_at_target"] <= 1.0


def test_small_cell_saturates():
    # Five-node layered DAGs: about 100 classes in the committed 4000-draw scan; 1500 draws
    # already find far fewer distinct topologies than draws, so a 400-root cell is impossible.
    mod = _load()
    r = mod.scan_cell("layered_dag", 5, 1500, 400)
    assert r["distinct"] < 150
    assert r["dup_rate_at_target"] is None


def test_committed_table_present():
    with open(os.path.join(ROOT, "docs", "topology_capacity.md")) as fh:
        text = fh.read()
    for fam in ("layered_dag", "tree_plus", "sparse_random", "ring_chords", "two_cluster"):
        for n in range(5, 19):
            assert "| %s | %d |" % (fam, n) in text
