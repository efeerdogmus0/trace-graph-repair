import pytest

from trace_vr import analysis, arms, bootstrap, tasks
from test_dataset import built  # noqa: F401  (module-scoped fixture)


def test_quantile_is_nearest_rank():
    v = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert analysis.quantile(v, 0.0) == 1.0 and analysis.quantile(v, 1.0) == 5.0
    assert analysis.quantile(v, 0.5) == 3.0 and analysis.quantile(v, 0.2) == 1.0
    with pytest.raises(ValueError):
        analysis.quantile([], 0.5)


def test_root_density_reads_the_root_drawing(built):
    rows = built["samples"]["test_id"]
    dens = analysis.root_density(rows)
    assert set(dens) == {s["graph_group_id"] for s in rows}
    for s in rows:
        if s["metadata"]["image_variant"] == "root":
            md = s["metadata"]
            assert dens[s["graph_group_id"]] == md["edge_count"] / md["node_count"]


def test_density_matched_counts_and_scores(built):
    rows = arms.eval_samples(built["samples"]["test_id"], 4)
    dens = analysis.root_density(rows)
    groups = sorted(dens, key=dens.get)
    # window excluding the sparsest and the densest root
    window = {"lo": dens[groups[1]], "hi": dens[groups[-2]]}
    keep = {g for g in groups if window["lo"] <= dens[g] <= window["hi"]}
    flags = {s["sample_id"]: (s["task"] == tasks.REACHABILITY) for s in rows}
    res = analysis.density_matched(rows, window, flags)
    assert res["roots"] == len(dens) and res["matched_roots"] == len(keep)
    assert res["below"] + res["above"] + res["matched_roots"] == res["roots"]
    acc = res["accuracy"]
    assert acc["all"][tasks.REACHABILITY]["accuracy"] == 1.0
    assert acc["matched"][tasks.CRITICAL_EDGE]["accuracy"] == 0.0
    assert acc["matched"]["non_edit_macro"] == 0.5
    n_matched = sum(1 for s in rows if s["graph_group_id"] in keep)
    assert sum(c["n"] for t, c in acc["matched"].items() if t != "non_edit_macro") == n_matched
    # without flags only counts are returned
    assert "accuracy" not in analysis.density_matched(rows, window)


def test_full_window_matches_everything(built):
    rows = built["samples"]["test_id"]
    w = analysis.density_window(analysis.root_density(rows), 0.0, 1.0)
    res = analysis.density_matched(rows, w)
    assert res["matched_roots"] == res["roots"] and res["below"] == res["above"] == 0
