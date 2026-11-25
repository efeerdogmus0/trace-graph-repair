"""final_primary.py: the pre-registered six-cell primary statistic, pooled over seeds, all-or-nothing.
Synthetic flags only (constant answers or hand-made sample lists); no model output on the locked test is read."""
import importlib.util, json
from pathlib import Path
import pytest
from trace_vr import arms, bootstrap, dataset

ROOT = Path(__file__).resolve().parents[1]
P = "primary_six_cell"
SEEDS = (3407, 2027, 9173)


def load():
    spec = importlib.util.spec_from_file_location("final_primary", ROOT / "scripts/final_primary.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(split, n_groups):
    rows = []
    for g in range(n_groups):
        for task in ("reachability", "critical_edge"):
            for k in range(2):
                rows.append({"sample_id": "%s-%d-%s-%d" % (split, g, task, k), "graph_group_id": "%s-g%d" % (split, g),
                             "task": task, "metadata": {"image_variant": "root"}})
    return rows


def _synthetic(acc):
    """acc: {arm: {split: {task: fraction of each group's two samples right (0, 0.5, 1)}}}, same for every seed."""
    m = load()
    out = {}
    for s in m.PRIMARY_SPLITS:
        rows = _rows(s, 4)
        by_arm = {}
        for a, spec in acc.items():
            fl = {}
            for r in rows:
                right = int(round(spec[s][r["task"]] * 2))
                fl[r["sample_id"]] = int(r["sample_id"][-1]) < right
            by_arm[a] = {sd: dict(fl) for sd in SEEDS}
        out[s] = (rows, by_arm)
    return m, out


def test_primary_is_mean_of_six_cells_and_reads_by_seed_rule():
    spec_b = {s: {"reachability": 0.5, "critical_edge": 0.5} for s in ("test_id", "test_style_ood", "test_struct_ood")}
    spec_d = {s: {"reachability": 1.0, "critical_edge": 0.5} for s in ("test_id", "test_style_ood", "test_struct_ood")}
    spec_d["test_struct_ood"]["critical_edge"] = 0.0
    m, data = _synthetic({"B": spec_b, "D": spec_d, "E": spec_b})
    t = m.compute(data, "B", n_boot=50)
    rows = {a["label"]: a for a in t["arms"]}
    assert rows["B"]["pooled"][P]["point"] == pytest.approx(0.5)
    assert rows["D"]["pooled"][P]["point"] == pytest.approx((1 + 0.5 + 1 + 0.5 + 1 + 0) / 6)
    assert rows["D"]["pooled"]["critical_edge"]["point"] == pytest.approx(1 / 3)
    diffs = {(d["a"], d["b"]): d for d in t["differences"]}
    assert set(diffs) == {("D", "B"), ("E", "B"), ("E", "D")}
    dB = diffs[("D", "B")]
    assert dB["diff"][P]["point"] == pytest.approx(1 / 6)
    assert dB["seed_matched"] == {str(s): pytest.approx(1 / 6) for s in SEEDS}
    # constant flags per group: every replicate gives the same difference -> interval collapses above 0
    assert dB["diff"][P]["lo"] > 0 and dB["reading"] == "positive_clear"
    assert dB["task_drop"] == ["critical_edge"]           # critical edge mean 1/3 vs 1/2
    assert diffs[("E", "B")]["reading"] == "inconclusive"   # identical arms
    assert diffs[("E", "B")]["splits_same_sign"] is False
    md = m.markdown(t)
    assert "D - B" in md and "positive_clear" in md and "E - D" in md
    assert "\\toprule" in m.latex(t)


def test_bootstrap_is_reproducible_and_stratified():
    spec = {s: {"reachability": 0.5, "critical_edge": 1.0} for s in ("test_id", "test_style_ood", "test_struct_ood")}
    m, data = _synthetic({"B": spec, "D": spec})
    a = m.compute(data, "B", n_boot=30)
    b = m.compute(data, "B", n_boot=30)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["root_graphs"] == {"test_id": 4, "test_style_ood": 4, "test_struct_ood": 4}


def test_refuses_until_every_final_run_is_collected(tmp_path):
    m = load()
    jobs = [{"job": "jobF%s%d" % (a, s), "arm": a, "seed": s, "dev_dir": "runs/%s_%d" % (a, s)} for s in SEEDS for a in ("B", "D")]
    with pytest.raises(SystemExit) as e:
        m.build(jobs, m.final_table.SPLITS, "B", n_boot=10, root=tmp_path)
    assert "refused" in str(e.value) and "6 of 6" in str(e.value)


def test_primary_splits_are_the_first_three_locked_splits():
    m = load()
    assert [s for _, s, _ in m.splits_used(m.final_table.SPLITS)] == list(m.PRIMARY_SPLITS)
    assert [s for _, s, _ in m.final_table.SPLITS][:3] == list(m.PRIMARY_SPLITS)


needs_data = pytest.mark.skipif(not all((ROOT / ("data/pilot_test/%s.samples.jsonl" % s)).exists()
                                        for s in ("test_id", "test_style_ood", "test_struct_ood")),
                                reason="locked test data not built")


@needs_data
def test_build_on_real_sample_lists_with_constant_answers(tmp_path):
    m = load()
    small = [("pilot_test", s, 4) for s in m.PRIMARY_SPLITS]
    jobs = [{"job": "jobF%s%d" % (a, s), "arm": a, "seed": s, "dev_dir": "runs/%s_%d" % (a, s)} for s in SEEDS for a in ("B", "D")]
    man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests/pilot_test.json")
    for e in jobs:
        d = tmp_path / e["dev_dir"]; d.mkdir(parents=True)
        (d / "final_summary.json").write_text("{}")
        for _, split, roots in small:
            rows = arms.eval_samples(by_split[split], roots)
            answer = '{"answer": true}' if e["arm"] == "B" else '{"answer": false}'
            out = bootstrap.score_flags(rows, {s["sample_id"]: answer for s in rows})
            out.update({"split": split, "sample_set": "main", "eval_root_graphs": roots, "content_digest": man["content_digest"]})
            (d / ("flags_%s.json" % split)).write_text(json.dumps(out))
    t = m.build(jobs, small, "B", n_boot=20, root=tmp_path)
    rows = {a["label"]: a for a in t["arms"]}
    b, d = rows["B"]["pooled"][m.PRIMARY], rows["D"]["pooled"][m.PRIMARY]
    # constant true + constant false answer every binary question once: per cell the two accuracies sum to 1
    assert b["point"] + d["point"] == pytest.approx(1.0)
    assert len(t["files"]) == 18
