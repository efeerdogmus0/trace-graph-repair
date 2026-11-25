"""final_table.py: the locked-test table is all-or-nothing and pools seeds per split like the dev table.
Flags here come from constant answers scored against the labels, so no model output on the locked test is involved."""
import importlib.util, json
from pathlib import Path
import pytest
from trace_vr import arms, bootstrap, dataset

ROOT = Path(__file__).resolve().parents[1]
SMALL = [("pilot_test", "test_id", 6), ("pilot_diag", "test_size_ood", 5)]
needs_data = pytest.mark.skipif(not (ROOT / "data/pilot_test/test_id.samples.jsonl").exists()
                                or not (ROOT / "data/pilot_diag/test_size_ood.samples.jsonl").exists(),
                                reason="locked test data not built")
MACRO = "non_edit_macro_accuracy"


def load():
    spec = importlib.util.spec_from_file_location("final_table", ROOT / "scripts/final_table.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _jobs(arm_list=("B", "D")):
    return [{"job": "jobF%s%d" % (a, s), "arm": a, "seed": s, "dev_dir": "runs/%s_%d" % (a, s)}
            for s in (3407, 2027, 9173) for a in arm_list]


def _write(tmp_path, jobs, skip=None):
    for e in jobs:
        d = tmp_path / e["dev_dir"]; d.mkdir(parents=True)
        (d / "final_summary.json").write_text("{}")
        for manifest, split, roots in SMALL:
            if skip == (e["job"], split):
                continue
            man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
            rows = arms.eval_samples(by_split[split], roots)
            answer = '{"answer": true}' if e["arm"] == "B" else '{"answer": false}'
            out = bootstrap.score_flags(rows, {s["sample_id"]: answer for s in rows})
            out.update({"split": split, "sample_set": "main", "eval_root_graphs": roots, "content_digest": man["content_digest"]})
            (d / ("flags_%s.json" % split)).write_text(json.dumps(out))


@needs_data
def test_table_pools_three_seeds_per_split(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs)
    t = m.build(jobs, SMALL, "B", n_boot=50, root=tmp_path)
    assert t["splits"] == ["test_id", "test_size_ood"] and len(t["jobs"]) == 6
    for split, roots in (("test_id", 6), ("test_size_ood", 5)):
        tab = t["tables"][split]
        rows = {a["label"]: a for a in tab["arms"]}
        assert tab["root_graphs"] == roots and tab["split"] == split and rows["B"]["seeds"] == [2027, 3407, 9173]
        per = list(rows["B"]["per_seed"].values())
        assert per[0] == per[1] == per[2]                       # identical answers -> identical seeds
        assert rows["B"]["pooled"][MACRO]["point"] == pytest.approx(per[0][MACRO])
        d = rows["D"]["vs_ref"][MACRO]["point"]
        assert d == pytest.approx(rows["D"]["pooled"][MACRO]["point"] - rows["B"]["pooled"][MACRO]["point"])
    md = m.markdown(t)
    assert "| test_id | 6 |" in md and "D - B" in md and "Locked test" in md
    tex = m.latex(t).splitlines()
    assert tex[1] == "\\begin{tabular}{lrccc}" and "B & D & D $-$ B" in tex[3] and tex[-1] == "\\end{tabular}"
    row = next(l for l in tex if l.startswith("test\\_size\\_ood & 5 & "))
    b, d = (rows_ := {a["label"]: a for a in t["tables"]["test_size_ood"]["arms"]})["B"], rows_["D"]
    assert row.count("&") == 4 and row.endswith("\\\\")
    assert "%.3f {" % b["pooled"][MACRO]["point"] in row and "$%+.3f$ {" % d["vs_ref"][MACRO]["point"] in row


@needs_data
def test_refuses_and_names_the_incomplete_run(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs, skip=("jobFD9173", "test_size_ood"))
    assert m.missing(jobs, SMALL, tmp_path) == [("jobFD9173", ["flags_test_size_ood.json"])]
    with pytest.raises(SystemExit, match="jobFD9173"):
        m.build(jobs, SMALL, "B", n_boot=10, root=tmp_path)


@needs_data
def test_refuses_an_arm_without_all_seeds(tmp_path):
    m = load()
    jobs = _jobs()[:-1]
    _write(tmp_path, jobs)
    with pytest.raises(SystemExit, match="three pre-registered seeds"):
        m.build(jobs, SMALL, "B", n_boot=10, root=tmp_path)


def test_real_queue_is_refused_until_every_final_run_is_collected(tmp_path):
    m = load()
    jobs = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())["jobs"]
    gaps = m.missing(jobs, m.SPLITS)
    if not gaps:
        pytest.skip("all final runs collected")
    out = tmp_path / "t.md"
    with pytest.raises(SystemExit, match="refused"):
        m.main(["--out-md", str(out), "--out-tex", str(tmp_path / "t.tex")])
    assert not out.exists() and not (tmp_path / "t.tex").exists()


def test_reference_comes_from_the_dev_choice(tmp_path):
    m = load()
    assert m.reference(ROOT / "reports/pilot/reference_baseline.json") == "B"
    bad = tmp_path / "r.json"
    bad.write_text(json.dumps({"split": "test_id", "chosen": "C"}))
    with pytest.raises(SystemExit, match="dev"):
        m.reference(bad)
