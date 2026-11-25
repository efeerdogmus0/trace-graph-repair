"""final_density.py: the locked-test density-matched slice is all-or-nothing and equals a direct per-task count.
Flags here are constant answers scored against the labels, or bits set from drawing metadata, so no model output
on the locked test is involved."""
import importlib.util, json
from pathlib import Path
import pytest
from trace_vr import analysis, arms, bootstrap, dataset, tasks

ROOT = Path(__file__).resolve().parents[1]
SMALL = [("pilot_test", "test_id", 40), ("pilot_diag", "test_size_ood", 30)]
NARROW = {"lo": 1.1, "hi": 1.3, "lo_q": 0.05, "hi_q": 0.95, "ref_roots": 512, "ref_mean": 1.2, "ref": "made up for the test"}
needs_data = pytest.mark.skipif(not (ROOT / "data/pilot_test/test_id.samples.jsonl").exists()
                                or not (ROOT / "data/pilot_diag/test_size_ood.samples.jsonl").exists()
                                or not (ROOT / "data/pilot/train.samples.jsonl").exists(),
                                reason="pilot / locked test data not built")


def load():
    spec = importlib.util.spec_from_file_location("final_density", ROOT / "scripts/final_density.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _jobs(arm_list=("B", "D")):
    return [{"job": "jobF%s%d" % (a, s), "arm": a, "seed": s, "dev_dir": "runs/%s_%d" % (a, s)}
            for s in (3407, 2027, 9173) for a in arm_list]


def _rows(manifest, split, roots):
    man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
    return man, arms.select_samples(by_split[split], roots, "main")


def _inside(rows, win):
    dens = analysis.root_density(rows)
    return {g for g, d in dens.items() if win["lo"] <= d <= win["hi"]}


def _write(tmp_path, jobs, win, skip=None):
    """B: always true. D: right exactly on the roots inside the window (seed 3407), always false otherwise."""
    for e in jobs:
        d = tmp_path / e["dev_dir"]; d.mkdir(parents=True)
        (d / "final_summary.json").write_text("{}")
        for manifest, split, roots in SMALL:
            if skip == (e["job"], split):
                continue
            man, rows = _rows(manifest, split, roots)
            answer = '{"answer": true}' if e["arm"] == "B" else '{"answer": false}'
            out = bootstrap.score_flags(rows, {s["sample_id"]: answer for s in rows})
            if e["arm"] == "D" and e["seed"] == 3407:
                keep = _inside(rows, win)
                out["correct"] = "".join("1" if s["graph_group_id"] in keep else "0" for s in rows)
            out.update({"split": split, "sample_set": "main", "eval_root_graphs": roots, "content_digest": man["content_digest"]})
            (d / ("flags_%s.json" % split)).write_text(json.dumps(out))


def _macro(rows, flags, keep=None):
    acc = []
    for task in (tasks.REACHABILITY, tasks.CRITICAL_EDGE):
        sel = [s for s in rows if s["task"] == task and (keep is None or s["graph_group_id"] in keep)]
        acc.append(sum(flags[s["sample_id"]] for s in sel) / len(sel))
    return sum(acc) / 2


@needs_data
def test_window_is_the_one_stated_in_the_paper():
    w = load().window()
    assert (w["lo"], round(w["hi"], 3), w["ref_roots"], w["ref"]) == (0.875, 1.667, 512, "pilot.json:train")
    recorded = json.loads((ROOT / "reports/density/pilot_dev.json").read_text())["window"]
    assert {k: w[k] for k in ("lo", "hi", "lo_q", "hi_q", "ref_roots")} == {k: recorded[k] for k in ("lo", "hi", "lo_q", "hi_q", "ref_roots")}
    tex = (ROOT / "paper/sections/analysis.tex").read_text()
    assert "$[0.875, 1.667]$" in tex and "density-matched slice" in tex


@needs_data
def test_slice_equals_a_direct_count(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs, NARROW)
    t = m.build(jobs, SMALL, root=tmp_path, win=NARROW)
    assert t["splits"] == ["test_id", "test_size_ood"] and t["labels"] == ["B", "D"] and len(t["jobs"]) == 6
    for manifest, split, roots in SMALL:
        tab = t["tables"][split]
        _, rows = _rows(manifest, split, roots)
        keep = _inside(rows, NARROW)
        assert tab["roots"] == roots and tab["matched_roots"] == len(keep)
        assert 0 < len(keep) < roots, "the made-up window must split %s" % split
        assert tab["below"] + tab["above"] + tab["matched_roots"] == roots
        by = {a["label"]: a for a in tab["arms"]}
        assert by["B"]["seeds"] == by["D"]["seeds"] == [2027, 3407, 9173]
        for e in jobs:
            flags = bootstrap.decode_flags(rows, json.loads((tmp_path / e["dev_dir"] / ("flags_%s.json" % split)).read_text()))
            cell = by[e["arm"]]["per_seed"][str(e["seed"])]
            assert cell["all"] == pytest.approx(_macro(rows, flags))
            assert cell["matched"] == pytest.approx(_macro(rows, flags, keep))
        d = by["D"]["per_seed"]
        assert d["3407"]["matched"] == 1.0 and d["3407"]["all"] < 1.0      # right exactly inside the window
        b = by["B"]["per_seed"]
        assert b["3407"] == b["2027"] == b["9173"]                         # same constant answer in every seed
        for label in ("B", "D"):
            for sub in ("all", "matched"):
                vals = [by[label]["per_seed"][s][sub] for s in ("3407", "2027", "9173")]
                assert by[label]["mean"][sub] == pytest.approx(sum(vals) / 3)
    md = m.markdown(t)
    assert "| test_id | 40 | " in md and "| test_size_ood | 30 | " in md and "Descriptive" in md and "[1.100, 1.300]" in md
    tex = m.latex(t).splitlines()
    assert tex[1] == "\\begin{tabular}{lrrcc}" and tex[3] == "Split & Roots & Matched & B & D \\\\" and tex[-1] == "\\end{tabular}"
    body = [l for l in tex if l.startswith("test\\_")]
    assert len(body) == 2 and all(l.count("&") == 4 and "_" not in l.replace("\\_", "") for l in body)


@needs_data
def test_empty_slice_is_shown_as_not_available(tmp_path):
    m = load()
    jobs = _jobs()
    empty = dict(NARROW, lo=50.0, hi=60.0)
    _write(tmp_path, jobs, empty)
    t = m.build(jobs, SMALL, root=tmp_path, win=empty)
    for split in t["splits"]:
        tab = t["tables"][split]
        assert tab["matched_roots"] == 0 and all(a["mean"]["matched"] is None and a["mean"]["all"] is not None for a in tab["arms"])
    assert " / n/a |" in m.markdown(t)


@needs_data
def test_refuses_and_names_the_uncollected_run(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs, NARROW, skip=("jobFD9173", "test_size_ood"))
    with pytest.raises(SystemExit, match="jobFD9173"):
        m.build(jobs, SMALL, root=tmp_path, win=NARROW)


@needs_data
def test_refuses_flags_from_another_build_or_sample_set(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs, NARROW)
    p = tmp_path / jobs[0]["dev_dir"] / "flags_test_id.json"
    good = json.loads(p.read_text())
    p.write_text(json.dumps(dict(good, sample_set="invariance")))
    with pytest.raises(SystemExit, match="main test_id"):
        m.build(jobs, SMALL, root=tmp_path, win=NARROW)
    p.write_text(json.dumps(dict(good, content_digest="0" * 64)))
    with pytest.raises(SystemExit, match="different dataset build"):
        m.build(jobs, SMALL, root=tmp_path, win=NARROW)


@needs_data
def test_refuses_an_arm_without_all_seeds(tmp_path):
    m = load()
    jobs = _jobs()[:-1]
    _write(tmp_path, jobs, NARROW)
    with pytest.raises(SystemExit, match="three pre-registered seeds"):
        m.build(jobs, SMALL, root=tmp_path, win=NARROW)


def test_real_queue_is_refused_until_every_final_run_is_collected(tmp_path):
    m = load()
    jobs = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())["jobs"]
    if not m.final_table.missing(jobs, m.SPLITS):
        pytest.skip("all final runs collected")
    with pytest.raises(SystemExit, match="refused"):
        m.main(["--out-md", str(tmp_path / "t.md"), "--out-tex", str(tmp_path / "t.tex"), "--out-json", str(tmp_path / "t.json")])
    assert not list(tmp_path.iterdir())
