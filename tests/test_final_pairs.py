"""final_pairs.py: the locked-test twin-pair / answer-bias table is all-or-nothing and equals seed_pairs.py per split.
Flags here come from constant answers scored against the labels, so no model output on the locked test is involved."""
import importlib.util, json
from pathlib import Path
import pytest
from trace_vr import arms, bootstrap, dataset, tasks

ROOT = Path(__file__).resolve().parents[1]
SMALL = [("pilot_test", "test_id", 40), ("pilot_diag", "test_size_ood", 30)]
needs_data = pytest.mark.skipif(not (ROOT / "data/pilot_test/test_id.samples.jsonl").exists()
                                or not (ROOT / "data/pilot_diag/test_size_ood.samples.jsonl").exists(),
                                reason="locked test data not built")


def load():
    spec = importlib.util.spec_from_file_location("final_pairs", ROOT / "scripts/final_pairs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _jobs(arm_list=("B", "D")):
    return [{"job": "jobF%s%d" % (a, s), "arm": a, "seed": s, "dev_dir": "runs/%s_%d" % (a, s)}
            for s in (3407, 2027, 9173) for a in arm_list]


def _answer(arm, seed):
    """B: always true. D: true in seed 3407, false in the other two."""
    return '{"answer": true}' if arm == "B" or seed == 3407 else '{"answer": false}'


def _write(tmp_path, jobs, skip=None):
    for e in jobs:
        d = tmp_path / e["dev_dir"]; d.mkdir(parents=True)
        (d / "final_summary.json").write_text("{}")
        for manifest, split, roots in SMALL:
            if skip == (e["job"], split):
                continue
            man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
            rows = arms.select_samples(by_split[split], roots, "main")
            out = bootstrap.score_flags(rows, {s["sample_id"]: _answer(e["arm"], e["seed"]) for s in rows})
            out.update({"split": split, "sample_set": "main", "eval_root_graphs": roots, "content_digest": man["content_digest"]})
            (d / ("flags_%s.json" % split)).write_text(json.dumps(out))


@needs_data
def test_constant_answers_never_get_both_twins_right(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs)
    t = m.build(jobs, SMALL, root=tmp_path)
    assert t["splits"] == ["test_id", "test_size_ood"] and t["labels"] == ["B", "D"] and len(t["jobs"]) == 6
    for split in t["splits"]:
        tab = t["tables"][split]
        rows = {a["label"]: a for a in tab["arms"]}
        assert rows["B"]["seeds"] == rows["D"]["seeds"] == [2027, 3407, 9173]
        assert set(rows["B"]["summed"]["pairs"]) & set(tasks.BOOLEAN_TASKS), "no yes/no twin pairs on %s" % split
        for label in ("B", "D"):
            for task, tot in rows[label]["summed"]["pairs"].items():
                per = [rows[label]["per_seed"][s]["pairs"][task] for s in ("2027", "3407", "9173")]
                assert len({c["pairs"] for c in per}) == 1 and tot["pairs"] == 3 * per[0]["pairs"] > 0
                if task in tasks.BOOLEAN_TASKS:
                    # twins have opposite answers: a constant answer gets exactly one of the two right
                    assert tot["both_right"] == tot["both_wrong"] == 0 and tot["one_right"] == tot["pairs"]
                else:
                    # repair pairs: a yes/no answer is no repair action, so both are wrong
                    assert tot["both_right"] == tot["one_right"] == 0 and tot["both_wrong"] == tot["pairs"]
                assert tot["both_right_rate"] == 0.0
        for task, b in rows["B"]["summed"]["answer_bias"].items():
            d = rows["D"]["summed"]["answer_bias"][task]
            assert b["no_answer"] == d["no_answer"] == 0 and b["n"] == d["n"]
            assert b["said_true_rate"] == 1.0 and d["said_true_rate"] == pytest.approx(1 / 3)
            assert b["truth_true_rate"] == d["truth_true_rate"]
        # same object as seed_pairs.py applied to this split's files
        entries = [(e["arm"], e["seed"], tmp_path / e["dev_dir"] / ("flags_%s.json" % split)) for e in jobs]
        direct = m.seed_pairs.build(str(ROOT / "manifests" / (tab["manifest"] + ".json")), entries)
        assert tab["arms"] == direct["arms"] and tab["content_digest"] == direct["content_digest"]
        assert all(not f["file"].startswith("/") for f in tab["files"])
    md = m.markdown(t)
    assert "| test_id | " in md and "| test_size_ood | " in md and "Descriptive" in md and " (0.000) |" in md
    assert "| 1.000 | 0.333 |" in md
    tex = m.latex(t).splitlines()
    assert tex[1] == "\\begin{tabular}{llrcc}" and tex[3] == "Split & Task & Pairs & B & D \\\\" and tex[-1] == "\\end{tabular}"
    body = [l for l in tex if l.startswith("test\\_")]
    assert len(body) == sum(len(m._tasks(t["tables"][s], "pairs")) for s in t["splits"])
    assert all(l.count("&") == 4 and "_" not in l.replace("\\_", "") for l in body)


@needs_data
def test_refuses_and_names_the_uncollected_run(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs, skip=("jobFD9173", "test_size_ood"))
    with pytest.raises(SystemExit, match="jobFD9173"):
        m.build(jobs, SMALL, root=tmp_path)


@needs_data
def test_refuses_invariance_flags_in_place_of_main(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs)
    p = tmp_path / jobs[0]["dev_dir"] / "flags_test_id.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), sample_set="invariance")))
    with pytest.raises(SystemExit, match="main test_id"):
        m.build(jobs, SMALL, root=tmp_path)


@needs_data
def test_refuses_an_arm_without_all_seeds(tmp_path):
    m = load()
    jobs = _jobs()[:-1]
    _write(tmp_path, jobs)
    with pytest.raises(SystemExit, match="three pre-registered seeds"):
        m.build(jobs, SMALL, root=tmp_path)


def test_real_queue_is_refused_until_every_final_run_is_collected(tmp_path):
    m = load()
    jobs = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())["jobs"]
    if not m.final_table.missing(jobs, m.SPLITS):
        pytest.skip("all final runs collected")
    with pytest.raises(SystemExit, match="refused"):
        m.main(["--out-md", str(tmp_path / "t.md"), "--out-tex", str(tmp_path / "t.tex"), "--out-json", str(tmp_path / "t.json")])
    assert not list(tmp_path.iterdir())
