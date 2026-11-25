"""final_invariance.py: the locked-test invariance-pair table is all-or-nothing and sums pairs over tasks and seeds.
Flags here come from constant answers scored against the labels, so no model output on the locked test is involved."""
import importlib.util, json
from pathlib import Path
import pytest
from trace_vr import arms, bootstrap, dataset

ROOT = Path(__file__).resolve().parents[1]
SMALL = [("pilot_test", "test_id", 40), ("pilot_diag", "test_size_ood", 30)]
needs_data = pytest.mark.skipif(not (ROOT / "data/pilot_test/test_id.samples.jsonl").exists()
                                or not (ROOT / "data/pilot_diag/test_size_ood.samples.jsonl").exists(),
                                reason="locked test data not built")


def load():
    spec = importlib.util.spec_from_file_location("final_invariance", ROOT / "scripts/final_invariance.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _jobs(arm_list=("B", "D")):
    return [{"job": "jobF%s%d" % (a, s), "arm": a, "seed": s, "dev_dir": "runs/%s_%d" % (a, s)}
            for s in (3407, 2027, 9173) for a in arm_list]


def _answer(arm, seed, sample_set):
    """B: always true. D: true on roots; on copies true for one seed only, so its copies disagree with its roots."""
    if arm == "B" or sample_set == "main" or seed == 3407:
        return '{"answer": true}'
    return '{"answer": false}'


def _write(tmp_path, jobs, skip=None):
    counts = {}
    for e in jobs:
        d = tmp_path / e["dev_dir"]; d.mkdir(parents=True)
        (d / "final_summary.json").write_text("{}")
        for manifest, split, roots in SMALL:
            man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
            for sample_set, suffix in (("main", ""), ("invariance", "_inv")):
                if skip == (e["job"], split + suffix):
                    continue
                rows = arms.select_samples(by_split[split], roots, sample_set)
                out = bootstrap.score_flags(rows, {s["sample_id"]: _answer(e["arm"], e["seed"], sample_set) for s in rows})
                out.update({"split": split, "sample_set": sample_set, "eval_root_graphs": roots, "content_digest": man["content_digest"]})
                (d / ("flags_%s%s.json" % (split, suffix))).write_text(json.dumps(out))
                if sample_set == "invariance":
                    counts[split] = len(rows)
    return counts


@needs_data
def test_pairs_are_summed_over_tasks_and_seeds(tmp_path):
    m = load()
    jobs = _jobs()
    copies = _write(tmp_path, jobs)
    t = m.build(jobs, SMALL, root=tmp_path)
    assert t["splits"] == ["test_id", "test_size_ood"] and t["labels"] == ["B", "D"] and len(t["jobs"]) == 6
    for split, roots in (("test_id", 40), ("test_size_ood", 30)):
        tab = t["tables"][split]
        assert tab["root_graphs"] == roots and tab["invariance_copies"] == copies[split] > 0
        rows = {a["label"]: a for a in tab["arms"]}
        assert sum(rows["B"]["kinds"][k]["pairs_per_seed"] for k in tab["kinds"]) == copies[split]
        for kind in tab["kinds"]:
            b, d = rows["B"]["kinds"][kind], rows["D"]["kinds"][kind]
            n = b["pairs_per_seed"]
            assert b["pooled"]["pairs"] == d["pooled"]["pairs"] == 3 * n
            assert sum(c["pairs"] for c in b["per_task"].values()) == 3 * n
            # a constant answerer gives the copy the same answer as its root: always the same correctness
            assert b["pooled"]["same_correctness_rate"] == 1.0
            assert b["pooled"]["both_right"] == b["pooled"]["root_right"] == b["pooled"]["copy_right"]
            assert b["both_right_seed_range"]["min"] == b["both_right_seed_range"]["max"] == b["pooled"]["both_right_rate"]
            # D answers its copies the opposite way in two of three seeds: only seed 3407 is consistent
            assert d["pooled"]["root_right"] == b["pooled"]["root_right"]
            assert d["pooled"]["same_correctness"] == n and d["per_seed"]["3407"] == b["per_seed"]["3407"]
            assert d["per_seed"]["2027"]["both_right"] == d["per_seed"]["9173"]["same_correctness"] == 0
            assert d["pooled"]["copy_right"] == b["per_seed"]["3407"]["copy_right"] + 2 * (n - b["per_seed"]["3407"]["copy_right"])
            assert d["pooled"]["both_right_rate"] == pytest.approx(sum(v["both_right_rate"] for v in d["per_seed"].values()) / 3)
    md = m.markdown(t)
    assert "| test_id | relabel | " in md and "| test_size_ood | style | " in md and "Descriptive" in md
    tex = m.latex(t).splitlines()
    assert tex[1] == "\\begin{tabular}{llrcc}" and tex[3] == "Split & Pair kind & Pairs & B & D \\\\" and tex[-1] == "\\end{tabular}"
    body = [l for l in tex if l.startswith("test\\_")]
    assert len(body) == sum(len(t["tables"][s]["kinds"]) for s in t["splits"]) and all(l.count("&") == 4 for l in body)


@needs_data
def test_refuses_and_names_the_run_without_invariance_flags(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs, skip=("jobFD9173", "test_size_ood_inv"))
    assert m.missing(jobs, SMALL, tmp_path) == [("jobFD9173", ["flags_test_size_ood_inv.json"])]
    with pytest.raises(SystemExit, match="jobFD9173"):
        m.build(jobs, SMALL, root=tmp_path)


@needs_data
def test_refuses_flags_of_the_wrong_sample_set(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs)
    d = tmp_path / jobs[0]["dev_dir"]
    (d / "flags_test_id_inv.json").write_text((d / "flags_test_id.json").read_text())
    with pytest.raises(SystemExit, match="invariance test_id"):
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
    if not m.missing(jobs, m.SPLITS):
        pytest.skip("all final runs collected")
    out = tmp_path / "t.md"
    with pytest.raises(SystemExit, match="refused"):
        m.main(["--out-md", str(out), "--out-tex", str(tmp_path / "t.tex"), "--out-json", str(tmp_path / "t.json")])
    assert not list(tmp_path.iterdir())
