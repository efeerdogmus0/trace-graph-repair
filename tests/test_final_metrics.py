"""final_metrics.py: the locked-test per-task detail table is all-or-nothing, averages the collected metrics files over
seeds and refuses a metrics file that disagrees with its flags file. Metrics and flags here come from constant answers
scored against the labels, so no model output on the locked test is involved."""
import importlib.util, json, re
from pathlib import Path
import pytest
from trace_vr import arms, bootstrap, dataset, metrics, tasks

ROOT = Path(__file__).resolve().parents[1]
SMALL = [("pilot_test", "test_id", 40), ("pilot_diag", "test_size_ood", 30)]
needs_data = pytest.mark.skipif(not (ROOT / "data/pilot_test/test_id.samples.jsonl").exists()
                                or not (ROOT / "data/pilot_diag/test_size_ood.samples.jsonl").exists(),
                                reason="locked test data not built")
EXT = {"exact": 0.5, "format_ok": 1.0, "edge_precision": 0.9, "edge_recall": 0.8, "direction_accuracy": 0.7, "node_recall": 1.0}


def load():
    spec = importlib.util.spec_from_file_location("final_metrics", ROOT / "scripts/final_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _jobs(arm_list=("B", "E")):
    return [{"job": "jobF%s%d" % (a, s), "arm": a, "seed": s, "dev_dir": "runs/%s_%d" % (a, s)}
            for s in (3407, 2027, 9173) for a in arm_list]


def _answer(arm, seed):
    """B: always true. E: true in seed 3407, prose (a format failure with a lenient reading) in the other two."""
    return '{"answer": true}' if arm == "B" or seed == 3407 else 'I think {"answer": false}'


def _write(tmp_path, jobs, skip=None):
    for e in jobs:
        d = tmp_path / e["dev_dir"]; d.mkdir(parents=True)
        summary = {"metrics": {}}
        for manifest, split, roots in SMALL:
            man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
            rows = arms.select_samples(by_split[split], roots, "main")
            resp = {s["sample_id"]: _answer(e["arm"], e["seed"]) for s in rows}
            head = {"split": split, "sample_set": "main", "eval_root_graphs": roots, "content_digest": man["content_digest"]}
            if e["arm"] == "E":   # eval_parse_solve.py layout: no lenient_accuracy, a second metrics block, extraction quality
                m = dict(head, truncated_at_limit=1, max_new_tokens=512, images=7, metrics=metrics.evaluate(rows, resp),
                         metrics_lenient_parse=metrics.evaluate(rows, {k: tasks.lenient_candidate(v) for k, v in resp.items()}),
                         extraction_strict=dict(EXT, exact=EXT["exact"] + (0.3 if e["seed"] == 3407 else 0.0)))
                del m["sample_set"]
            else:
                m = dict(head, truncated_at_limit=0, max_new_tokens=96, metrics=metrics.evaluate(rows, resp, lenient=True))
            summary["metrics"][split] = m
            if skip == (e["job"], split):
                continue
            out = bootstrap.score_flags(rows, resp)
            out.update(head)
            (d / ("flags_%s.json" % split)).write_text(json.dumps(out))
        (d / "final_summary.json").write_text(json.dumps(summary))


@needs_data
def test_cells_are_seed_means_of_the_metrics_files(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs)
    t = m.build(jobs, SMALL, root=tmp_path)
    assert t["splits"] == ["test_id", "test_size_ood"] and t["labels"] == ["B", "E"] and len(t["jobs"]) == 6
    for manifest, split, roots in SMALL:
        man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
        rows = arms.select_samples(by_split[split], roots, "main")
        truth = metrics.evaluate(rows, {s["sample_id"]: '{"answer": true}' for s in rows}, lenient=True)["per_task"]
        falsy = metrics.evaluate(rows, {s["sample_id"]: '{"answer": false}' for s in rows})["per_task"]
        got = {a["label"]: a for a in t["tables"][split]["arms"]}
        assert got["B"]["seeds"] == got["E"]["seeds"] == [2027, 3407, 9173]
        assert set(got["B"]["mean"]) == set(truth) and tasks.REPAIR in truth and tasks.REACHABILITY in truth
        for task, cell in truth.items():
            b, e = got["B"]["mean"][task], got["E"]["mean"][task]
            assert b["n"] == e["n"] == cell["n"]
            # B answers the same in every seed: the mean is that seed's value, strict = lenient, no format failure
            assert b["accuracy"] == pytest.approx(cell["accuracy"]) and b["lenient_accuracy"] == pytest.approx(cell["accuracy"])
            assert b["format_failure_rate"] == cell["format_failure_rate"] == (0.0 if task in tasks.BOOLEAN_TASKS else 1.0)
            # E: one strict seed, two prose seeds (strict 0, format failure 1, lenient = the constant-false score)
            assert e["accuracy"] == pytest.approx(cell["accuracy"] / 3)
            assert e["format_failure_rate"] == pytest.approx((cell["format_failure_rate"] + 2) / 3)
            assert e["lenient_accuracy"] == pytest.approx((cell["accuracy"] + 2 * falsy[task]["accuracy"]) / 3)
        rep = got["B"]["mean"][tasks.REPAIR]
        for k in m.REPAIR_FIELDS:
            assert rep[k] == pytest.approx(truth[tasks.REPAIR][k])
        assert not any(k in got["B"]["mean"][tasks.REACHABILITY] for k in m.REPAIR_FIELDS)
        assert got["B"]["truncated_at_limit"] == {"2027": 0, "3407": 0, "9173": 0}
        assert got["E"]["truncated_at_limit"] == {"2027": 1, "3407": 1, "9173": 1}
        assert "extraction_mean" not in got["B"]
        assert got["E"]["extraction_mean"]["exact"] == pytest.approx(0.6)
        assert got["E"]["extraction_mean"]["edge_recall"] == pytest.approx(0.8)
        assert got["E"]["per_seed"]["3407"]["extraction"]["exact"] == pytest.approx(0.8)
    md = m.markdown(t)
    assert "Descriptive" in md and "| test_id | reachability | " in md and "Extraction quality of arm E" in md
    assert "Extraction quality of arm B" not in md and "| test_size_ood | 0.600 | 1.000 | 0.900 | 0.800 | 0.700 | 1.000 |" in md
    assert "| test_id | 0, 0, 0 | 1, 1, 1 |" in md
    tex = m.latex(t).splitlines()
    assert tex[1] == "\\begin{tabular}{llrcc}" and tex[3] == "Split & Task & Questions & B & E \\\\" and tex[-1] == "\\end{tabular}"
    body = [l for l in tex if l.startswith("test\\_")]
    assert len(body) == 2 * len(got["B"]["mean"])
    assert all(l.count("&") == 4 and "_" not in l.replace("\\_", "") for l in body)


@needs_data
def test_refuses_a_metrics_file_that_disagrees_with_its_flags(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs)
    p = tmp_path / jobs[0]["dev_dir"] / "final_summary.json"
    s = json.loads(p.read_text())
    cell = s["metrics"]["test_id"]["metrics"]["per_task"][tasks.REACHABILITY]
    cell["accuracy"] = cell["accuracy"] + 0.01
    p.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="reachability accuracy in the metrics file differs"):
        m.build(jobs, SMALL, root=tmp_path)


@needs_data
def test_refuses_and_names_the_uncollected_run(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs, skip=("jobFE9173", "test_size_ood"))
    with pytest.raises(SystemExit, match="jobFE9173"):
        m.build(jobs, SMALL, root=tmp_path)


@needs_data
def test_refuses_invariance_metrics_in_place_of_main(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs)
    p = tmp_path / jobs[0]["dev_dir"] / "final_summary.json"
    s = json.loads(p.read_text())
    s["metrics"]["test_id"]["sample_set"] = "invariance"
    p.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="main test_id"):
        m.build(jobs, SMALL, root=tmp_path)


@needs_data
def test_refuses_an_arm_without_all_seeds(tmp_path):
    m = load()
    jobs = _jobs()[:-1]
    _write(tmp_path, jobs)
    with pytest.raises(SystemExit, match="three pre-registered seeds"):
        m.build(jobs, SMALL, root=tmp_path)


def test_fields_are_the_ones_the_paper_names():
    m = load()
    text = " ".join((ROOT / "paper/sections/analysis.tex").read_text().split())
    para = re.search(r"\\paragraph\{Cells and task means\.\}(.*?)\\paragraph", text).group(1)
    for phrase in ("status accuracy", "valid-edit rate", "minimal-repair accuracy", "Format failures (strict)", "lenient score"):
        assert phrase in para
    assert m.REPAIR_FIELDS == ("status_accuracy", "valid_edit_rate", "minimal_repair_accuracy")
    assert m.TASK_FIELDS == ("accuracy", "format_failure_rate", "lenient_accuracy")
    ext = re.search(r"\(ii\) Arm E is scored with (.*?) of its extraction", text).group(1)
    assert ext == "node and edge precision and recall and direction accuracy"
    assert {"edge_precision", "edge_recall", "direction_accuracy", "node_recall"} <= set(m.EXTRACTION_FIELDS)


def test_real_queue_is_refused_until_every_final_run_is_collected(tmp_path):
    m = load()
    jobs = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())["jobs"]
    if not m.final_table.missing(jobs, m.SPLITS):
        pytest.skip("all final runs collected")
    with pytest.raises(SystemExit, match="refused"):
        m.main(["--out-md", str(tmp_path / "t.md"), "--out-tex", str(tmp_path / "t.tex"), "--out-json", str(tmp_path / "t.json")])
    assert not list(tmp_path.iterdir())
