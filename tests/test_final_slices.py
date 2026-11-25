"""final_slices.py: the locked-test slice breakdown is all-or-nothing and equals a direct count per slice.
Flags here are constant answers scored against the labels, or bits set from drawing metadata, so no model output
on the locked test is involved."""
import importlib.util, json
from pathlib import Path
import pytest
from trace_vr import analysis, arms, bootstrap, dataset, metrics

ROOT = Path(__file__).resolve().parents[1]
SMALL = [("pilot_test", "test_id", 40), ("pilot_diag", "test_size_ood", 30)]
needs_data = pytest.mark.skipif(not (ROOT / "data/pilot_test/test_id.samples.jsonl").exists()
                                or not (ROOT / "data/pilot_diag/test_size_ood.samples.jsonl").exists(),
                                reason="locked test data not built")


def load():
    spec = importlib.util.spec_from_file_location("final_slices", ROOT / "scripts/final_slices.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _jobs(arm_list=("B", "D")):
    return [{"job": "jobF%s%d" % (a, s), "arm": a, "seed": s, "dev_dir": "runs/%s_%d" % (a, s)}
            for s in (3407, 2027, 9173) for a in arm_list]


def _rows(manifest, split, roots):
    man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
    return man, arms.select_samples(by_split[split], roots, "main")


def _write(tmp_path, jobs, skip=None):
    """B: always true. D: always false, except seed 3407, which is right exactly on root (unedited) images."""
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
                out["correct"] = "".join("1" if s["metadata"]["image_variant"] == "root" else "0" for s in rows)
            out.update({"split": split, "sample_set": "main", "eval_root_graphs": roots, "content_digest": man["content_digest"]})
            (d / ("flags_%s.json" % split)).write_text(json.dumps(out))


@needs_data
def test_slices_equal_a_direct_count(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs)
    t = m.build(jobs, SMALL, root=tmp_path)
    assert t["splits"] == ["test_id", "test_size_ood"] and t["labels"] == ["B", "D"] and len(t["jobs"]) == 6
    assert t["facets"] == list(analysis.FACETS) == ["truth", "image_variant", "graph_family", "size", "layout", "style"]
    checked = 0
    for manifest, split, roots in SMALL:
        _, rows = _rows(manifest, split, roots)
        main_rows = [s for s in rows if s["metadata"]["image_variant"] in metrics.MAIN_VARIANTS]
        tab = t["tables"][split]["tasks"]
        assert set(tab) == {s["task"] for s in main_rows}
        flags = {(e["arm"], e["seed"]): bootstrap.decode_flags(rows, json.loads(
            (tmp_path / e["dev_dir"] / ("flags_%s.json" % split)).read_text())) for e in jobs}
        for task, facets in tab.items():
            for facet, fn in analysis.FACETS.items():
                values = {fn(s) for s in main_rows if s["task"] == task}
                assert set(facets[facet]) == values                       # every value listed, none added
                assert sum(c["n"] for c in facets[facet].values()) == sum(1 for s in main_rows if s["task"] == task)
                for value in values:
                    sel = [s for s in main_rows if s["task"] == task and fn(s) == value]
                    cell = facets[facet][value]
                    assert cell["n"] == len(sel)
                    for arm in ("B", "D"):
                        right = [sum(flags[(arm, seed)][s["sample_id"]] for s in sel) for seed in (3407, 2027, 9173)]
                        got = cell["arms"][arm]
                        assert got["right"] == sum(right) and got["n"] == 3 * len(sel)
                        assert got["accuracy"] == pytest.approx(sum(right) / (3 * len(sel)))
                        assert [got["per_seed"][s] for s in ("3407", "2027", "9173")] == pytest.approx([r / len(sel) for r in right])
                        checked += 1
            if "root" in facets["image_variant"]:
                assert facets["image_variant"]["root"]["arms"]["D"]["per_seed"]["3407"] == 1.0   # right exactly on roots
            b = facets["truth"]
            if "True" in b:
                assert b["True"]["arms"]["B"]["accuracy"] == 1.0          # constant "true" is right whenever true is
            if "False" in b:
                assert b["False"]["arms"]["B"]["accuracy"] == 0.0
    assert checked > 50
    md = m.markdown(t)
    assert "| Split | Task | Facet | Value | n | B | D |" in md and "Descriptive" in md
    assert md.count("\n| test_id | ") + md.count("\n| test_size_ood | ") == checked // 2
    tex = m.latex(t).splitlines()
    assert tex[1] == "\\begin{longtable}{llllrcc}" and tex[-1] == "\\end{longtable}"
    body = [l for l in tex if l.startswith("test\\_")]
    assert len(body) == checked // 2
    assert all(l.count("&") == 6 and "_" not in l.replace("\\_", "") and "<" not in l and ">" not in l for l in body)


@needs_data
def test_refuses_and_names_the_uncollected_run(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs, skip=("jobFD9173", "test_size_ood"))
    with pytest.raises(SystemExit, match="jobFD9173"):
        m.build(jobs, SMALL, root=tmp_path)


@needs_data
def test_refuses_flags_from_another_build_or_sample_set(tmp_path):
    m = load()
    jobs = _jobs()
    _write(tmp_path, jobs)
    p = tmp_path / jobs[0]["dev_dir"] / "flags_test_id.json"
    good = json.loads(p.read_text())
    p.write_text(json.dumps(dict(good, sample_set="invariance")))
    with pytest.raises(SystemExit, match="main test_id"):
        m.build(jobs, SMALL, root=tmp_path)
    p.write_text(json.dumps(dict(good, content_digest="0" * 64)))
    with pytest.raises(SystemExit, match="different dataset build"):
        m.build(jobs, SMALL, root=tmp_path)


@needs_data
def test_refuses_an_arm_without_all_seeds(tmp_path):
    m = load()
    jobs = _jobs()[:-1]
    _write(tmp_path, jobs)
    with pytest.raises(SystemExit, match="three pre-registered seeds"):
        m.build(jobs, SMALL, root=tmp_path)


def test_facets_are_the_ones_named_in_the_paper():
    tex = " ".join((ROOT / "paper/sections/analysis.tex").read_text().split())
    assert ("Accuracy is broken down by true answer, image variant (root or edited), graph family, size bucket "
            "($\\le 8$, 9--12, $\\ge 13$ nodes), layout and style.") in tex
    assert load().FACETS == ("truth", "image_variant", "graph_family", "size", "layout", "style")


def test_real_queue_is_refused_until_every_final_run_is_collected(tmp_path):
    m = load()
    jobs = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())["jobs"]
    if not m.final_table.missing(jobs, m.SPLITS):
        pytest.skip("all final runs collected")
    with pytest.raises(SystemExit, match="refused"):
        m.main(["--out-md", str(tmp_path / "t.md"), "--out-tex", str(tmp_path / "t.tex"), "--out-json", str(tmp_path / "t.json")])
    assert not list(tmp_path.iterdir())
