"""scripts/dev_figures.py: figures drawn from recorded dev metrics.

The plotted values must be copies of the seed record and the run summaries (nothing re-estimated),
a record of another split or a summary that disagrees with the record must be refused, the stored
figures_dev.json must match its sources, and the cost sentence of the paper must state its numbers."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "reports/pilot/seeds_dev_interim.json"
OUT = ROOT / "reports/pilot/figures"


def _load():
    spec = importlib.util.spec_from_file_location("dev_figures", ROOT / "scripts" / "dev_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fig = _load()
REC = json.loads(SEEDS.read_text())
STORED = json.loads((OUT / "figures_dev.json").read_text())


def _copy_tree(tmp_path, rec):
    """Seed record plus the summaries it points to, in a scratch folder."""
    (tmp_path / "reports/pilot").mkdir(parents=True)
    path = tmp_path / "reports/pilot/seeds.json"
    path.write_text(json.dumps(rec))
    for f in rec["files"]:
        src = (ROOT / f["file"]).parent / "summary.json"
        dst = (tmp_path / f["file"]).parent / "summary.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())
    return path


def test_stored_record_matches_sources():
    data = fig.build(SEEDS, base=ROOT)
    assert {k: v for k, v in STORED.items() if k != "files"} == data
    for name in STORED["files"]:
        assert (ROOT / name).stat().st_size > 1000, name
    assert fig.main(["--check"]) == 0


def test_arm_points_are_copies_of_the_seed_record():
    arms = {a["label"]: a for a in REC["arms"]}
    rows = STORED["arms_dev"]
    assert len(rows) == len(arms) * len(fig.TASKS)
    for r in rows:
        a = arms[r["arm"]]
        assert (r["point"], r["lo"], r["hi"]) == tuple(a["pooled"][r["task"]][k] for k in ("point", "lo", "hi"))
        assert r["per_seed"] == {str(s): a["per_seed"][str(s)][r["task"]] for s in a["seeds"]}
        assert 0.0 <= r["lo"] <= r["point"] <= r["hi"] <= 1.0


def test_cost_points_are_the_measured_seconds():
    rows = STORED["cost_dev"]
    assert [(r["arm"], r["seed"]) for r in rows] == [(f["label"], f["seed"]) for f in REC["files"]]
    for r in rows:
        s = json.loads((ROOT / r["summary"]).read_text())
        assert r["train_seconds"] == s["train"]["train_seconds"]
        assert r["eval_seconds"] == s["eval_dev"]["eval_seconds"]
        assert r["gpu_seconds"] == r["train_seconds"] + r["eval_seconds"]
        assert s["eval_dev"]["split"] == "dev" and s["train"]["gpu"] == "Tesla T4"


def test_other_split_is_refused(tmp_path):
    rec = copy.deepcopy(REC)
    rec["split"] = "test_id"
    path = _copy_tree(tmp_path, rec)
    with pytest.raises(SystemExit, match="dev split only"):
        fig.build(path, base=tmp_path)


def test_summary_that_disagrees_is_refused(tmp_path):
    rec = copy.deepcopy(REC)
    rec["arms"][0]["per_seed"]["3407"][fig.MACRO] += 0.01
    path = _copy_tree(tmp_path, rec)
    with pytest.raises(SystemExit, match="reports macro"):
        fig.build(path, base=tmp_path)


def test_missing_summary_is_refused(tmp_path):
    path = _copy_tree(tmp_path, REC)
    ((tmp_path / REC["files"][0]["file"]).parent / "summary.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        fig.build(path, base=tmp_path)


def test_check_notices_a_stale_record(tmp_path):
    path = _copy_tree(tmp_path, REC)
    out = tmp_path / "figs"
    out.mkdir()
    stale = dict(STORED)
    stale["root_graphs"] = STORED["root_graphs"] + 1
    (out / "figures_dev.json").write_text(json.dumps(stale))
    assert fig.main(["--seeds", str(path), "--base", str(tmp_path), "--out-dir", str(out), "--check"]) == 1


def test_drawing_writes_all_files_with_full_axes(tmp_path):
    pytest.importorskip("matplotlib")
    path = _copy_tree(tmp_path, REC)
    out = tmp_path / "figs"
    assert fig.main(["--seeds", str(path), "--base", str(tmp_path), "--out-dir", str(out), "--paper-dir", str(tmp_path / "paper")]) == 0
    for name in ("arms_dev.svg", "arms_dev.png", "cost_dev.svg", "cost_dev.png", "errors_dev.svg", "errors_dev.png",
                 "figures_dev.json", "README.md"):
        assert (out / name).exists(), name
    assert (tmp_path / "paper/arms_dev.pdf").exists()
    assert (tmp_path / "paper/errors_dev.pdf").exists()
    plt = fig._mpl()
    data = fig.build(path, base=tmp_path)
    for draw in (fig.draw_arms, fig.draw_cost):
        f = draw(data, plt)
        assert f.axes[0].get_ylim() == (0.0, 1.0)
        plt.close(f)
    f = fig.draw_errors(data, plt)
    assert len(f.axes) == 3 and all(ax.get_ylim() == (0.0, 1.0) for ax in f.axes)
    for ax in f.axes:  # every bar stack ends at 1: nothing is left undrawn
        tops = {}
        for patch in ax.patches:
            tops[patch.get_x()] = max(tops.get(patch.get_x(), 0.0), patch.get_y() + patch.get_height())
        assert len(tops) == 12 and all(abs(v - 1.0) < 1e-9 for v in tops.values())
    plt.close(f)
    f = fig.draw_cost(data, plt)
    assert f.axes[0].get_xlim()[0] == 0.0
    plt.close(f)


def test_paper_cost_sentence_states_recorded_seconds():
    tex = " ".join((ROOT / "paper/sections/results.tex").read_text().split())
    rows = STORED["cost_dev"]
    e2e = [r["eval_seconds"] for r in rows if r["arm"] in ("B", "C", "D")]
    e = [r["eval_seconds"] for r in rows if r["arm"] == "E"]
    tr = [r["train_seconds"] for r in rows]
    assert len(rows) == 12 and all(r["eval_samples"] == 1200 for r in rows)
    assert "one run of B, C or D took $%.1f$ to $%.1f$ T4 seconds" % (min(e2e), max(e2e)) in tex
    assert "E took $%.1f$ to $%.1f$ seconds" % (min(e), max(e)) in tex
    assert "between $%.1f$ and $%.1f$ seconds for all twelve runs" % (min(tr), max(tr)) in tex
    ratio = (sum(e) / len(e)) / (sum(e2e) / len(e2e))
    assert 3.0 < ratio < 4.0, ratio  # "more than three times"
    assert "\\includegraphics[width=\\textwidth]{figures/arms_dev.pdf}" in tex


def test_error_rows_are_copies_of_the_summaries_and_partition():
    rows = STORED["errors_dev"]
    assert len(rows) == len(REC["files"]) * 3
    per_seed = {a["label"]: a["per_seed"] for a in REC["arms"]}
    for r in rows:
        ev = json.loads((ROOT / r["summary"]).read_text())["eval_dev"]
        m = ev["metrics"]["per_task"][r["task"]]
        assert ev["split"] == "dev"
        assert (r["correct"], r["format_failure"], r["n"]) == (m["accuracy"], m["format_failure_rate"], m["n"])
        assert r["correct"] == per_seed[r["arm"]][str(r["seed"])][r["task"]]
        assert abs(r["correct"] + r["wrong_well_formed"] + r["format_failure"] - 1.0) < 1e-12
        assert min(r["correct"], r["wrong_well_formed"], r["format_failure"]) >= 0.0
        if r["task"] == fig.REPAIR:
            assert r["repair_detail"] == {k: m[k] for k in fig.REPAIR_DETAIL}
        else:
            assert "repair_detail" not in r
        if r["arm"] == "E" and r["task"] == "critical_edge":
            assert r["premise_failures"] == ev["reasons"][fig.PREMISE] == round(r["format_failure"] * r["n"])
        else:
            assert "premise_failures" not in r


def test_error_rows_refuse_a_summary_that_disagrees(tmp_path):
    path = _copy_tree(tmp_path, REC)
    target = (tmp_path / REC["files"][-1]["file"]).parent / "summary.json"   # an arm E run
    s = json.loads(target.read_text())
    s["eval_dev"]["reasons"][fig.PREMISE] += 1
    target.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="premise failures"):
        fig.build(path, base=tmp_path)
    s["eval_dev"]["reasons"][fig.PREMISE] -= 1
    s["eval_dev"]["metrics"]["per_task"]["reachability"]["format_failure_rate"] = 0.5
    target.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="exceed 1"):
        fig.build(path, base=tmp_path)


def test_paper_error_paragraph_states_recorded_counts():
    tex = " ".join((ROOT / "paper/sections/results.tex").read_text().split())
    rows = STORED["errors_dev"]
    order = [3407, 2027, 9173]

    def count(arm, seed, task, key):
        (r,) = [r for r in rows if (r["arm"], r["seed"], r["task"]) == (arm, seed, task)]
        return int(round(r[key] * r["n"])), r["n"]

    for arm in ("C", "D"):
        assert all(r["format_failure"] == 0.0 for r in rows if r["arm"] == arm)
    assert all(r["format_failure"] == 0.0 for r in rows if r["arm"] == "B" and r["task"] != fig.REPAIR)
    b = [count("B", s, fig.REPAIR, "format_failure") for s in order]
    assert "$%d$, $%d$ and $%d$ of $%d$ on repair (seeds 3407, 2027, 9173)" % (b[0][0], b[1][0], b[2][0], b[0][1]) in tex
    e_ff = [count("E", s, "critical_edge", "format_failure")[0] for s in order]
    e_wrong = [count("E", s, "critical_edge", "wrong_well_formed")[0] for s in order]
    e_err = [a + w for a, w in zip(e_ff, e_wrong)]
    assert "for $%d$, $%d$ and $%d$ of its $400$ critical-edge questions per run the premise fails" % tuple(e_ff) in tex
    assert "$%d$ of $%d$, $%d$ of $%d$ and $%d$ of $%d$ of E's critical-edge errors" % (
        e_ff[0], e_err[0], e_ff[1], e_err[1], e_ff[2], e_err[2]) in tex
    assert all(2 * f > e for f, e in zip(e_ff, e_err))  # "most"
    for f in REC["files"]:  # "the recorded lenient score equals the strict one ... in all nine runs"
        if f["label"] in ("B", "C", "D"):
            per_task = json.loads(((ROOT / f["file"]).parent / "summary.json").read_text())["eval_dev"]["metrics"]["per_task"]
            assert all(v["lenient_accuracy"] == v["accuracy"] for v in per_task.values())
    assert sum(f["label"] in ("B", "C", "D") for f in REC["files"]) == 9
    assert all(r["format_failure"] == 0.0 for r in rows if r["arm"] == "E" and r["task"] != "critical_edge")
    assert "\\includegraphics[width=\\textwidth]{figures/errors_dev.pdf}" in tex
