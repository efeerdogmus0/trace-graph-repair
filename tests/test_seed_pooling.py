"""Seed pooling (src/trace_vr/seeds.py): reduces to the single-seed bootstrap, pools by averaging,
and applies the pre-registered reading rule."""
import json, subprocess, sys
from pathlib import Path
import pytest
from trace_vr import seeds

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests/pilot.json"
built = (ROOT / "data/pilot/dev.samples.jsonl").exists()


def _run(*args):
    return subprocess.run([sys.executable, "scripts/seed_table.py", str(MAN), *args], cwd=ROOT,
                          capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})


@pytest.mark.skipif(not built, reason="pilot data not built")
def test_one_seed_reproduces_go_no_go(tmp_path):
    out = tmp_path / "s.json"
    res = _run("--ref", "B", "B@3407=runs/pilot3_B_kaggle/flags_dev.json", "D2@3407=runs/pilot3_D2/flags_dev.json",
               "--out-json", str(out))
    assert res.returncode == 0, res.stderr
    t = json.loads(out.read_text())
    rec = json.loads((ROOT / "runs/go_no_go_rev1.json").read_text())["entries"]["d2_vs_b"]
    d = {a["label"]: a for a in t["arms"]}["D2"]
    m = d["vs_ref"]["non_edit_macro_accuracy"]
    assert (m["point"], m["lo"], m["hi"]) == pytest.approx((rec["point"], rec["lo"], rec["hi"]), abs=1e-9)
    assert d["reading"] == "negative"


@pytest.mark.skipif(not built, reason="pilot data not built")
def test_pooled_point_is_mean_of_seeds(tmp_path):
    out = tmp_path / "s.json"
    res = _run("--ref", "B", "B@3407=runs/pilot3_B_kaggle/flags_dev.json", "B@2027=runs/pilot3_B/flags_dev.json",
               "--out-json", str(out), "--boot", "200")
    assert res.returncode == 0, res.stderr
    b = json.loads(out.read_text())["arms"][0]
    per = [v["non_edit_macro_accuracy"] for v in b["per_seed"].values()]
    assert per == pytest.approx([0.8075, 0.79625], abs=1e-9)  # results_dev.md: B_colab 0.807, B_kaggle 0.796
    assert b["pooled"]["non_edit_macro_accuracy"]["point"] == pytest.approx(sum(per) / 2)
    assert b["seed_spread"] == pytest.approx({"min": min(per), "max": max(per)})


def test_cli_refuses_unregistered_seed():
    res = _run("--ref", "B", "B@1=runs/pilot3_B/flags_dev.json")
    assert res.returncode != 0 and "pre-registered" in (res.stderr + res.stdout)


def _toy():
    samples, flags_hi, flags_lo = [], {}, {}
    for g in range(20):
        for task in ("reachability", "critical_edge"):
            sid = "g%d_%s" % (g, task)
            samples.append({"sample_id": sid, "graph_group_id": "g%d" % g, "task": task,
                            "metadata": {"image_variant": "root"}})
            flags_hi[sid] = g % 10 != 0   # 0.9
            flags_lo[sid] = g % 2 == 0    # 0.5
    return samples, flags_hi, flags_lo


def test_identical_seeds_equal_single_seed():
    samples, hi, lo = _toy()
    one = seeds.pooled_bootstrap(samples, {"A": {3407: hi}, "R": {3407: lo}}, "R", n_boot=300)
    three = seeds.pooled_bootstrap(samples, {"A": {s: hi for s in seeds.PREREGISTERED_SEEDS},
                                             "R": {s: lo for s in seeds.PREREGISTERED_SEEDS}}, "R", n_boot=300)
    for k, e in one["arms"][0]["vs_ref"].items():
        assert three["arms"][0]["vs_ref"][k] == pytest.approx(e, abs=1e-12)
    assert one["arms"][0]["reading"] == "incomplete_seeds"
    assert three["arms"][0]["reading"] == "positive_clear"


def test_reading_rules():
    full = list(seeds.PREREGISTERED_SEEDS)
    up = {"point": 0.06, "lo": 0.01, "hi": 0.1}
    assert seeds.reading(up, {"3407": .05, "2027": .07, "9173": .06}, full, full) == "positive_clear"
    assert seeds.reading(dict(up, point=0.03), {"3407": .02, "2027": .03, "9173": .04}, full, full) == "positive"
    assert seeds.reading(up, {"3407": .1, "2027": .1, "9173": -.01}, full, full) == "inconclusive"
    assert seeds.reading(up, {"3407": .05, "2027": .07}, full[:2], full[:2]) == "incomplete_seeds"
    down = {"point": -0.1, "lo": -0.15, "hi": -0.05}
    assert seeds.reading(down, {"3407": -.1}, full[:1], full[:1]) == "negative"
    assert seeds.reading({"point": 0.0, "lo": -0.02, "hi": 0.02}, {"3407": 0.0}, full, full) == "inconclusive"
