"""Perception split: readability from extraction, subset bootstrap, pre-registered reading rule."""
import json, subprocess, sys
from pathlib import Path
import pytest
from trace_vr import arms, bootstrap, dataset, perception

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests/pilot.json"
needs_data = pytest.mark.skipif(not (ROOT / "data/pilot/dev.samples.jsonl").exists(), reason="pilot data not built")


def test_reading_rule():
    r = lambda lo, hi: {"point": (lo + hi) / 2, "lo": lo, "hi": hi}
    assert perception.reading(r(-0.2, -0.1), r(-0.3, -0.1), 50, 50) == "reasoning_gap"
    assert perception.reading(r(-0.05, 0.02), r(-0.3, -0.1), 50, 50) == "perception_gap"
    assert perception.reading(r(-0.05, 0.02), r(-0.05, 0.1), 50, 50) == "no_gap"
    assert perception.reading(r(0.01, 0.1), r(-0.3, -0.1), 50, 50) == "unclear"
    assert perception.reading(r(-0.2, -0.1), r(-0.3, -0.1), 29, 50) == "too_few"
    assert perception.reading(None, r(-0.3, -0.1), 50, 50) == "too_few"


def _truth_text(t):
    return json.dumps({"nodes": sorted(t["nodes"]), "edges": sorted([list(e) for e in t["edges"]])})


@needs_data
def test_split_on_pilot_flags(tmp_path):
    man, _, by_split, _ = dataset.load_dataset(MAN)
    rows = arms.eval_samples(by_split["dev"], 200)
    imgs = []
    truth = {}
    for s in rows:
        if s["image_id"] not in truth:
            truth[s["image_id"]] = s["private_truth"]; imgs.append(s["image_id"])
    # every other image read exactly, the rest garbage
    ext = {i: (_truth_text(truth[i]) if n % 2 == 0 else "not json") for n, i in enumerate(imgs)}
    readable = perception.readable_images(rows, ext)
    assert sum(readable.values()) == (len(imgs) + 1) // 2
    with pytest.raises(ValueError):
        perception.readable_images(rows, {})
    p = tmp_path / "extraction_dev.jsonl"
    p.write_text("".join(json.dumps({"image_id": i, "text": t}) + "\n" for i, t in ext.items()))
    out = tmp_path / "s.json"
    res = subprocess.run([sys.executable, "scripts/perception_split.py", str(MAN), str(p), "--ref", "B",
                          "B=runs/pilot3_B_kaggle/flags_dev.json", "D2=runs/pilot3_D2/flags_dev.json",
                          "--out-json", str(out), "--boot", "300"], cwd=ROOT, capture_output=True, text=True,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert res.returncode == 0, res.stderr
    t = json.loads(out.read_text())
    main = bootstrap._main_rows(rows)
    sub = [s for s in main if readable[s["image_id"]]]
    assert t["subsets"]["readable"]["samples"] == len(sub)
    assert t["subsets"]["readable"]["samples"] + t["subsets"]["unreadable"]["samples"] == len(main)
    fb = bootstrap.decode_flags(rows, json.loads((ROOT / "runs/pilot3_B_kaggle/flags_dev.json").read_text()))
    direct = bootstrap.cluster_bootstrap(sub, fb, n_boot=300)["a"][perception.MACRO]
    assert t["arms"]["B"]["readable"]["acc"] == pytest.approx(direct)
    assert t["arms"]["D2"]["reading"] in {"reasoning_gap", "perception_gap", "no_gap", "unclear", "too_few"}
