"""scripts/seed_pairs.py: per-seed twin-pair and answer-bias counts equal analysis.py on each flags file,
sums are sums, and the committed report equals the generator output."""
import json, subprocess, sys
from pathlib import Path
import pytest
from trace_vr import analysis, arms, bootstrap, dataset

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests/pilot.json"
built = (ROOT / "data/pilot/dev.samples.jsonl").exists()
REPORT = ROOT / "reports/pilot/pairs_seeds_dev.json"


def _run(*args):
    return subprocess.run([sys.executable, "scripts/seed_pairs.py", str(MAN), *args], cwd=ROOT,
                          capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})


def _entries():
    return ["%s@%d=%s" % (f["label"], f["seed"], f["file"]) for f in json.loads(REPORT.read_text())["files"]]


@pytest.mark.skipif(not built, reason="pilot data not built")
def test_committed_report_equals_generator(tmp_path):
    out, md = tmp_path / "p.json", tmp_path / "p.md"
    res = _run(*_entries(), "--out-json", str(out), "--out-md", str(md))
    assert res.returncode == 0, res.stderr
    assert out.read_text() == REPORT.read_text()
    assert md.read_text() == REPORT.with_suffix(".md").read_text()


@pytest.mark.skipif(not built, reason="pilot data not built")
def test_counts_match_analysis_and_sum():
    t = json.loads(REPORT.read_text())
    _, _, by_split, _ = dataset.load_dataset(MAN)
    rows = arms.eval_samples(by_split[t["split"]], t["root_graphs"])
    files = {(f["label"], f["seed"]): f["file"] for f in t["files"]}
    assert {l for l, _ in files} == {"B", "C", "D", "E"} and len(files) == 12
    for a in t["arms"]:
        tot = {}
        for s in a["seeds"]:
            p = json.loads((ROOT / files[(a["label"], s)]).read_text())
            flags = bootstrap.decode_flags(rows, p)
            fmt = bootstrap.decode_flags(rows, p, "format_ok")
            assert a["per_seed"][str(s)]["pairs"] == analysis.pair_table(rows, flags)
            assert a["per_seed"][str(s)]["answer_bias"] == analysis.answer_bias(rows, flags, fmt)
            for task, cell in a["per_seed"][str(s)]["pairs"].items():
                for k, v in cell.items():
                    tot[(task, k)] = tot.get((task, k), 0) + v
        for task, cell in a["summed"]["pairs"].items():
            for k in ("pairs", "both_right", "one_right", "both_wrong"):
                assert cell[k] == tot[(task, k)]
            assert cell["both_right"] + cell["one_right"] + cell["both_wrong"] == cell["pairs"]
        for task, e in a["summed"]["answer_bias"].items():
            said = n = 0
            for s in a["seeds"]:
                p = json.loads((ROOT / files[(a["label"], s)]).read_text())
                flags = bootstrap.decode_flags(rows, p)
                fmt = bootstrap.decode_flags(rows, p, "format_ok")
                for r in rows:
                    if r["task"] == task and r["metadata"]["image_variant"] in ("root", "edit") and fmt[r["sample_id"]]:
                        truth = bool(r["training_target"]["answer"])
                        said += truth if flags[r["sample_id"]] else (not truth)
                        n += 1
            assert e["said_true"] == said and e["n"] - e["no_answer"] == n


def test_cli_refusals(tmp_path):
    assert _run("B@1234=runs/pilot3_B_kaggle/flags_dev.json").returncode != 0      # seed not pre-registered
    assert _run("B=runs/pilot3_B_kaggle/flags_dev.json").returncode != 0           # no seed
    res = _run("B@3407=runs/pilot3_B_kaggle/flags_dev.json", "B@3407=runs/pilot3_B_s2027/flags_dev.json")
    assert res.returncode != 0 and "duplicate" in res.stderr
