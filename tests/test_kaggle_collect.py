"""collect.py copies Kaggle outputs and refuses a log whose flags do not match the predictions."""
import json, subprocess, sys
from pathlib import Path
import pytest
from trace_vr import arms, dataset

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests/pilot.json"
pytestmark = pytest.mark.skipif(not (ROOT / "data/pilot/dev.samples.jsonl").exists(), reason="pilot data not built")


def _run(*args, **kw):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}, **kw)


def _fake_kernel(tmp_path, tamper=False, other_images=False):
    man, _, by_split, _ = dataset.load_dataset(MAN)
    rows = arms.select_samples(by_split["dev"], 3, "main")
    kdir = tmp_path / "k"; run = kdir / "runs/pilot3_X"; run.mkdir(parents=True)
    with open(run / "predictions_dev.jsonl", "w") as fh:
        for s in rows:
            fh.write(json.dumps({"sample_id": s["sample_id"], "task": s["task"], "text": '{"answer": true}'}) + "\n")
    (run / "train_summary.json").write_text(json.dumps({"arm": "X", "train_seconds": 10.0, "gpu": "Tesla T4"}))
    (run / "metrics_dev.json").write_text(json.dumps({"eval_root_graphs": 3, "eval_seconds": 5.5}))
    res = _run("scripts/export_flags.py", str(MAN), "dev", "3", str(run), "image")
    assert res.returncode == 0, res.stderr
    line = [l for l in res.stdout.splitlines() if l.startswith("FLAGS-SUMMARY ")][-1]
    if tamper:
        d = json.loads(line[14:]); d["predictions_sha256"] = "0" * 64; line = "FLAGS-SUMMARY " + json.dumps(d)
    if other_images:
        d = json.loads(line[14:]); d["images_digest"] = "f" * 64; line = "FLAGS-SUMMARY " + json.dumps(d)
    (kdir / "jobX.log").write_text("JOB-START jobX\npeft 0.21.2 transformers 5.22.2\n" + line + "\nJOB-END\n")
    return kdir


def test_collect_writes_summary_and_flags(tmp_path):
    kdir = _fake_kernel(tmp_path)
    out = tmp_path / "out"
    res = _run("scripts/kaggle/collect.py", "jobX", "X", "--kernel-dir", str(kdir), "--out", str(out))
    assert res.returncode == 0, res.stderr
    budget = json.loads(res.stdout.splitlines()[0])
    assert budget["gpu_seconds"] == 15.5 and budget["platform"] == "kaggle"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["train"]["arm"] == "X" and summary["eval_dev"]["eval_seconds"] == 5.5
    flags = json.loads((out / "flags_dev.json").read_text())
    assert flags["eval_root_graphs"] == 3 and flags["content_digest"] == json.loads(MAN.read_text())["content_digest"]
    again = _run("scripts/kaggle/collect.py", "jobX", "X", "--kernel-dir", str(kdir), "--out", str(out))
    assert again.returncode != 0 and "--force" in (again.stderr + again.stdout)


def test_collect_rejects_mismatched_log(tmp_path):
    kdir = _fake_kernel(tmp_path, tamper=True)
    res = _run("scripts/kaggle/collect.py", "jobX", "X", "--kernel-dir", str(kdir), "--out", str(tmp_path / "o"))
    assert res.returncode != 0 and "predictions_sha256" in (res.stderr + res.stdout)


def test_collect_accepts_other_machine_images_digest(tmp_path):
    kdir = _fake_kernel(tmp_path, other_images=True)
    out = tmp_path / "out"
    res = _run("scripts/kaggle/collect.py", "jobX", "X", "--kernel-dir", str(kdir), "--out", str(out))
    assert res.returncode == 0, res.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["images_digest"]["kernel"] == "f" * 64
    assert summary["images_digest"]["local"] == json.loads(MAN.read_text())["images_digest"]


def test_collect_arm_e_reads_train_summary_of_parser(tmp_path):
    kdir = _fake_kernel(tmp_path)
    (kdir / "runs/pilot3_X").rename(kdir / "runs/pilot3_E")
    parser = kdir / "runs/pilot3_C_desc"; parser.mkdir()
    (kdir / "runs/pilot3_E/train_summary.json").rename(parser / "train_summary.json")
    log = (kdir / "jobX.log").read_text().replace('"run": "pilot3_X"', '"run": "pilot3_E"')
    (kdir / "jobX.log").write_text(log)
    out = tmp_path / "out"
    res = _run("scripts/kaggle/collect.py", "jobX", "E", "--kernel-dir", str(kdir), "--out", str(out))
    assert res.returncode == 0, res.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["train_dir"] == "pilot3_C_desc" and summary["train"]["arm"] == "X"
    assert json.loads(res.stdout.splitlines()[0])["gpu_seconds"] == 15.5


def test_collect_arm_e_seed_reads_seeded_parser(tmp_path):
    # job_e.sh at seed 2027 writes runs/pilot3_C_desc_s2027 (parser) and runs/pilot3_E_s2027 (predictions)
    kdir = _fake_kernel(tmp_path)
    (kdir / "runs/pilot3_X").rename(kdir / "runs/pilot3_E_s2027")
    parser = kdir / "runs/pilot3_C_desc_s2027"; parser.mkdir()
    (kdir / "runs/pilot3_E_s2027/train_summary.json").rename(parser / "train_summary.json")
    log = (kdir / "jobX.log").read_text().replace('"run": "pilot3_X"', '"run": "pilot3_E_s2027"')
    (kdir / "jobX.log").write_text(log)
    out = tmp_path / "out"
    res = _run("scripts/kaggle/collect.py", "jobX", "E", "--seed", "2027", "--kernel-dir", str(kdir), "--out", str(out))
    assert res.returncode == 0, res.stderr
    assert json.loads((out / "summary.json").read_text())["train_dir"] == "pilot3_C_desc_s2027"
