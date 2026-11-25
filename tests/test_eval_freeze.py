"""eval_freeze.py: the locked-test kernels must run the evaluation code jobF01 ran."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
import eval_freeze  # noqa: E402


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo)] + list(a), check=True, capture_output=True, text=True).stdout.strip()


def _repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src" / "trace_vr").mkdir(parents=True)
    (r / "scripts").mkdir()
    (r / "src" / "trace_vr" / "m.py").write_text("x = 1\n")
    (r / "scripts" / "eval_arm.py").write_text("print(1)\n")
    (r / "scripts" / "final_slices.py").write_text("print(2)\n")
    _git(r, "init", "-q")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a")
    c = _git(r, "rev-parse", "HEAD")
    rec = tmp_path / "freeze.json"
    rec.write_text(json.dumps({"commit": c, "digest": eval_freeze.digest(c, r)}))
    return r, rec


def _commit(r, path, text):
    (r / path).write_text(text)
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "b")


def test_reporting_change_keeps_freeze(tmp_path):
    r, rec = _repo(tmp_path)
    _commit(r, "scripts/final_slices.py", "print(3)\n")
    ok, msg = eval_freeze.check("HEAD", rec, r)
    assert ok and msg.endswith("ok")


def test_eval_change_breaks_freeze(tmp_path):
    r, rec = _repo(tmp_path)
    _commit(r, "src/trace_vr/m.py", "x = 2\n")
    ok, msg = eval_freeze.check("HEAD", rec, r)
    assert not ok and "src/trace_vr/m.py" in msg and msg.endswith("stop")


def test_recorded_freeze_holds_at_head():
    rec = json.loads((ROOT / "scripts" / "kaggle" / "eval_freeze.json").read_text())
    assert rec["paths"] == eval_freeze.FROZEN_PATHS
    assert eval_freeze.check("HEAD")[0]
