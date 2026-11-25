"""Seed runs: only the pre-registered seeds are accepted, and the run directory names agree everywhere."""
import json, re, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "scripts/colab/job.sh"
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}


def _protocol_seeds():
    text = (ROOT / "docs/protocol.md").read_text()
    m = re.search(r"Three training seeds \(([\d, ]+)\)", text)
    return tuple(int(s) for s in m.group(1).split(","))


def test_seed_lists_match_protocol():
    seeds = _protocol_seeds()
    assert seeds == (3407, 2027, 9173)
    sys.path.insert(0, str(ROOT / "scripts/kaggle"))
    import make_kernel
    assert make_kernel.SEEDS == seeds and make_kernel.DEFAULT_SEED == seeds[0]
    assert "case \"$SEED\" in %s)" % "|".join(map(str, seeds)) in JOB.read_text()


def test_job_sh_rejects_unregistered_seed(tmp_path):
    res = subprocess.run(["bash", str(JOB), "jobX", "B", "3", "image"], capture_output=True, text=True,
                         env=dict(ENV, TRACE_SEED="1", TRACE_HOME=str(tmp_path), TRACE_TMP=str(tmp_path)))
    assert res.returncode == 2 and "not pre-registered" in res.stderr


@pytest.mark.parametrize("seed,name", [("3407", "pilot3_B"), ("2027", "pilot3_B_s2027"), ("9173", "pilot3_B_s9173")])
def test_job_sh_run_name(seed, name):
    # Execute only the seed/run-name lines of job.sh, so the rule tested is the one that runs on the GPU.
    lines = JOB.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("EPOCHS="))
    end = next(i for i, l in enumerate(lines) if l.startswith("RUN="))
    snippet = "ARM=B\n" + "\n".join(lines[start:end + 1]) + "\necho $RUN"
    res = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, env=dict(ENV, TRACE_SEED=seed))
    assert res.returncode == 0 and res.stdout.strip() == "runs/" + name


def test_make_kernel_seed(tmp_path):
    bad = subprocess.run([sys.executable, "scripts/kaggle/make_kernel.py", "jobS", "B", "3:image", "--seed", "5"],
                         cwd=ROOT, capture_output=True, text=True, env=ENV)
    assert bad.returncode != 0 and "not pre-registered" in bad.stderr
    ok = subprocess.run([sys.executable, "scripts/kaggle/make_kernel.py", "jobSeedTest", "B", "3:image", "--seed", "2027"],
                        cwd=ROOT, capture_output=True, text=True, env=ENV)
    out = ROOT / "dist/kaggle/jobSeedTest"
    try:
        assert ok.returncode == 0, ok.stderr
        kernel = (out / "kernel.py").read_text()
        assert 'TRACE_SEED="2027"' in kernel and "seed=2027" in kernel.splitlines()[0]
    finally:
        for f in out.glob("*"):
            f.unlink()
        if out.exists():
            out.rmdir()


def test_collect_seed_run_dir(tmp_path):
    if not (ROOT / "data/pilot/dev.samples.jsonl").exists():
        pytest.skip("pilot data not built")
    from test_kaggle_collect import _fake_kernel
    kdir = _fake_kernel(tmp_path)
    (kdir / "runs/pilot3_X").rename(kdir / "runs/pilot3_X_s2027")
    out = tmp_path / "out"
    res = subprocess.run([sys.executable, "scripts/kaggle/collect.py", "jobX", "X", "--seed", "2027",
                          "--kernel-dir", str(kdir), "--out", str(out)], cwd=ROOT, capture_output=True, text=True, env=ENV)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.splitlines()[0])["run_id"] == "pilot3_X_s2027_jobX"
    missing = subprocess.run([sys.executable, "scripts/kaggle/collect.py", "jobX", "X", "--kernel-dir", str(kdir),
                              "--out", str(tmp_path / "o2")], cwd=ROOT, capture_output=True, text=True, env=ENV)
    assert missing.returncode != 0


JOB_E = ROOT / "scripts/colab/job_e.sh"


@pytest.mark.parametrize("seed,train,run", [("3407", "pilot3_C_desc", "pilot3_E"),
                                            ("2027", "pilot3_C_desc_s2027", "pilot3_E_s2027"),
                                            ("9173", "pilot3_C_desc_s9173", "pilot3_E_s9173")])
def test_job_e_sh_seed_dirs(seed, train, run):
    # Seed 3407 must keep job13's directories (its collected files and the parser sha256 refer to them).
    lines = JOB_E.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("EPOCHS="))
    end = next(i for i, l in enumerate(lines) if l.startswith("if [ \"$SEED\" != 3407 ]"))
    snippet = "\n".join(lines[start:end + 1]) + "\necho $SEED $TRAIN $RUN"
    res = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, env=dict(ENV, TRACE_SEED=seed))
    assert res.returncode == 0 and res.stdout.split() == [seed, "runs/" + train, "runs/" + run]
    assert "$SEED $CKPT" in JOB_E.read_text()   # the parser is trained with that seed


def test_job_e_sh_rejects_unregistered_seed(tmp_path):
    res = subprocess.run(["bash", str(JOB_E), "jobX", "3"], capture_output=True, text=True,
                         env=dict(ENV, TRACE_SEED="1", TRACE_HOME=str(tmp_path), TRACE_TMP=str(tmp_path)))
    assert res.returncode == 2 and "not pre-registered" in res.stderr


def test_make_kernel_arm_e_seed():
    ok = subprocess.run([sys.executable, "scripts/kaggle/make_kernel.py", "jobSeedE", "E", "3:image", "--seed", "9173"],
                        cwd=ROOT, capture_output=True, text=True, env=ENV)
    out = ROOT / "dist/kaggle/jobSeedE"
    try:
        assert ok.returncode == 0, ok.stderr
        kernel = (out / "kernel.py").read_text()
        assert 'TRACE_SEED="9173"' in kernel and "job_e.sh" in kernel
    finally:
        for f in out.glob("*"):
            f.unlink()
        if out.exists():
            out.rmdir()
