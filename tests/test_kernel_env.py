"""The environment line of final kernels: the probe runs anywhere, and the reader reads only that line."""
import importlib.util, json, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/kaggle" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_prints_one_line_even_without_torch():
    probe = load("make_final_kernel").ENV_PROBE
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    assert len(lines) == 1 and lines[0].startswith("ENV-SUMMARY ")
    env = json.loads(lines[0][len("ENV-SUMMARY "):])
    assert env["python"] == "%d.%d.%d" % sys.version_info[:3]
    assert {"torch", "transformers", "peft", "cuda", "cudnn", "gpus", "image", "numpy"} <= set(env)
    import numpy
    assert env["numpy"] == numpy.__version__   # a real version where the module exists, null where it does not


def test_final_kernel_runs_the_probe_outside_the_job_log():
    text = load("make_final_kernel").KERNEL
    probe_at, job_log_at = text.index("{env_probe}"), text.index('log = open(work + "/{job}.log"')
    assert text.index("pip -q install") < probe_at < job_log_at   # after the pins, before <job>.log is opened


def _kernel_log(tmp_path, job, lines):
    d = tmp_path / job
    d.mkdir()
    (d / ("trace-%s.log" % job.lower())).write_text(
        json.dumps([{"stream_name": "stdout", "time": float(i), "data": l + "\n"} for i, l in enumerate(lines)]))


def test_reader_takes_only_the_env_line_and_reports_drift(tmp_path, capsys):
    ke = load("kernel_env")
    a = {"torch": "9.9.0", "cuda": "12.4", "gpus": ["Tesla T4", "Tesla T4"]}
    _kernel_log(tmp_path, "jobX1", ["fonts ok 6", "ENV-SUMMARY " + json.dumps(a), "EVAL-SUMMARY {\"macro\": 0.5}"])
    _kernel_log(tmp_path, "jobX2", ["fonts ok 6", "ENV-SUMMARY " + json.dumps(a)])
    _kernel_log(tmp_path, "jobX3", ["fonts ok 6", "ENV-SUMMARY " + json.dumps(dict(a, torch="9.9.1"))])
    _kernel_log(tmp_path, "jobOld", ["fonts ok 6", "Tesla T4, 15360 MiB"])
    assert ke.read_env(tmp_path / "jobX1" / "trace-jobx1.log") == a
    assert ke.read_env(tmp_path / "jobOld" / "trace-jobold.log") is None
    out = tmp_path / "env.json"
    assert ke.main(["jobX1", "jobX2", "jobOld", "--kdir", str(tmp_path), "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["not_recorded"] == ["jobOld"] and got["differing_fields"] == []
    assert "macro" not in out.read_text() and "macro" not in capsys.readouterr().out   # no score leaves the log
    assert ke.main(["jobX1", "jobX3", "--kdir", str(tmp_path)]) == 1
    assert ke.differing_fields({"a": a, "b": dict(a, torch="9.9.1"), "c": None}) == ["torch"]


def test_reader_refuses_a_missing_or_doubled_log(tmp_path):
    ke = load("kernel_env")
    with pytest.raises(SystemExit):
        ke.main(["jobNone", "--kdir", str(tmp_path)])
    _kernel_log(tmp_path, "jobX1", ["ENV-SUMMARY {}", "ENV-SUMMARY {}"])
    with pytest.raises(SystemExit):
        ke.main(["jobX1", "--kdir", str(tmp_path)])
