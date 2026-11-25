"""final_reports.py runs the seven locked-test scripts in the fixed order and stops at the first refusal.
With the real queue (not all twelve kernels collected) the main table refuses, so nothing is written."""
import importlib.util, json, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("final_reports", ROOT / "scripts/final_reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_order_and_outputs(tmp_path):
    m = load()
    cmds = m.commands(tmp_path, queue="q.json")
    assert [n for n, _ in cmds] == ["final_table", "final_primary", "final_invariance", "final_pairs", "final_density",
                                    "final_slices", "final_metrics", "final_floor"]
    for name, cmd in cmds:
        assert (ROOT / "scripts" / (name + ".py")).exists()
        for ext in ("json", "md", "tex"):
            assert str(tmp_path / ("%s.%s" % (name, ext))) in cmd
        assert cmd[-2:] == ["--queue", "q.json"]


def test_stops_at_first_refusal(tmp_path, monkeypatch):
    m = load()
    calls = []

    class R:
        def __init__(self, code): self.returncode, self.stdout, self.stderr = code, "", "refused: missing"

    def fake(cmd, **kw):
        calls.append(Path(cmd[1]).stem)
        return R(1 if Path(cmd[1]).stem == "final_pairs" else 0)

    monkeypatch.setattr(m.subprocess, "run", fake)
    done, refused = m.run(tmp_path)
    assert done == ["final_table", "final_primary", "final_invariance"] and refused == "final_pairs"
    assert calls == ["final_table", "final_primary", "final_invariance", "final_pairs"]


def test_real_queue_refuses_and_writes_nothing(tmp_path):
    q = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())
    if all((ROOT / "runs/kaggle" / j["job"]).exists() for j in q["jobs"]):
        pytest.skip("all final runs collected")
    out = tmp_path / "final"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/final_reports.py"), "--out-dir", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "final_table refused" in r.stdout
    assert not out.exists() or not any(out.iterdir())
