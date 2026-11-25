"""reports/budget.{md,json} are the generator's output for the tracked budget log, and the totals are
plain sums of its lines (no estimate, lost runs counted as zero and listed)."""
import importlib.util, json, re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("budget_report", ROOT / "scripts/budget_report.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _raw():
    return [json.loads(l) for l in (ROOT / "runs/budget.jsonl").read_text().splitlines() if l.strip()]


def test_tracked_report_is_current():
    mod = _mod()
    rows, notes = mod.load(ROOT / "runs/budget.jsonl")
    summary = mod.summarise(rows)
    assert (ROOT / "reports/budget.md").read_text() == mod.markdown(summary, notes)
    assert json.loads((ROOT / "reports/budget.json").read_text()) == json.loads(json.dumps(summary))
    assert mod.main(["--check"]) == 0


def test_total_is_the_plain_sum_of_the_log():
    summary = json.loads((ROOT / "reports/budget.json").read_text())
    raw = [r for r in _raw() if "run_id" in r]
    with_time = [r for r in raw if r.get("gpu_seconds") is not None]
    assert summary["lines"] == len(raw)
    assert summary["measured"] == {"lines": len(with_time), "gpu_seconds": round(sum(r["gpu_seconds"] for r in with_time), 1)}
    assert summary["cash_cost"] == 0 and all(r["cash_cost"] == 0 for r in raw)
    # groups and platforms each partition the measured lines
    for parts in (summary["groups"], list(summary["platforms"].values())):
        assert sum(p["lines"] for p in parts) == len(with_time)
        assert sum(p["gpu_seconds"] for p in parts) == pytest.approx(summary["measured"]["gpu_seconds"], abs=0.11)
        assert sum(p["lines_total"] for p in parts) == len(raw)


def test_every_line_without_time_is_named():
    summary = json.loads((ROOT / "reports/budget.json").read_text())
    md = (ROOT / "reports/budget.md").read_text()
    missing = [r["run_id"] for r in _raw() if "run_id" in r and r.get("gpu_seconds") is None]
    listed = [u["run_id"] for u in summary["unmeasured_lines"]] + [p["run_id"] for p in summary["placeholder_lines"]]
    assert sorted(missing) == sorted(listed) and missing
    assert all("`%s`" % run_id in md for run_id in missing)
    assert all(str(u["status"]).startswith(("lost", "not_read_back")) for u in summary["unmeasured_lines"])
    assert "lower bound" in md and "counted as zero" in md


def test_state_file_total_matches():
    if not (ROOT / "PROJECT_STATE.md").exists():
        pytest.skip("working-state file is not part of the public release (scripts/release_export.py)")
    summary = json.loads((ROOT / "reports/budget.json").read_text())
    state = (ROOT / "PROJECT_STATE.md").read_text()
    assert int(re.search(r"^gpu_seconds_total: (\d+)", state, re.M).group(1)) == round(summary["measured"]["gpu_seconds"])
    assert re.search(r"^budget_spent_cash: 0$", state, re.M)


def test_kinds_and_refusals(tmp_path):
    mod = _mod()
    base = {"platform": "kaggle", "device": None, "cash_cost": 0}
    rows = [dict(base, run_id="pilot3_B_jobF01", gpu_seconds=10.0, status="complete", line=1),
            dict(base, run_id="pilot3_B_job11", gpu_seconds=5, status="failed_x", line=2),
            dict(base, run_id="pilot3_x_lost", gpu_seconds=None, status="lost_vm", line=3),
            dict(base, run_id="pilot3_job6", status="running", line=4)]
    assert [mod.kind_of(r) for r in rows] == ["measured", "measured_failed", "unmeasured", "placeholder"]
    assert [mod.group_of(r["run_id"]) for r in rows] == ["final", "pilot3", "pilot3", "pilot3"]
    s = mod.summarise(rows)
    assert s["measured"] == {"lines": 2, "gpu_seconds": 15.0} and s["measured_failed"] == {"lines": 1, "gpu_seconds": 5.0}
    assert [g["key"] for g in s["groups"]] == ["final", "pilot3"]
    with pytest.raises(SystemExit):
        mod.group_of("something_new")
    with pytest.raises(SystemExit):
        mod.summarise([dict(base, run_id="pilot3_B_job11", gpu_seconds=-1, status="complete", line=1)])
    with pytest.raises(SystemExit):
        mod.summarise([dict(run_id="pilot3_B_job11", platform="kaggle", gpu_seconds=1, status="complete", line=1)])
    bad = tmp_path / "b.jsonl"; bad.write_text('{"gpu_seconds": 5}\n')
    with pytest.raises(SystemExit):
        mod.load(bad)


def test_stale_file_is_detected(tmp_path):
    mod = _mod()
    out = tmp_path / "budget.md"; out.write_text((ROOT / "reports/budget.md").read_text() + "x")
    assert mod.main(["--check", "--out", str(out)]) == 1
