"""Rule repro-v1 (scripts/repro_compare.py): GPU rerun from a clean clone vs the recorded run."""
import json, shutil, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repro_compare as rc  # noqa: E402

REC = ROOT / "runs/pilot3_D_kaggle"


def _copy(tmp_path, name="rerun"):
    d = tmp_path / name
    d.mkdir()
    for f in ("summary.json", "flags_dev.json"):
        shutil.copy(REC / f, d / f)
    return d


def _edit(d, fn_summary=None, fn_flags=None):
    for name, fn in (("summary.json", fn_summary), ("flags_dev.json", fn_flags)):
        if fn:
            p = d / name
            obj = json.loads(p.read_text())
            fn(obj)
            p.write_text(json.dumps(obj))


def test_same_run_is_identical(tmp_path):
    r = rc.compare(REC, _copy(tmp_path))
    assert r["verdict"] == "identical" and r["bit_agreement"] == 1.0
    assert r["per_task"]["reachability"]["recorded"] == 0.8125


def _rerun_with(tmp_path, deltas, macro_delta):
    d = _copy(tmp_path)

    def s(obj):
        m = obj["eval_dev"]["metrics"]
        for t, dv in deltas.items():
            m["per_task"][t]["accuracy"] += dv
        m["non_edit_macro_accuracy"] += macro_delta

    def f(obj):
        obj["predictions_sha256"] = "0" * 64
        c = list(obj["correct"]); c[0] = "1" if c[0] == "0" else "0"; obj["correct"] = "".join(c)

    _edit(d, s, f)
    return d


def test_within_tolerance_is_reproduced(tmp_path):
    r = rc.compare(REC, _rerun_with(tmp_path, {"critical_edge": 0.03, "minimal_graph_repair": -0.0275}, 0.015))
    assert r["verdict"] == "reproduced"
    assert r["bit_agreement"] == pytest.approx(1 - 1 / 1200)


def test_task_outside_tolerance_is_not_reproduced(tmp_path):
    r = rc.compare(REC, _rerun_with(tmp_path, {"reachability": -0.035}, -0.0175))
    assert r["verdict"] == "not_reproduced"


def test_macro_outside_tolerance_is_not_reproduced(tmp_path):
    r = rc.compare(REC, _rerun_with(tmp_path, {}, 0.025))
    assert r["verdict"] == "not_reproduced"


@pytest.mark.parametrize("field", ["content_digest", "order_sha", "n"])
def test_different_sample_set_is_refused(tmp_path, field):
    d = _copy(tmp_path)
    _edit(d, fn_flags=lambda o: o.__setitem__(field, "x"))
    with pytest.raises(SystemExit, match="refused"):
        rc.compare(REC, d)


def test_other_seed_is_refused():
    with pytest.raises(SystemExit, match="seed"):
        rc.compare(REC, ROOT / "runs/pilot3_D_s2027")


def test_tolerances_cover_twice_the_recorded_seed_spread():
    seeds = json.loads((ROOT / "reports/pilot/seeds_dev_interim.json").read_text())
    d = next(a for a in seeds["arms"] if a["label"] == "D")["per_seed"]
    for t in rc.TASKS:
        vals = [v[t] for v in d.values()]
        assert rc.TASK_TOL >= 2 * (max(vals) - min(vals)) - 1e-9
    macro = [v["non_edit_macro_accuracy"] for v in d.values()]
    assert rc.MACRO_TOL >= 2 * (max(macro) - min(macro))


def test_budget_rule_groups_rerun_lines():
    import budget_report
    assert budget_report.group_of("pilot3_D_jobR1") == "repro"
    assert budget_report.group_of("pilot3_D_jobF04") == "final"


def test_cli_writes_reports(tmp_path):
    out, md = tmp_path / "r.json", tmp_path / "r.md"
    subprocess.run([sys.executable, str(ROOT / "scripts/repro_compare.py"), str(REC), str(_copy(tmp_path)),
                    "--out", str(out), "--out-md", str(md)], check=True, capture_output=True)
    assert json.loads(out.read_text())["rule"] == "repro-v1"
    assert "Verdict: **identical**" in md.read_text()
