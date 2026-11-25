"""rehearse_final_reports.py: synthetic collected outputs in the collect_final.py layout (no locked-test output)."""
import importlib.util, json, sys
from pathlib import Path
from conftest import require_data

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("rehearse_final_reports", ROOT / "scripts/rehearse_final_reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_answers_are_fixed_and_mixed():
    m = load()
    got = [m.answer("s%d" % i, "B", 3407) for i in range(60)]
    assert got == [m.answer("s%d" % i, "B", 3407) for i in range(60)]
    assert set(got) == set(m.ANSWERS)
    assert got != [m.answer("s%d" % i, "D", 3407) for i in range(60)]


def test_tables_list_matches_final_reports():
    m = load()
    spec = importlib.util.spec_from_file_location("final_reports", ROOT / "scripts/final_reports.py")
    fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
    assert sorted(m.FINAL_TABLES) == sorted(fr.ORDER)


def test_write_run_layout(tmp_path, monkeypatch):
    require_data("pilot_test")
    monkeypatch.setenv("TRACE_UNLOCK_TEST", "final-evaluation")
    m = load()
    repo = tmp_path / "repo"
    (repo / "manifests").mkdir(parents=True)
    (repo / "manifests" / "pilot_test.json").write_text((ROOT / "manifests" / "pilot_test.json").read_text())
    (repo / "data").symlink_to(ROOT / "data")
    (repo / "src").symlink_to(ROOT / "src")
    d = repo / "runs" / "x"; d.mkdir(parents=True)
    (d / "flags_test_id.json").write_text("REAL")          # a collected file must be replaced, never read
    entry = {"job": "jobF99", "arm": "E", "seed": 1, "run": "x", "dev_dir": "runs/x"}
    m.write_run(repo, entry, [("pilot_test", "test_id", 5)], ("main", "invariance"))
    s = json.loads((d / "final_summary.json").read_text())
    assert s["synthetic"] and s["evaluations"] == ["test_id", "test_id_inv"]
    assert set(s["metrics"]["test_id"]) >= {"metrics", "metrics_lenient_parse", "extraction_strict", "content_digest"}
    for suffix in ("test_id", "test_id_inv"):
        f = json.loads((d / ("flags_%s.json" % suffix)).read_text())
        assert f["split"] == "test_id" and f["eval_root_graphs"] == 5
