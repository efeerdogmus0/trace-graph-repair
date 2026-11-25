"""paper/tables/compute.tex is the generator's output for the tracked budget report, and the
reproducibility appendix quotes compute numbers only through its macros."""
import importlib.util, json, re, shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "paper/sections/reproducibility.tex").read_text()


def _mod():
    spec = importlib.util.spec_from_file_location("compute_table", ROOT / "scripts/compute_table.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_table_is_generated_and_wired():
    mod = _mod()
    assert (ROOT / "paper/tables/compute.tex").read_text() == mod.latex(mod.load())
    assert RAW.count("\\input{tables/compute}") == 1 and RAW.count("\\computeTable") == 1
    assert "\\label{tab:compute}" in RAW
    assert RAW.index("\\input{tables/compute}") < RAW.index("\\computeMeasuredHours")


def test_every_macro_used_in_prose_is_defined():
    defined = set(re.findall(r"\\newcommand\{\\(compute\w+)\}", (ROOT / "paper/tables/compute.tex").read_text()))
    used = set(re.findall(r"\\(compute[A-Za-z]+)", RAW))
    assert used and used <= defined, used - defined


def test_totals_match_report():
    report = json.loads((ROOT / "reports/budget.json").read_text())
    text = (ROOT / "paper/tables/compute.tex").read_text()
    assert "\\newcommand{\\computeMeasuredHours}{%.2f}" % (report["measured"]["gpu_seconds"] / 3600) in text
    groups = {g["key"]: g for g in report["groups"]}
    rows = re.findall(r"^(.+?) & (\d+) & ([0-9.]+) & (\d+) \\\\$", text, re.M)
    body = rows[:len(groups)]
    assert sum(int(r[1]) for r in body) == report["measured"]["lines"]
    assert abs(sum(float(r[2]) for r in body) - report["measured"]["gpu_seconds"] / 3600) < 0.01 * len(body)
    assert rows[-1][0] == "Total" and int(rows[-1][1]) + int(rows[-1][3]) == report["lines"]


def test_refuses_stale_report(tmp_path):
    mod = _mod()
    for rel in ("reports/budget.json", "runs/budget.jsonl"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    assert mod.load(tmp_path) == mod.load()
    with (tmp_path / "runs/budget.jsonl").open("a") as f:
        f.write(json.dumps({"run_id": "pilot3_X_new", "platform": "kaggle", "gpu_seconds": 10, "cash_cost": 0,
                            "status": "ok"}) + "\n")
    with pytest.raises(SystemExit):
        mod.load(tmp_path)
