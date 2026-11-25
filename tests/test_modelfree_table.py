"""paper/tables/modelfree.tex is the generator's output for the tracked validator reports, and the
benchmark section's sentences about it hold for every split."""
import importlib.util, json, re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "paper/sections/benchmark.tex").read_text()
TEX = " ".join(" ".join(l for l in RAW.splitlines() if not l.lstrip().startswith("%")).split())
MAIN_TASKS = ("reachability", "critical_edge", "minimal_graph_repair")


def _mod():
    spec = importlib.util.spec_from_file_location("modelfree_table", ROOT / "scripts/modelfree_table.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _probes():
    for build in _mod().BUILDS:
        report = json.loads((ROOT / "reports" / build / "validation.json").read_text())
        for split, tasks in report["text_only_shortcuts"].items():
            yield build, split, tasks


def test_table_is_generated():
    mod = _mod()
    assert (ROOT / "paper/tables/modelfree.tex").read_text() == mod.latex(mod.rows())
    assert RAW.count("\\input{tables/modelfree}") == 1 and "\\label{tab:modelfree}" in RAW


def test_one_row_per_split_with_copied_values():
    mod = _mod()
    table = {r["split"]: r for r in mod.rows()}
    seen = {split: tasks for _, split, tasks in _probes()}
    assert set(table) == set(seen) == set(mod.NAMES) and len(table) == 8
    for split, tasks in seen.items():
        for t in ("reachability", "critical_edge"):
            assert table[split][t]["constant"] == max(tasks[t]["always_true"], tasks[t]["always_false"])
            assert table[split][t]["text_only"] == tasks[t]["text_only_ceiling"]
        rep = tasks["minimal_graph_repair"]
        assert table[split]["minimal_graph_repair"]["constant"] == max(rep["always_unchanged"], rep["always_impossible"])
        assert table[split]["minimal_graph_repair"]["heuristic"] == max(rep[k] for k in mod.HEURISTICS)
        assert "%.4f" % table[split]["minimal_graph_repair"]["heuristic"] in mod.latex(mod.rows())


def test_paragraph_holds_on_every_split():
    stated = float(re.search(r"no text heuristic exceeds \$([0-9.]+)\$", TEX).group(1))
    assert "constant answer scores $0.500$" in TEX and "constant status scores $0.250$" in TEX
    assert "ceiling is $0.500$ on all three main tasks in every split" in TEX
    best = 0.0
    for _, split, tasks in _probes():
        for t in ("reachability", "critical_edge"):
            assert tasks[t]["always_true"] == tasks[t]["always_false"] == 0.5, split
        rep = tasks["minimal_graph_repair"]
        assert rep["always_unchanged"] == rep["always_impossible"] == 0.25, split
        for t in MAIN_TASKS:
            assert tasks[t]["text_only_ceiling"] == 0.5, (split, t)
        best = max([best] + [rep[k] for k in _mod().HEURISTICS])
    assert best <= stated < best + 0.01


def test_refuses_unclean_or_foreign_report(tmp_path):
    mod = _mod()
    for build in mod.BUILDS:
        (tmp_path / "reports" / build).mkdir(parents=True)
        (tmp_path / "manifests").mkdir(exist_ok=True)
        (tmp_path / "reports" / build / "validation.json").write_text((ROOT / "reports" / build / "validation.json").read_text())
        (tmp_path / "manifests" / (build + ".json")).write_text((ROOT / "manifests" / (build + ".json")).read_text())
    assert mod.rows(tmp_path) == mod.rows()
    path = tmp_path / "reports/pilot_test/validation.json"
    good = json.loads(path.read_text())
    for change in ({"error_count": 1}, {"ok": False}, {"manifest": "manifests/pilot.json"}):
        path.write_text(json.dumps({**good, **change}))
        with pytest.raises(SystemExit):
            mod.rows(tmp_path)
    dropped = json.loads(json.dumps(good)); dropped["text_only_shortcuts"].pop("test_id")
    path.write_text(json.dumps(dropped))
    with pytest.raises(SystemExit):
        mod.rows(tmp_path)
