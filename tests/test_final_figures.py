"""scripts/final_figures.py: the locked-test figure is drawn from the accepted final table only.

The table used here is made up in the test (round values chosen by hand), so no locked-test output is involved.
Plotted values must be copies of the table, the script must refuse an incomplete or foreign table, and with the
real repository (final table not written yet) it must refuse and write nothing."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MACRO = "non_edit_macro_accuracy"


def _load():
    spec = importlib.util.spec_from_file_location("final_figures", ROOT / "scripts" / "final_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fig = _load()
JOBS = ["jobF%02d" % i for i in range(1, 13)]
ARMS = ["B", "C", "D", "E"]


def _table():
    tables = {}
    for k, (manifest, split, roots) in enumerate(fig.SPLITS):
        arms = []
        for j, label in enumerate(ARMS):
            point = 0.9 - 0.1 * j - 0.05 * k
            arms.append({"label": label, "seeds": list(fig.SEEDS),
                         "per_seed": {str(s): {MACRO: point + 0.01 * (i - 1)} for i, s in enumerate(fig.SEEDS)},
                         "pooled": {MACRO: {"point": point, "lo": point - 0.03, "hi": point + 0.02}}})
        tables[split] = {"split": split, "manifest": manifest, "root_graphs": roots, "arms": arms}
    return {"reference": "B", "n_boot": 2000, "jobs": list(JOBS), "splits": [s for _, s, _ in fig.SPLITS], "tables": tables}


def _write(tmp_path, t):
    p = tmp_path / "final_table.json"
    p.write_text(json.dumps(t))
    return p


def test_points_are_copies_of_the_table(tmp_path):
    t = _table()
    data = fig.build(_write(tmp_path, t), JOBS)
    assert data["splits"] == [s for _, s, _ in fig.SPLITS] and data["arms"] == ARMS and data["runs"] == 12
    assert len(data["ood_test"]) == len(fig.SPLITS) * len(ARMS)
    for r in data["ood_test"]:
        a = {x["label"]: x for x in t["tables"][r["split"]]["arms"]}[r["arm"]]
        assert (r["point"], r["lo"], r["hi"]) == tuple(a["pooled"][MACRO][k] for k in ("point", "lo", "hi"))
        assert r["per_seed"] == {str(s): a["per_seed"][str(s)][MACRO] for s in fig.SEEDS}
        at_id = {x["label"]: x for x in t["tables"]["test_id"]["arms"]}[r["arm"]]["pooled"][MACRO]["point"]
        assert r["minus_test_id"] == r["point"] - at_id
    assert [r["minus_test_id"] for r in data["ood_test"] if r["split"] == "test_id"] == [0.0] * len(ARMS)
    assert "no interval" in data["note"]


def test_every_locked_split_has_a_name():
    assert set(fig.SPLIT_NAMES) == {s for _, s, _ in fig.SPLITS}
    assert fig.IN_DISTRIBUTION == fig.SPLITS[0][1]


@pytest.mark.parametrize("damage, message", [
    (lambda t: t["jobs"].pop(), "the table covers 11 runs"),
    (lambda t: t["splits"].pop(), "the table's splits"),
    (lambda t: t["tables"]["test_size_ood"].update(root_graphs=99), "is not 100 roots"),
    (lambda t: t["tables"]["test_style_ood"]["arms"].pop(), "arms differ"),
    (lambda t: t["tables"]["test_id"]["arms"][2]["seeds"].pop(), "has seeds"),
    (lambda t: t.update(reference="Z"), "reference"),
])
def test_incomplete_or_foreign_table_is_refused(tmp_path, damage, message):
    t = copy.deepcopy(_table())
    damage(t)
    with pytest.raises(SystemExit) as e:
        fig.build(_write(tmp_path, t), JOBS)
    assert "refused" in str(e.value) and message in str(e.value)


def test_missing_table_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        fig.build(tmp_path / "final_table.json", JOBS)
    assert "does not exist" in str(e.value)


def test_draws_reproducibly_and_check_detects_a_changed_table(tmp_path):
    pytest.importorskip("matplotlib")
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"jobs": [{"job": j} for j in JOBS]}))
    table = _write(tmp_path, _table())
    args = ["--table", str(table), "--queue", str(queue), "--out-dir", str(tmp_path / "fig"), "--paper-dir", str(tmp_path / "paper")]
    assert fig.main(args) == 0
    names = ["fig/ood_test.svg", "fig/ood_test.png", "paper/ood_test.pdf", "fig/figures_test.json", "fig/README.md"]
    first = {n: (tmp_path / n).read_bytes() for n in names}
    assert all(len(b) > 1000 for b in first.values())
    assert fig.main(args) == 0
    assert first == {n: (tmp_path / n).read_bytes() for n in names}
    stored = json.loads(first["fig/figures_test.json"])
    assert stored["rule"] == "final-figures-v1" and len(stored["files"]) == 3
    readme = first["fig/README.md"].decode()
    assert readme.count("\n| test_") == len(fig.SPLITS) * len(ARMS) and "no interval" in readme
    assert fig.main(args + ["--check"]) == 0
    t = _table()
    t["tables"]["test_id"]["arms"][0]["pooled"][MACRO]["point"] += 0.001
    table.write_text(json.dumps(t))
    assert fig.main(args + ["--check"]) == 1


def test_real_repository_refuses_until_the_final_table_exists(tmp_path):
    """The final table is written only after all twelve final kernels are collected. Until then: exit != 0, no file."""
    if (ROOT / "reports/final/final_table.json").exists():
        pytest.skip("final table written; the locked test has been reported")
    out = tmp_path / "fig"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/final_figures.py"), "--out-dir", str(out), "--paper-dir", str(tmp_path / "p")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "refused" in r.stderr
    assert not out.exists() and not (tmp_path / "p").exists()


def test_reads_no_flags_or_predictions():
    src = (ROOT / "scripts" / "final_figures.py").read_text()
    code = src.split('"""', 2)[2]
    for word in ("flags_", "final_summary", "predictions", "TRACE_UNLOCK_TEST"):
        assert word not in code, word
