"""The seed paragraph of paper/sections/results.tex states only numbers from seeds_dev_interim.json,
and paper/tables/seeds_dev.tex is the generator's output for that file."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = " ".join((ROOT / "paper/sections/results.tex").read_text().split())
T = json.loads((ROOT / "reports/pilot/seeds_dev_interim.json").read_text())
ARMS = {a["label"]: a for a in T["arms"]}
M = "non_edit_macro_accuracy"


def m(x):
    return "$%+.3f$" % x if x >= 0 else "$%.3f$" % x


def test_table_is_generated_from_json():
    import importlib.util
    spec = importlib.util.spec_from_file_location("seed_table", ROOT / "scripts/seed_table.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert (ROOT / "paper/tables/seeds_dev.tex").read_text() == mod.latex(T)
    assert "\\input{tables/seeds_dev}" in TEX


def test_seed_paragraph_numbers():
    assert T["reference"] == "B" and T["split"] == "dev" and T["n_boot"] == 2000
    for label in ("B", "C", "D", "E"):
        assert ARMS[label]["seeds"] == [2027, 3407, 9173]
    d = ARMS["E"]["vs_ref"][M]
    assert "E, whose description parser was retrained per seed, is %s (%s to %s) above B" % (m(d["point"]), m(d["lo"]), m(d["hi"])) in TEX
    gaps = ARMS["E"]["seed_matched_macro_diff"].values()
    assert ARMS["E"]["reading"] == "positive_clear" and d["lo"] > 0 and all(g > 0 for g in gaps)
    assert "E: %s to %s" % (m(min(gaps)), m(max(gaps))) in TEX
    for label in ("C", "D"):
        d = ARMS[label]["vs_ref"][M]
        assert "%s is %s (%s to %s)" % (label, m(d["point"]), m(d["lo"]), m(d["hi"])) in TEX
        assert ARMS[label]["reading"] == "negative" and d["hi"] < 0
        gaps = ARMS[label]["seed_matched_macro_diff"].values()
        assert all(g < 0 for g in gaps)
        assert "%s: %s to %s" % (label, m(min(gaps)), m(max(gaps))) in TEX
