"""paper/sections/analysis.tex must state only numbers the code and generated reports fix."""
import json, re
from pathlib import Path

from conftest import require_data
from trace_vr import analysis, arms, dataset

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper" / "sections" / "analysis.tex").read_text()
FLAT = " ".join(TEX.split())


def test_wired_into_main():
    assert "\\input{sections/analysis}" in (ROOT / "paper" / "main.tex").read_text()


def test_size_buckets_match_facets():
    f = analysis.FACETS["size"]
    mk = lambda n: {"metadata": {"node_count": n}}
    assert (f(mk(8)), f(mk(9)), f(mk(12)), f(mk(13))) == ("nodes<=8", "nodes9-12", "nodes9-12", "nodes>=13")
    assert r"($\le 8$, 9--12, $\ge 13$ nodes)" in FLAT
    for facet in ("truth", "image_variant", "graph_family", "layout", "style"):
        assert facet in analysis.FACETS


def test_density_window_matches_report():
    w = json.loads((ROOT / "reports" / "density" / "pilot_dev.json").read_text())["window"]
    assert (w["lo_q"], w["hi_q"], w["ref_roots"]) == (0.05, 0.95, 512)
    assert "central 5--95\\% (nearest rank) of the 512 training roots" in FLAT
    assert "$[%.3f, %.3f]$" % (w["lo"], w["hi"]) in FLAT


def test_solver_sensitivity_numbers():
    d = json.loads((ROOT / "reports" / "parse_solve" / "sensitivity_dev.json").read_text())
    assert d["eval_root_graphs"] == 200 and "200 pilot development roots" in FLAT
    c = {k: v["non_edit_macro"] for k, v in d["conditions"].items()}
    assert c["drop_0"] == c["flip_0"] == 1.0
    assert "from $1.0$ to $%s$ and $%s$" % (c["drop_1"], c["drop_2"]) in FLAT
    assert "lowers it to $%s$ and $%s$" % (c["flip_1"], c["flip_2"]) in FLAT


def test_invariance_counts_and_final_eval():
    require_data("pilot", "pilot_test", "pilot_diag")
    _, _, by, _ = dataset.load_dataset(ROOT / "manifests" / "pilot.json")
    main, inv = arms.eval_samples(by["dev"], 200), arms.invariance_samples(by["dev"], 200)
    tab = analysis.invariance_table(main, {s["sample_id"]: True for s in main}, inv, {s["sample_id"]: True for s in inv})
    per_cell = {c["pairs"] for t in tab.values() for c in t.values()}
    assert "adds %d copies (%d per pair kind and task) to 1{,}%03d main samples" % (
        len(inv), per_cell.pop(), len(main) - 1000) in FLAT and not per_cell
    sh = (ROOT / "scripts" / "colab" / "final_eval.sh").read_text()
    total = 0
    for item in re.search(r'SPLITS="([^"]+)"', sh).group(1).split():
        name, split, roots = item.split(":")
        _, _, b, _ = dataset.load_dataset(ROOT / "manifests" / (name + ".json"))
        total += len(arms.invariance_samples(b[split], int(roots)))  # sample metadata only, no scoring
    assert "%d copies over the six locked test splits" % total in FLAT and len(sh.split('SPLITS="')[1].split('"')[0].split()) == 6
