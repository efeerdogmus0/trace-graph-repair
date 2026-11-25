"""paper/sections/external.tex states only numbers from the VGCure pooling records, and
paper/tables/vgcure_dev.tex is the generator's output for them."""
import importlib.util, json, re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "paper/sections/external.tex").read_text()
TEX = " ".join(" ".join(l for l in RAW.splitlines() if not l.lstrip().startswith("%")).split())
MAIN = (ROOT / "paper/main.tex").read_text()
VS_B = ROOT / "reports/external/vgcure_dev_seeds_vs_B.json"
VS_C = ROOT / "reports/external/vgcure_dev_seeds_vs_C.json"
B, C = json.loads(VS_B.read_text()), json.loads(VS_C.read_text())
MACRO, EDGE, REACH = "external_macro", "edge_exists_accuracy", "reachability_balanced_accuracy"


def _mod():
    spec = importlib.util.spec_from_file_location("vgcure_table", ROOT / "scripts/vgcure_table.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def m(x):
    return "$%s%.3f$" % ("+" if x >= 0 else "-", abs(x))


def span(d, sign=1):
    lo, hi = sorted((sign * d["lo"], sign * d["hi"]))
    return m(sign * d["point"]), m(lo), m(hi)


def test_wired_after_locked_test_section():
    assert MAIN.count("\\input{sections/external}") == 1
    assert MAIN.index("\\input{sections/test_results}") < MAIN.index("\\input{sections/external}") < MAIN.index("\\input{sections/negative}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", "".join(p.read_text() for p in (ROOT / "paper/sections").glob("*.tex"))))
    assert set(re.findall(r"\\ref\{([^}]+)\}", RAW)) <= labels


def test_table_is_generated_from_both_records():
    mod = _mod()
    text = mod.latex(mod.load([VS_B, VS_C]))
    assert (ROOT / "paper/tables/vgcure_dev.tex").read_text() == text
    assert "\\input{tables/vgcure_dev}" in TEX
    for line in text.splitlines()[2:]:
        assert "_" not in line, line   # a bare underscore would not compile
    spec = re.search(r"\\begin\{tabular\}\{(\w+)\}", text).group(1)
    rows = [l for l in text.splitlines() if l.endswith("\\\\")]
    assert len(rows) == 4 and all(l.count(" & ") == len(spec) - 1 for l in rows)   # else LaTeX stops
    assert text.count("ref") == 2 and "B QA-SFT & 3 &" in text and "D repair & 3 &" in text


def test_generator_refuses_mixed_evaluations(tmp_path):
    mod = _mod()
    other = dict(C, subset_sha256="0" * 64)
    p = tmp_path / "c.json"; p.write_text(json.dumps(other))
    with pytest.raises(SystemExit):
        mod.load([VS_B, p])
    moved = json.loads(VS_C.read_text()); moved["arms"]["D"]["pooled"][MACRO] += 0.001
    p.write_text(json.dumps(moved))
    with pytest.raises(SystemExit):
        mod.load([VS_B, p])
    with pytest.raises(SystemExit):
        mod.load([VS_B, VS_B])


def test_subset_sizes():
    pre = json.loads((ROOT / "reports/external/vgcure_subset.json").read_text())
    dev = json.loads((ROOT / "reports/external/vgcure_dev_subset.json").read_text())
    assert (pre["per_source"], pre["n_samples"]) == (50, 1500)
    assert (dev["per_source"], dev["skip"], dev["n_samples"]) == (40, 50, 1200)
    assert not set(pre["sample_ids"]) & set(dev["sample_ids"])
    assert B["record"] == C["record"] == "dev" and B["graphs"] == 400 and B["n"] == 1200 and B["resamples"] == 10000
    assert B["subset_sha256"] == C["subset_sha256"] == dev["subset_sha256"]
    for s in ("$500$ graphs, $1{,}500$ items: the first $50$ usable graphs", "$400$ graphs, $1{,}200$ items: the next $40$ usable graphs",
              "$10{,}000$ resamples", "$512\\times512$"):
        assert s in TEX, s


def test_exploratory_paragraph_numbers():
    dc, db = C["arms"]["D"]["diff_pooled"], B["arms"]["D"]["diff_pooled"]
    assert "D is %s (%s to %s) above C and %s (%s to %s) above B in external macro" % (span(dc[MACRO]) + span(db[MACRO])) in TEX
    assert "edge-existence items (%s, %s to %s)" % span(db[EDGE]) in TEX and db[EDGE]["lo"] > 0
    assert "the two arms are level (%s, %s to %s)" % span(db[REACH]) in TEX and db[REACH]["lo"] < 0 < db[REACH]["hi"]
    seeds = [s[MACRO] for s in B["arms"]["D"]["diff_per_seed"]]
    assert len(seeds) == 3 and all(s["point"] > 0 for s in seeds) and sum(s["lo"] > 0 for s in seeds) == 1
    assert "positive three times, but its interval excludes zero at only one seed" in TEX
    bc = C["arms"]["B"]["diff_pooled"][REACH]
    assert "on reachability (%s against B, %s to %s)" % span(bc, -1) in TEX and bc["lo"] > 0
    assert C["arms"]["D"]["diff_pooled"][REACH]["lo"] > 0   # "far below both"


def test_comparison_with_dev_split_is_what_the_seed_table_says():
    t = json.loads((ROOT / "reports/pilot/seeds_dev_interim.json").read_text())
    d = {a["label"]: a for a in t["arms"]}["D"]["vs_ref"]
    assert t["reference"] == "B" and d["non_edit_macro_accuracy"]["hi"] < 0          # "clearly below B"
    assert d["critical_edge"]["hi"] < 0 and d["reachability"]["lo"] < 0 < d["reachability"]["hi"]
    assert abs(d["reachability"]["point"]) < 0.03
    assert "D's deficit is in the critical-edge task while its reachability is close to B's" in TEX


def test_scope_statements():
    for s in ("We do not report official VGCure numbers", "reported as exploratory and support no claim on their own",
              "not the model of the VGCure authors", "No VGCure file enters any training pool",
              "No arm was changed or retrained after these numbers were seen",
              "No decision rule was pre-registered for the external subset",
              "not as a transfer result"):
        assert s in TEX, s
    assert "Pending" not in TEX and "IfFileExists" not in RAW


FB = ROOT / "reports/external/vgcure_final_seeds_vs_B.json"
FC = ROOT / "reports/external/vgcure_final_seeds_vs_C.json"
FINAL_B, FINAL_C = json.loads(FB.read_text()), json.loads(FC.read_text())


def test_final_table_is_generated_from_preregistered_records():
    pre = json.loads((ROOT / "reports/external/vgcure_subset.json").read_text())
    for r in (FINAL_B, FINAL_C):
        assert r["record"] == "preregistered" and r["subset_sha256"] == pre["subset_sha256"]
        assert (r["graphs"], r["n"], r["resamples"]) == (500, 1500, 10000)
        assert sorted(r["arms"]) == ["B", "C", "D", "E"] and all(len(a["runs"]) == 3 for a in r["arms"].values())
    assert FINAL_B["reference"] == "B" and FINAL_C["reference"] == "C"
    mod = _mod()
    assert (ROOT / "paper/tables/vgcure_final.tex").read_text() == mod.latex(mod.load([FB, FC]))
    assert "\\input{tables/vgcure_final}" in TEX


def test_final_paragraph_numbers():
    d = FINAL_B["arms"]["D"]
    assert "D is %s (%s to %s) above B in external macro" % span(d["diff_pooled"][MACRO]) in TEX
    seeds = [s[MACRO] for s in d["diff_per_seed"]]
    assert all(s["lo"] > 0 for s in seeds) and "above zero at every seed (%s, %s, %s, each interval excluding zero)" % tuple(
        m(s["point"]) for s in seeds) in TEX
    assert "edge existence %s (%s to %s)" % span(d["diff_pooled"][EDGE]) in TEX
    assert "reachability %s (%s to %s)" % span(d["diff_pooled"][REACH]) in TEX
    c = FINAL_B["arms"]["C"]["diff_pooled"]
    assert "C is %s (%s to %s) below B" % span(c[MACRO]) in TEX and c[MACRO]["hi"] < 0
    assert "again on reachability (%s, %s to %s)" % span(c[REACH]) in TEX
    assert "equals B's (%s, %s to %s)" % span(c[EDGE]) in TEX and c[EDGE]["lo"] < 0 < c[EDGE]["hi"]
    e = FINAL_B["arms"]["E"]
    assert "E drops to $%.3f$, %s (%s to %s) below B" % ((e["pooled"][MACRO],) + span(e["diff_pooled"][MACRO])) in TEX
    dev = B["arms"]["D"]["diff_pooled"][MACRO]["point"]
    assert d["diff_pooled"][MACRO]["point"] > dev and "by more than on the exploratory subset" in TEX


def test_final_reading_matches_locked_test():
    for x in ("0.784", "0.789"):
        assert "$%s$" % x in TEX
    assert "D is level with B on reachability ($0.784$ against $0.789$)" in TEX
    md = (ROOT / "reports/final/final_primary.md").read_text()
    assert "| B | 0.762 [0.750, 0.773] | 0.789 " in md and "| D | 0.657 [0.646, 0.667] | 0.784 " in md


def test_e_transcription_failures_match_metrics():
    runs = [ROOT / "runs/kaggle" / j / "runs" / r / "metrics_vgcure.json"
            for j, r in (("jobV2", "pilot3_E"), ("jobV3", "pilot3_E_s2027"), ("jobV4", "pilot3_E_s9173"))]
    if not all(p.exists() for p in runs):
        pytest.skip("Kaggle VGCure metrics not downloaded (gitignored)")
    failed = []
    for p in runs:
        reasons = json.loads(p.read_text())["reasons"]
        bad = sum(v for k, v in reasons.items() if k.startswith("extraction_format:"))
        assert bad % 3 == 0 and sum(reasons.values()) == 1500   # one transcription per image, three items each
        failed.append(bad // 3)
    assert "no usable graph for $%d$, $%d$ and $%d$ of the $500$ images" % tuple(failed) in TEX
