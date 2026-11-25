"""paper/sections/negative.tex may state only numbers present in the evidence files."""
import json, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = " ".join((ROOT / "paper" / "sections" / "negative.tex").read_text().split())
MAIN = (ROOT / "paper" / "main.tex").read_text()


def load(p):
    return json.loads((ROOT / p).read_text())


def r(x, nd):
    # half-up on the shortest repr (0.6475 -> 0.648), as the text was written
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 10))).quantize(q, rounding=ROUND_HALF_UP))


def test_wired_into_main():
    assert "\\input{sections/negative}" in MAIN
    assert "\\section{What did not work}" not in MAIN  # heading lives in the section file


def test_pilot3_d_vs_b():
    b = load("runs/pilot3_B/bootstrap_D_vs_B_dev.json")
    assert b["root_graphs"] == 200 and "200 development roots" in TEX
    m = "non_edit_macro_accuracy"
    assert f"${r(b['a'][m]['point'], 3)}$ (${r(b['a'][m]['lo'], 3)}$--${r(b['a'][m]['hi'], 3)}$)" in TEX
    assert f"${r(b['b'][m]['point'], 4)}$" in TEX
    assert f"(${r(b['b'][m]['lo'], 3)}$--${r(b['b'][m]['hi'], 3)}$)" in TEX
    d = b["a_minus_b"][m]
    assert f"$-{r(-d['point'], 3)}$ ($-{r(-d['lo'], 3)}$ to $-{r(-d['hi'], 3)}$)" in TEX
    assert d["share_above_zero"] == 0.0 and "no bootstrap replicate was above zero" in TEX
    re_, ce = b["a_minus_b"]["reachability"], b["a_minus_b"]["critical_edge"]
    assert f"${r(b['a']['reachability']['point'], 3)}$ vs.\\ ${r(b['b']['reachability']['point'], 3)}$" in TEX
    assert f"$-{r(-re_['point'], 3)}$, $-{r(-re_['lo'], 3)}$ to $+{r(re_['hi'], 3)}$" in TEX
    assert f"${r(b['a']['critical_edge']['point'], 4)}$ vs.\\ ${r(b['b']['critical_edge']['point'], 3)}$" in TEX
    assert f"$-{r(-ce['point'], 3)}$, $-{r(-ce['lo'], 3)}$ to $-{r(-ce['hi'], 3)}$" in TEX


def test_answer_bias_and_twins():
    d, bb = load("runs/pilot3_D/breakdown_dev.json"), load("runs/pilot3_B/breakdown_dev.json")
    ce_d, ce_b = d["answer_bias"]["critical_edge"], bb["answer_bias"]["critical_edge"]
    assert f"``true'' on ${round(100 * ce_d['said_true_rate'])}\\%$" in TEX
    assert f"(B: ${round(100 * ce_b['said_true_rate'])}\\%$, truth: ${round(100 * ce_d['truth_true_rate'])}\\%$)" in TEX
    pd, pb = d["critical_pairs"]["critical_edge"], bb["critical_pairs"]["critical_edge"]
    assert f"only {pd['both_right']} of {pd['pairs']} pairs (B: {pb['both_right']})" in TEX


def test_oracle_text_condition():
    o = load("reports/pilot/oracle/bootstrap_D_edges_vs_D_image_dev100.json")
    assert o["root_graphs"] == 100 and "100 development roots" in TEX
    assert o["inputs"][0]["run"] == "pilot3_D_edges" and o["inputs"][1]["run"] == "pilot3_D"
    m = "non_edit_macro_accuracy"
    assert f"${r(o['a'][m]['point'], 4)}$ with text and ${r(o['b'][m]['point'], 4)}$ with the image" in TEX
    d = o["a_minus_b"][m]
    assert f"$+{r(d['point'], 3)}$, $-{r(-d['lo'], 4)}$ to $+{r(d['hi'], 3)}$" in TEX
    assert f"${r(o['a']['critical_edge']['point'], 2)}$ vs.\\ ${r(o['b']['critical_edge']['point'], 3)}$" in TEX


def test_frozen_base():
    a = load("runs/pilot_A/summary.json")["eval_dev_first_30_roots"]
    assert f"({a['n']} questions) with a {a['max_new_tokens']}-token limit, {a['truncated']} responses" in TEX
    assert f"${a['non_edit_macro_accuracy']:.2f}$" in TEX and f"${a['non_edit_macro_lenient_accuracy']:.2f}$" in TEX


def test_layout_budget_numbers():
    doc = (ROOT / "docs" / "layout_budget.md").read_text()
    rows = {}
    for fam, n, budget, acc, ea, er in re.findall(
        r"\| (\w+) \| (\d+) \| (\d+) \| \d+/\d+ \(([\d.]+)\) \| ([\d.]+) / ([\d.]+) \|", doc):
        if n == "18":
            rows[(fam, budget)] = (acc, ea, er)
    sr, ld = rows[("sparse_random", "180")], rows[("layered_dag", "180")]
    assert f"from ${sr[0]}$ to ${rows[('sparse_random', '720')][0]}$" in TEX
    assert f"from ${ld[0]}$ to ${rows[('layered_dag', '720')][0]}$" in TEX
    assert f"({sr[1]} vs.\\ {sr[2]} edges and {ld[1]} vs.\\ {ld[2]} at budget 180)" in TEX


def test_never_run_list_matches_log():
    log = (ROOT / "docs" / "negative_results.md").read_text()
    never = log.split("## Discussed but never run")[1]
    for key in ("Recurrent visual working memory", "Qwen3-VL-2B"):
        assert key in never
    # C and E were run (job12/job13); neither may be listed as never run
    never_bullets = [l for l in never.split("## ")[0].splitlines() if l.startswith("- ") and "were run" not in l]
    assert not any("arm C" in l or "arm E" in l for l in never_bullets)
    tex_never = TEX.split("\\paragraph{Discussed but never run.}")[1]
    assert "arm C" not in tex_never and "arm E" not in tex_never


def _arm(label):
    d = load("reports/pilot/results_dev.json")
    assert d["reference"] == "B_kaggle" and d["root_graphs"] == 200
    return next(a for a in d["arms"] if a["label"] == label)


def test_d2_revision():
    a = _arm("D2_kaggle")
    m = a["acc"]["non_edit_macro_accuracy"]
    assert f"${r(m['point'], 4)}$ (${r(m['lo'], 3)}$--${r(m['hi'], 3)}$)" in TEX
    v = a["vs_ref"]["non_edit_macro_accuracy"]
    assert f"$-{r(-v['point'], 3)}$ ($-{r(-v['lo'], 3)}$ to $-{r(-v['hi'], 3)}$) below that B" in TEX
    assert f"critical-edge accuracy was ${r(a['acc']['critical_edge']['point'], 3)}$" in TEX
    bias = load("runs/pilot3_D2/breakdown_dev.json")["answer_bias"]["critical_edge"]
    assert f"``true'' on only ${round(100 * bias['said_true_rate'])}\\%$" in TEX


def test_structural_c_and_perception_split():
    a = _arm("C_kaggle")
    m, v = a["acc"]["non_edit_macro_accuracy"], a["vs_ref"]["non_edit_macro_accuracy"]
    assert f"${r(m['point'], 4)}$ (${r(m['lo'], 4)}$--${r(m['hi'], 3)}$)" in TEX
    assert f"$-{r(-v['point'], 3)}$ ($-{r(-v['lo'], 3)}$ to $-{r(-v['hi'], 3)}$) below B" in TEX
    assert f"accuracy ${r(a['acc']['critical_edge']['point'], 3)}$, below chance" in TEX
    ps = load("reports/pilot/perception_split_dev.json")
    assert ps["reference"] == "B_kaggle"
    assert f"the {ps['subsets']['readable']['root_graphs']} development roots" in TEX
    d = ps["arms"]["D_colab"]["readable"]["vs_ref"]
    assert f"(D$-$B $-{r(-d['point'], 3)}$, $-{r(-d['lo'], 3)}$ to $-{r(-d['hi'], 3)}$)" in TEX
    for arm in ("D_colab", "D2_kaggle", "C_kaggle"):
        rd = ps["arms"][arm]["readable"]["vs_ref"]
        assert rd["hi"] < 0  # the "reasoning, not reading" sentence needs every repair/structural arm below B


def test_three_seed_per_task_paragraph():
    T = load("reports/pilot/seeds_dev_interim.json")
    arms = {a["label"]: a for a in T["arms"]}
    assert T["split"] == "dev" and T["root_graphs"] == 200 and T["reference"] == "B"
    assert "the same 200 development roots (Table~\\ref{tab:seeds-dev})" in TEX
    for label in ("B", "C", "D"):
        assert arms[label]["seeds"] == [2027, 3407, 9173]

    def f(x):
        return "$%+.3f$" % x if x >= 0 else "$%.3f$" % x

    def diff(label, task):
        v = arms[label]["vs_ref"][task]
        return v, "%s, %s to %s" % (f(v["point"]), f(v["lo"]), f(v["hi"]))

    pd, pb = arms["D"]["pooled"], arms["B"]["pooled"]
    v, txt = diff("D", "reachability")
    assert "reachability is $%.3f$ for D and $%.3f$ for B (D$-$B %s)" % (
        pd["reachability"]["point"], pb["reachability"]["point"], txt) in TEX
    assert v["lo"] < 0 < v["hi"]  # the text does not call this a gap
    v, txt = diff("D", "critical_edge")
    assert "critical-edge accuracy is $%.3f$ for D and $%.3f$ for B (D$-$B %s)" % (
        pd["critical_edge"]["point"], pb["critical_edge"]["point"], txt) in TEX
    assert v["hi"] < 0
    vr, tr = diff("C", "reachability")
    vc, tc = diff("C", "critical_edge")
    assert "(reachability %s; critical edge %s)" % (tr, tc) in TEX and vr["hi"] < 0 and vc["hi"] < 0
    ce = [s["critical_edge"] for s in arms["D"]["per_seed"].values()]
    assert "stays within $%s$--$%s$ across seeds" % (r(min(ce), 4), r(max(ce), 4)) in TEX
    # Once the locked test is opened (final_primary.json exists) the paragraph points to it instead of
    # saying it has not been evaluated; it still quotes no test number.
    if (ROOT / "reports/final/final_primary.json").exists():
        assert "have not been evaluated" not in TEX
        assert "the locked test is reported in Section~\\ref{sec:test-results}" in TEX
    else:
        assert "locked test splits have not been evaluated" in TEX


def test_three_seed_twin_pair_sentence():
    P = load("reports/pilot/pairs_seeds_dev.json")
    assert P["split"] == "dev" and P["root_graphs"] == 200
    arms = {a["label"]: a for a in P["arms"]}
    for label in ("B", "D"):
        assert arms[label]["seeds"] == [2027, 3407, 9173]
    d, b = arms["D"], arms["B"]
    dc, bc = d["summed"]["pairs"]["critical_edge"], b["summed"]["pairs"]["critical_edge"]
    per = [s["pairs"]["critical_edge"]["both_right"] for s in d["per_seed"].values()]
    assert "on %d of %d pairs (%d--%d of 200 per seed; B: %d of %d)" % (
        dc["both_right"], dc["pairs"], min(per), max(per), bc["both_right"], bc["pairs"]) in TEX
    bias = d["summed"]["answer_bias"]["critical_edge"]
    rates = [s["answer_bias"]["critical_edge"]["said_true_rate"] for s in d["per_seed"].values()]
    assert bias["no_answer"] == 0 and bias["truth_true_rate"] == 0.5
    assert "to $%s$ of critical-edge questions ($%s$--$%s$ per seed; \\emph{true} is correct for half)" % (
        r(bias["said_true_rate"], 3), r(min(rates), 3), r(max(rates), 3)) in TEX
    dr, br = d["summed"]["pairs"]["reachability"], b["summed"]["pairs"]["reachability"]
    assert "On reachability pairs D has both right on %d of %d and B on %d of %d" % (
        dr["both_right"], dr["pairs"], br["both_right"], br["pairs"]) in TEX
    assert "These counts are descriptive and carry no interval" in TEX
    # the pair counts must agree with the seed table's pooled accuracy: (2 both + one) / samples
    S = {a["label"]: a for a in load("reports/pilot/seeds_dev_interim.json")["arms"]}
    for label, cell in (("D", dc), ("B", bc)):
        acc = (2 * cell["both_right"] + cell["one_right"]) / (2 * cell["pairs"])
        assert abs(acc - S[label]["pooled"]["critical_edge"]["point"]) < 1e-9


def test_oracle_text_is_not_a_clean_bound_for_b():
    md = (ROOT / "reports/pilot/go_no_go_rev1.md").read_text()
    row = [l for l in md.splitlines() if l.startswith("| Non-edit macro |")][0]
    edges, image = [c.strip() for c in row.split("|")[2:4]]
    assert "(non-edit macro $%s$ against $%s$" % (edges, image) in TEX and float(edges) < float(image)
    assert "did not resolve D's critical-edge deficit" in TEX and "not reading the drawing" not in TEX
