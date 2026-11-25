"""paper/sections/why.tex and the framing-v2 texts state only numbers from tracked reports, and the paper shows
framing-v2 while the pre-registered finding-v1 / conclusion-v1 outputs stay as the record."""
import importlib.util, json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "paper/sections/why.tex").read_text()
TEX = " ".join(" ".join(l for l in RAW.splitlines() if not l.lstrip().startswith("%")).split())
FINAL = ROOT / "reports/final"
BRK = json.loads((ROOT / "reports/external/vgcure_final_breakdown.json").read_text())
EXT = json.loads((ROOT / "reports/external/vgcure_final_seeds_vs_B.json").read_text())
PRIMARY = json.loads((FINAL / "final_primary.json").read_text())


def _framing():
    spec = importlib.util.spec_from_file_location("paper_framing", ROOT / "scripts/paper_framing.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def n3(x):
    return "$%.3f$" % x


def test_generated_texts_are_current_and_wired():
    mod = _framing()
    for name, text in mod.render(mod.facts()).items():
        assert (FINAL / name).read_text() == text, name
    main = (ROOT / "paper/main.tex").read_text()
    assert "\\IfFileExists{../reports/final/framing_abstract.tex}{\\input{../reports/final/framing_abstract.tex}}" in main
    assert main.index("\\input{sections/external}") < main.index("\\input{sections/why}") < main.index("\\input{sections/negative}")
    assert "framing_intro.tex" in (ROOT / "paper/sections/intro.tex").read_text()
    assert "framing_conclusion.tex" in (ROOT / "paper/sections/conclusion.tex").read_text()
    for kept in ("abstract.tex", "intro_finding.tex", "conclusion.tex"):   # pre-registered record stays
        assert (FINAL / kept).exists()


def test_locked_sentence_is_the_preregistered_one():
    mod = _framing()
    abstract = (FINAL / "framing_abstract.tex").read_text()
    pf = mod.pf
    e = mod.facts()["d"]["diff"][mod.rs.PRIMARY]
    assert "D$-$B is $%+.3f$ (95\\%% interval [$%+.3f$, $%+.3f$])" % (e["point"], e["lo"], e["hi"]) in abstract
    assert "is not supported at this scale" in abstract and pf.SCALE in abstract and mod.BENCH in abstract
    assert "more than training it on those questions directly" in mod.BENCH            # H1 as in setup.tex
    assert "more than fine-tuning the same base on those questions directly" in (ROOT / "paper/sections/setup.tex").read_text()
    assert "On an adapted $500$-graph VGCure evaluation" in abstract
    assert "separately trained extraction adapter" in abstract
    assert "was decided after both were read" in (FINAL / "framing_intro.tex").read_text()


def test_refuses_when_a_stated_direction_fails():
    mod = _framing()
    ext = json.loads(json.dumps(EXT)); ext["arms"]["D"]["diff_per_seed"][1]["external_macro"]["lo"] = -0.01
    with pytest.raises(SystemExit):
        mod.facts(ext=ext)
    brk = json.loads(json.dumps(BRK)); brk["arms"]["D"]["pooled"]["accuracy"]["edge_false"] = 0.5
    with pytest.raises(SystemExit):
        mod.facts(brk=brk)


def test_locked_test_paragraph_numbers():
    arms = {a["label"]: a["pooled"] for a in PRIMARY["arms"]}
    assert "D is level with B (%s against %s" % (n3(arms["D"]["reachability"]["point"]), n3(arms["B"]["reachability"]["point"])) in TEX
    assert "critical-edge task (%s against %s)" % (n3(arms["D"]["critical_edge"]["point"]), n3(arms["B"]["critical_edge"]["point"])) in TEX
    mod = _framing()
    slices = json.loads((FINAL / "final_slices.json").read_text())
    d, b = mod.said_true(slices, "critical_edge", "D"), mod.said_true(slices, "critical_edge", "B")
    assert "D answers \\emph{true} to %s of the questions" % n3(d["said_true"]) in TEX
    assert "right on %s of the items whose answer is true and on only %s" % (n3(d["acc_true"]), n3(d["acc_false"])) in TEX
    assert "(B: %s and %s)" % (n3(b["acc_true"]), n3(b["acc_false"])) in TEX
    pairs = json.loads((FINAL / "final_pairs.json").read_text())
    both = {}
    for s in mod.MAIN_SPLITS:
        for arm in pairs["tables"][s]["arms"]:
            for seed in arm["per_seed"].values():
                c = seed["pairs"]["critical_edge"]
                both.setdefault(arm["label"], [0, 0])
                both[arm["label"]][0] += c["both_right"]; both[arm["label"]][1] += c["pairs"]
    assert both["D"][1] == 1800 and "correctly on $%d$ of $1{,}800$ pairs, against $%d$ for B" % (both["D"][0], both["B"][0]) in TEX
    r = {a: mod.said_true(slices, "reachability", a) for a in "BD"}
    assert "D is right on %s of unreachable pairs and %s of reachable ones (B: %s and %s)" % (
        n3(r["D"]["acc_false"]), n3(r["D"]["acc_true"]), n3(r["B"]["acc_false"]), n3(r["B"]["acc_true"])) in TEX
    met = json.loads((FINAL / "final_metrics.json").read_text())

    def reach(split, label):
        arm = [a for a in met["tables"][split]["arms"] if a["label"] == label][0]
        v = [s["per_task"]["reachability"]["accuracy"] for s in arm["per_seed"].values()]
        return sum(v) / len(v)
    style = reach("test_style_ood", "D") - reach("test_style_ood", "B")
    ind = reach("test_id", "D") - reach("test_id", "B")
    ext = EXT["arms"]["D"]["diff_pooled"]["reachability_balanced_accuracy"]["point"]
    assert "D$-$B is $%+.3f$ on our style-OOD split and $%+.3f$ on the in-distribution split, against $%+.3f$ on VGCure" % (
        style, ind, ext) in TEX
    for s in ("test_id", "test_style_ood"):   # our reachability items are balanced, like VGCure's balanced accuracy
        tr = json.loads((FINAL / "final_slices.json").read_text())["tables"][s]["tasks"]["reachability"]["truth"]
        assert tr["True"]["n"] == tr["False"]["n"]


def test_vgcure_paragraph_numbers():
    assert BRK["record"] == "preregistered" and BRK["subset_sha256"] == EXT["subset_sha256"]
    B, D, E = (BRK["arms"][a]["pooled"] for a in "BDE")
    assert "B says that the edge exists to %s of them and accepts %s of the false claims" % (
        n3(B["said_true"]["edge"]), n3(1 - B["accuracy"]["edge_false"])) in TEX
    assert "D says so to %s and accepts %s" % (n3(D["said_true"]["edge"]), n3(1 - D["accuracy"]["edge_false"])) in TEX
    assert "D is right on %s of the false claims against %s" % (n3(D["accuracy"]["edge_false"]), n3(B["accuracy"]["edge_false"])) in TEX
    assert "(%s against %s)" % (n3(D["accuracy"]["edge_true"]), n3(B["accuracy"]["edge_true"])) in TEX
    assert "unreachable pairs (%s against %s; reachable pairs %s against %s)" % (
        n3(D["accuracy"]["reach_no"]), n3(B["accuracy"]["reach_no"]), n3(D["accuracy"]["reach_yes"]), n3(B["accuracy"]["reach_yes"])) in TEX
    src = B["per_source"]
    edge_above = sum(D["per_source"][s]["edge"] > src[s]["edge"] + 1e-12 for s in src)
    reach = [D["per_source"][s]["reach"] - src[s]["reach"] for s in src]
    assert len(src) == 10 and edge_above == 10 and "above B's in all ten VGCure sources" in TEX
    above, equal = sum(x > 1e-9 for x in reach), sum(abs(x) <= 1e-9 for x in reach)
    words = {7: "seven", 1: "one", 2: "two"}
    assert "above in %s, equal in %s and below in %s" % (words[above], words[equal], words[10 - above - equal]) in TEX
    assert "no usable graph for $%d\\%%$" % round(100 * E["format_failure"]) in TEX
    assert "accepts only %s of the true edge claims and %s of the reachable pairs" % (
        n3(E["accuracy"]["edge_true"]), n3(E["accuracy"]["reach_yes"])) in TEX and "(%s)" % n3(E["accuracy"]["edge_false"]) in TEX
    e_locked = [d for d in PRIMARY["differences"] if d["a"] == "E" and d["b"] == "B"][0]["diff"]["primary_six_cell"]["point"]
    assert "($%+.3f$ over B)" % e_locked in TEX and "($%+.3f$)" % EXT["arms"]["E"]["diff_pooled"]["external_macro"]["point"] in TEX


def test_disclosure_and_scope():
    for s in ("was written after both results were known", "taken after reading both", "\\texttt{finding-v1}",
              "the hypothesis is still decided by the locked test alone", "which we did not test",
              "we leave as a hypothesis to test", "Nor can we separate a task effect from a style effect",
              "does not rule out a part for perception", "the repair prompt lists the edges that may be added"):
        assert s in TEX, s
    assert "word for word" not in TEX and "the problem is not perception" not in TEX


def test_repair_prompt_lists_absent_additions():
    src = (ROOT / "src/trace_vr/validate.py").read_text()
    assert '"repair.additions_addable"' in src and "not in eset" in src


def _raw_reach(arm):
    per = BRK["arms"][arm]["per_seed"]
    raw = [(s["accuracy"]["reach_yes"] * s["n"]["reach_yes"] + s["accuracy"]["reach_no"] * s["n"]["reach_no"])
           / (s["n"]["reach_yes"] + s["n"]["reach_no"]) for s in per]
    return sum(raw) / len(raw)


def test_vgcure_reachability_is_imbalanced_and_raw_accuracy_given():
    n = BRK["arms"]["B"]["per_seed"][0]["n"]
    d, b = _raw_reach("D"), _raw_reach("B")
    claim = "%s against %s ($%+.3f$)" % (n3(d), n3(b), round(d, 3) - round(b, 3))
    assert "($%d$ yes, $%d$ no)" % (n["reach_yes"], n["reach_no"]) in TEX and n["reach_yes"] != n["reach_no"]
    assert claim in TEX and "balanced between yes and no on both test sets" not in TEX
    ext = " ".join((ROOT / "paper/sections/external.tex").read_text().split())
    assert "D leads by $%+.3f$ (%s against %s)" % (round(d, 3) - round(b, 3), n3(d), n3(b)) in ext
    assert "$%d$ yes and $%d$ no" % (n["reach_yes"], n["reach_no"]) in ext


def test_abstract_gives_e_external_number_and_conclusion_hedges_perception():
    mod = _framing()
    f = mod.facts()
    abstract = (FINAL / "framing_abstract.tex").read_text()
    assert "scores $%.3f$ below direct question-answer fine-tuning on VGCure" % -f["e_ext"]["point"] in abstract
    assert "gives no usable graph for %d\\%% of the images" % round(100 * f["e_unusable"]) in abstract
    assert "fails to transcribe the foreign drawings" not in abstract
    concl = (FINAL / "framing_conclusion.tex").read_text()
    split = json.loads((ROOT / "reports/pilot/perception_split_dev.json").read_text())["arms"]["D_colab"]["readable"]["vs_ref"]
    assert "the repair arm stays $%.3f$ below the reference" % -split["point"] in concl
    assert "do not separate perception from reasoning" in concl and "that separate perception errors" not in concl
    for path in ("results.tex", "negative.tex", "related.tex", "analysis.tex"):
        t = " ".join((ROOT / "paper/sections" / path).read_text().split())
        assert "this is a reasoning gap, not a perception gap" not in t and "one of reasoning, not" not in t
        assert "our separation of perception" not in t and "same trained adapter answers" not in t
