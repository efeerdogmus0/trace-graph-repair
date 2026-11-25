"""Abstract, intro finding paragraph and conclusion that give both pre-registered evaluations their place (CPU only).

Usage: python scripts/paper_framing.py [--out-dir reports/final]

Rule `framing-v2`, chosen by the owner on 2026-11-23 AFTER both the locked test and the pre-registered VGCure subset
had been read (DECISIONS 2026-11-23T20:30Z). It replaces finding-v1 / conclusion-v1 as the text the paper shows; those
two rules fixed, before the locked test was opened, that only the primary statistic goes into the abstract. Their
outputs (`abstract.tex`, `intro_finding.tex`, `conclusion.tex`) are kept unchanged as the record, and the paper says
that the emphasis was chosen after reading (paper/sections/why.tex).

What does not change: the hypothesis (H1 of paper/sections/setup.tex: repair training improves the non-edit
questions more than direct question-answer training) is decided by the locked test only, with the same numbers and
reading as finding-v1; the pilot-scale sentence is imported from scripts/paper_finding.py. The locked-test sentence
itself is reworded, not copied. Every other number is read from a tracked report:
reports/external/vgcure_final_seeds_vs_B.json, reports/external/vgcure_final_breakdown.json,
reports/final/final_slices.json. The script refuses (writes nothing) if any direction a fixed sentence states does not
hold in those files.

Writes framing_abstract.tex, framing_intro.tex and framing_conclusion.tex into --out-dir.
"""
import importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = "framing-v2"
_spec = importlib.util.spec_from_file_location("paper_conclusion", ROOT / "scripts/paper_conclusion.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)
pf, rs = pc.pf, pc.rs
MAIN_SPLITS = ("test_id", "test_style_ood", "test_struct_ood")
EXT_B = ROOT / "reports/external/vgcure_final_seeds_vs_B.json"
BREAK = ROOT / "reports/external/vgcure_final_breakdown.json"
SLICES = ROOT / "reports/final/final_slices.json"
SPLIT = ROOT / "reports/pilot/perception_split_dev.json"


def said_true(slices, task, arm, splits=MAIN_SPLITS):
    """Share of "true" answers and accuracy on true / false items, summed over splits and seeds."""
    yes = n = rt = nt = rf = nf = 0
    for s in splits:
        tr = slices["tables"][s]["tasks"][task]["truth"]
        t, f = tr["True"]["arms"][arm], tr["False"]["arms"][arm]
        yes += t["right"] + f["n"] - f["right"]; n += t["n"] + f["n"]
        rt, nt, rf, nf = rt + t["right"], nt + t["n"], rf + f["right"], nf + f["n"]
    return {"said_true": yes / n, "acc_true": rt / nt, "acc_false": rf / nf}


def facts(primary=None, ext=None, brk=None, slices=None):
    t = primary if primary is not None else rs.load(str(ROOT / "reports/final/final_primary.json"))
    ext = ext if ext is not None else json.loads(EXT_B.read_text())
    brk = brk if brk is not None else json.loads(BREAK.read_text())
    slices = slices if slices is not None else json.loads(SLICES.read_text())
    ref, d, _diffs, _arms = rs.parts(t)
    dx = ext["arms"]["D"]
    f = {"ref": ref, "d": d, "reading": d["reading"],
         "ext": dx["diff_pooled"]["external_macro"], "ext_seeds": [s["external_macro"] for s in dx["diff_per_seed"]],
         "ext_edge": dx["diff_pooled"]["edge_exists_accuracy"], "ext_reach": dx["diff_pooled"]["reachability_balanced_accuracy"],
         "e_ext": ext["arms"]["E"]["diff_pooled"]["external_macro"],
         "e_locked": [x for x in t["differences"] if x["a"] == "E" and x["b"] == ref][0]["diff"][rs.PRIMARY],
         "e_unusable": brk["arms"]["E"]["pooled"]["format_failure"],
         "readable": json.loads(SPLIT.read_text())["arms"]["D_colab"]["readable"]["vs_ref"],
         "crit_D": said_true(slices, "critical_edge", "D"), "crit_B": said_true(slices, "critical_edge", "B"),
         "vB": brk["arms"]["B"]["pooled"], "vD": brk["arms"]["D"]["pooled"]}
    checks = [
        (ext["record"] == "preregistered" and ext["reference"] == ref == "B", "external record is not the pre-registered one vs B"),
        (f["reading"] == "negative", "the sentences below describe a negative primary reading"),
        (f["ext"]["lo"] > 0 and all(s["lo"] > 0 for s in f["ext_seeds"]), "D is not above B on VGCure at every seed"),
        (f["ext_edge"]["lo"] > 0 and f["ext_reach"]["lo"] > 0, "D's VGCure lead is not in both tasks"),
        (f["e_ext"]["hi"] < 0, "E is not below B on VGCure"),
        (f["e_locked"]["lo"] > 0, "E is not above B on the locked test"),
        (f["readable"]["hi"] < 0, "D is not below B on the exactly transcribed development drawings"),
        (f["crit_D"]["said_true"] > 0.7 and f["crit_B"]["said_true"] < 0.6, "D does not lean on 'true' for critical edge"),
        (f["vD"]["accuracy"]["edge_false"] > f["vB"]["accuracy"]["edge_false"]
         and f["vD"]["accuracy"]["reach_no"] > f["vB"]["accuracy"]["reach_no"], "D's VGCure gain is not on absent edges/paths"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    if bad:
        raise SystemExit("framing-v2 refuses: " + "; ".join(bad))
    return f


def n3(x):
    return "$%.3f$" % x


def pct(x):
    return "%d\\%%" % round(100 * x)


def s3(x):
    return "$%+.3f$" % x


def interval(e):
    return "%s (95\\%% interval [%s, %s]" % (s3(e["point"]), s3(e["lo"]), s3(e["hi"]))


BENCH = ("We introduce a procedural benchmark of drawn directed graphs with exact labels, a verifier that accepts every "
         "optimal repair edit, and answer-flipping twins that cap any image-blind predictor at $0.50$ per task, and use it "
         "to test whether training a small vision-language model only to repair drawings improves its answers to "
         "reachability and edge-criticality questions more than training it on those questions directly.")


def external_sentence(f):
    return ("On an adapted $500$-graph VGCure evaluation (drawings by other authors in a style none of our adapters saw, "
            "with our prompts, images scaled to our input size and strict scoring), the repair-trained adapter (arm~D) "
            "answers edge-existence and reachability questions better than direct question-answer fine-tuning (arm~B): "
            "the difference is %s, mean of 3 seeds, above zero at every seed)." % interval(f["ext"]))


def locked_sentence(f):
    e = f["d"]["diff"][rs.PRIMARY]
    return ("On our own pre-registered locked test, which decides the hypothesis, D does worse than B: D$-$B is %s), "
            "so the hypothesis is not supported at this scale." % interval(e))


def why_sentence(f):
    return ("The loss is confined to edge criticality, a task VGCure does not contain, where D answers \\emph{true} to "
            "%s of the questions (true is right for half); the VGCure gain comes mostly from false edge claims and "
            "unreachable pairs, which D rejects more often than B." % pct(f["crit_D"]["said_true"]))


def e_sentence(f):
    return ("A parse-then-solve arm, which reads the graph with a separately trained extraction adapter, performs best on "
            "our benchmark (E$-$B %s) but scores %s below direct question-answer fine-tuning on VGCure, where its parser "
            "gives no usable graph for %s of the images." % (s3(f["e_locked"]["point"]), n3(-f["e_ext"]["point"]),
                                                             pct(f["e_unusable"])))


DISCLOSE = ("Both evaluations were fixed before any of their outputs existed; giving the external result this place in "
            "the abstract and introduction was decided after both were read, while the wording rule fixed in advance put "
            "only the locked-test statistic here (Section~\\ref{sec:why}).")


def contrib(f):
    return ("We built a procedural benchmark of drawn directed graphs in which every label and every optimal repair is "
            "checked by an independent verifier, answer-flipping twins bound any image-blind predictor at $0.50$ per task, "
            "and splits are grouped by root graph with held-out styles, families and sizes. On it we ran a pre-registered, "
            "exposure-matched comparison of repair training against question-answering and structural fine-tuning, with "
            "oracle-graph and parse-and-solve diagnostics. They narrow down but do not separate perception from reasoning: "
            "even on the development drawings that the parse-and-solve arm transcribes exactly, the repair arm stays %s "
            "below the reference, so its deficit does not come from hard-to-read drawings, while neither diagnostic "
            "measures what the end-to-end arms perceive. We release the generator, verifier, protocol, frozen evaluation "
            "code and per-sample results so the comparison can be rerun or extended." % n3(-f["readable"]["point"]))


def render(f):
    tag = "%% generated by scripts/paper_framing.py, rule %s (chosen after reading, see DECISIONS 2026-11-23T20:30Z); do not edit\n" % RULE
    abstract = tag + " ".join([BENCH, external_sentence(f), locked_sentence(f), why_sentence(f), e_sentence(f),
                               pf.SCALE]) + "\n"
    intro = tag + ("\\paragraph{Main findings.} " + " ".join([
        external_sentence(f),
        "The lead comes from both tasks: edge existence %s and reachability %s." % (s3(f["ext_edge"]["point"]), s3(f["ext_reach"]["point"])),
        locked_sentence(f),
        "Section~\\ref{sec:why} traces the disagreement: on our test D's deficit is in the critical-edge task, where it "
        "answers \\emph{true} to %s of the questions, while its reachability is level with B's; on VGCure, which has no "
        "critical-edge items, D gains on false edge claims and unreachable pairs." % pct(f["crit_D"]["said_true"]),
        pf.SCALE, DISCLOSE,
        "Per-split, per-task and secondary comparisons are in Section~\\ref{sec:test-results}."]) + "\n")
    p2 = " ".join([
        "Repair training gave mixed results.",
        "On an adapted evaluation on drawings made by other authors (VGCure) it beat direct question-answer training on "
        "edge existence and reachability (D$-$B %s); on our pre-registered locked test it did worse (D$-$B %s), so the "
        "hypothesis is not supported at this scale." % (s3(f["ext"]["point"]), s3(f["d"]["diff"][rs.PRIMARY]["point"])),
        "The two results separate by task: the loss is in edge criticality, where the repair model's answers follow an "
        "answer prior, and the gain is on false edge claims and unreachable pairs; why repair training helps there is a "
        "hypothesis we did not test (Section~\\ref{sec:why}).",
        "Section~\\ref{sec:negative} records the development observations that pointed the same way."])
    p3 = " ".join([pf.SCALE,
                   "Whether the gain on foreign drawings holds for critical-edge questions on such drawings, with more "
                   "data or with a second base model is the next question; the released protocol can be rerun unchanged "
                   "for each.", pc.LIMITS])
    conclusion = tag + "\n\n".join([contrib(f), p2, p3]) + "\n"
    return {"framing_abstract.tex": abstract, "framing_intro.tex": intro, "framing_conclusion.tex": conclusion}


def main(argv):
    out = Path(argv[argv.index("--out-dir") + 1]) if "--out-dir" in argv else ROOT / "reports/final"
    texts = render(facts())
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (out / name).write_text(text)
        print("wrote %s" % (out / name))


if __name__ == "__main__":
    main(sys.argv[1:])
