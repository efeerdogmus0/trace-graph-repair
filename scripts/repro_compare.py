"""Compare a clean-clone GPU rerun with the recorded run it repeats (rule `repro-v1`, CPU only).

Usage: python scripts/repro_compare.py <recorded_run_dir> <rerun_dir> [--out reports/repro.json] [--out-md reports/repro.md]
  e.g. runs/pilot3_D_kaggle runs/repro_D_s3407

Rule `repro-v1`, written 2026-11-20 before the rerun exists (DECISIONS 2026-11-20T09:40Z):
- Both directories must hold `summary.json` and `flags_dev.json` of the same arm, seed, split, sample set,
  input mode, number of samples, content digest and sample order (`order_sha`); otherwise the script refuses.
  The images digest is recorded, not compared (image bytes are rendered per machine, DECISIONS 2026-11-12).
- Reported: per-task accuracy of both runs and the difference (rerun - recorded), the non-edit macro, the share
  of samples whose correct/wrong bit agrees, whether the predictions files are byte-identical, and the first/last
  100-update mean training loss of both runs.
- Verdict:
    identical       predictions sha256 equal (then every number above is equal too);
    reproduced      every per-task |difference| <= 0.03 and non-edit macro |difference| <= 0.02;
    not_reproduced  otherwise.
  The tolerances are about twice the largest spread between arm D's three pre-registered seeds on dev
  (per task 0.015, macro 0.00625; reports/pilot/seeds_dev_interim.json). A rerun of the same seed on the same
  machine type is expected to be at least that close; GPU kernels are not bit-deterministic, so identity is not
  required. The rule is not changed after the rerun is read.
Reads only the two run directories; no locked-test file.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = "repro-v1"
TASKS = ("minimal_graph_repair", "reachability", "critical_edge")
TASK_TOL = 0.03
MACRO_TOL = 0.02
SAME_FLAGS = ("n", "split", "input_mode", "sample_set", "eval_root_graphs", "content_digest", "order_sha")
SAME_TRAIN = ("arm", "seed", "manifest", "epochs", "base_id", "revision", "content_digest")


def load(run_dir: Path):
    out = {}
    for name in ("summary.json", "flags_dev.json"):
        p = run_dir / name
        if not p.exists():
            raise SystemExit("missing %s" % p)
        out[name] = json.loads(p.read_text())
    s = out["summary.json"]
    if "eval_dev" not in s or "train" not in s:
        raise SystemExit("%s/summary.json has no train / eval_dev block" % run_dir)
    return s, out["flags_dev.json"]


def compare(recorded_dir: Path, rerun_dir: Path) -> dict:
    s0, f0 = load(recorded_dir)
    s1, f1 = load(rerun_dir)
    for key in SAME_FLAGS:
        if f0.get(key) != f1.get(key):
            raise SystemExit("refused: flags field %r differs (%r vs %r)" % (key, f0.get(key), f1.get(key)))
    for key in SAME_TRAIN:
        if s0["train"].get(key) != s1["train"].get(key):
            raise SystemExit("refused: training field %r differs (%r vs %r)"
                             % (key, s0["train"].get(key), s1["train"].get(key)))
    for f in (f0, f1):
        if len(f["correct"]) != f["n"]:
            raise SystemExit("refused: flags length differs from n")
    m0, m1 = s0["eval_dev"]["metrics"], s1["eval_dev"]["metrics"]
    tasks = {}
    for t in TASKS:
        a, b = m0["per_task"][t]["accuracy"], m1["per_task"][t]["accuracy"]
        tasks[t] = {"recorded": a, "rerun": b, "difference": round(b - a, 6)}
    macro0, macro1 = m0["non_edit_macro_accuracy"], m1["non_edit_macro_accuracy"]
    macro = {"recorded": macro0, "rerun": macro1, "difference": round(macro1 - macro0, 6)}
    agree = sum(x == y for x, y in zip(f0["correct"], f1["correct"])) / f0["n"]
    identical = f0["predictions_sha256"] == f1["predictions_sha256"]
    within = all(abs(v["difference"]) <= TASK_TOL for v in tasks.values()) and abs(macro["difference"]) <= MACRO_TOL
    verdict = "identical" if identical else ("reproduced" if within else "not_reproduced")
    loss = {k: {"recorded": s0["train"].get(k), "rerun": s1["train"].get(k)}
            for k in ("mean_loss_first_100", "mean_loss_last_100")}
    return {
        "rule": RULE, "task_tolerance": TASK_TOL, "macro_tolerance": MACRO_TOL,
        "recorded": str(recorded_dir.relative_to(ROOT)) if recorded_dir.is_relative_to(ROOT) else str(recorded_dir),
        "rerun": str(rerun_dir.relative_to(ROOT)) if rerun_dir.is_relative_to(ROOT) else str(rerun_dir),
        "arm": s0["train"].get("arm"), "seed": s0["train"].get("seed"), "samples": f0["n"],
        "per_task": tasks, "non_edit_macro_accuracy": macro,
        "bit_agreement": round(agree, 6), "predictions_identical": identical,
        "predictions_sha256": {"recorded": f0["predictions_sha256"], "rerun": f1["predictions_sha256"]},
        "images_digest": {"recorded": f0.get("images_digest"), "rerun": f1.get("images_digest")},
        "train_loss": loss,
        "environment": {"recorded": s0.get("kernel_environment_lines"), "rerun": s1.get("kernel_environment_lines")},
        "verdict": verdict,
    }


def markdown(r: dict) -> str:
    lines = ["# GPU rerun from a clean clone (rule `%s`)" % r["rule"], "",
             "Recorded run `%s`, rerun `%s`; arm %s, seed %s, %d dev samples." % (r["recorded"], r["rerun"], r["arm"], r["seed"], r["samples"]),
             "", "| task | recorded | rerun | difference |", "|---|---|---|---|"]
    for t, v in r["per_task"].items():
        lines.append("| %s | %.4f | %.4f | %+.4f |" % (t, v["recorded"], v["rerun"], v["difference"]))
    m = r["non_edit_macro_accuracy"]
    lines.append("| non-edit macro | %.4f | %.4f | %+.4f |" % (m["recorded"], m["rerun"], m["difference"]))
    lines += ["", "Per-sample correct/wrong agreement: %.4f. Predictions byte-identical: %s." % (r["bit_agreement"], "yes" if r["predictions_identical"] else "no"),
              "", "Verdict: **%s** (tolerance: each task %.2f, macro %.2f)." % (r["verdict"], r["task_tolerance"], r["macro_tolerance"]), ""]
    return "\n".join(lines)


def main(argv):
    out = out_md = None
    if "--out" in argv:
        i = argv.index("--out"); out = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if "--out-md" in argv:
        i = argv.index("--out-md"); out_md = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if len(argv) != 2:
        raise SystemExit(__doc__)
    paths = [Path(a) if Path(a).is_absolute() else ROOT / a for a in argv]
    r = compare(*paths)
    text = json.dumps(r, indent=1) + "\n"
    if out:
        Path(out).write_text(text)
    if out_md:
        Path(out_md).write_text(markdown(r))
    print("verdict", r["verdict"], "macro diff %+.4f" % r["non_edit_macro_accuracy"]["difference"], "bit agreement %.4f" % r["bit_agreement"])


if __name__ == "__main__":
    main(sys.argv[1:])
