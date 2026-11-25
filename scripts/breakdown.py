"""Error analysis from a saved flags file (CPU only).

Usage: python scripts/breakdown.py <manifest> <flags.json> [--invariance flags_inv.json] [--out file.json]"""
import json, sys
from pathlib import Path
from trace_vr import analysis, arms, bootstrap, dataset

args = sys.argv[1:]
out_path = inv_path = None
if "--invariance" in args:
    i = args.index("--invariance"); inv_path = Path(args[i + 1]); del args[i:i + 2]
if "--out" in args:
    i = args.index("--out"); out_path = Path(args[i + 1]); del args[i:i + 2]
man, _, by_split, _ = dataset.load_dataset(Path(args[0]))
payload = json.loads(Path(args[1]).read_text())
if payload["content_digest"] != man["content_digest"]:
    raise SystemExit("flags come from a different dataset build")
rows = arms.eval_samples(by_split[payload["split"]], payload["eval_root_graphs"])
flags = bootstrap.decode_flags(rows, payload)
fmt = bootstrap.decode_flags(rows, payload, "format_ok")
res = {"run": payload.get("run"), "predictions_sha256": payload.get("predictions_sha256"),
       "format_failures": sum(1 for v in fmt.values() if not v),
       "answer_bias": analysis.answer_bias(rows, flags, fmt),
       "critical_pairs": analysis.pair_table(rows, flags),
       "breakdown": analysis.breakdown(rows, flags)}
if inv_path:
    inv = json.loads(inv_path.read_text())
    if (inv["content_digest"], inv["split"], inv["eval_root_graphs"], inv.get("sample_set")) != (
            payload["content_digest"], payload["split"], payload["eval_root_graphs"], "invariance"):
        raise SystemExit("invariance flags do not belong to the same build, split and roots")
    inv_rows = arms.invariance_samples(by_split[payload["split"]], payload["eval_root_graphs"])
    res["invariance"] = analysis.invariance_table(rows, flags, inv_rows, bootstrap.decode_flags(inv_rows, inv))
if out_path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1) + "\n")
print("run %s, format failures %d" % (res["run"], res["format_failures"]))
for task, e in res["answer_bias"].items():
    print("bias  %-22s said true %s, true was right %.3f (n %d, no answer %d)" % (
        task, "n/a" if e["said_true_rate"] is None else "%.3f" % e["said_true_rate"], e["truth_true_rate"], e["n"], e["no_answer"]))
for task, e in res["critical_pairs"].items():
    print("pairs %-22s both right %d, one right %d, both wrong %d (of %d)" % (task, e["both_right"], e["one_right"], e["both_wrong"], e["pairs"]))
for task, facets in res["breakdown"].items():
    for facet, values in facets.items():
        print("%-22s %-14s %s" % (task, facet, "  ".join("%s %.3f (%d)" % (k, v["accuracy"], v["n"]) for k, v in sorted(values.items()))))
for kind, per_task in res.get("invariance", {}).items():
    for task, e in per_task.items():
        print("inv   %-18s %-22s both right %.3f, same correctness %.3f (of %d)" % (kind, task, e["both_right_rate"], e["same_correctness_rate"], e["pairs"]))
