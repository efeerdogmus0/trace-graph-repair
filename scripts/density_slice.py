"""Edges-per-node-matched slice of an OOD split (CPU only; protocol item 7).

Usage:
  python scripts/density_slice.py <manifest> <split> [--ref <manifest> <split>] [--flags flags.json]
                                  [--lo 0.05] [--hi 0.95] [--out file.json]

Without --flags it reports only how many roots of <split> fall inside the reference window
(central lo..hi quantile range of edges per node over the reference roots; default reference:
manifests/pilot.json train). Density is read from drawing metadata, not labels. With --flags
(a FLAGS-SUMMARY file for <split>) it also scores all vs matched roots, which reads labels, so
locked test splits need the final-evaluation unlock."""
import json, sys
from pathlib import Path
from trace_vr import analysis, arms, bootstrap, dataset

args = sys.argv[1:]
def opt(name, n=1, default=None):
    if name not in args:
        return default
    i = args.index(name); v = args[i + 1:i + 1 + n]; del args[i:i + 1 + n]
    return v if n > 1 else v[0]

ref = opt("--ref", 2, ["manifests/pilot.json", "train"])
flags_path = opt("--flags")
lo, hi = float(opt("--lo", default="0.05")), float(opt("--hi", default="0.95"))
out_path = opt("--out")
manifest_path, split = Path(args[0]), args[1]
if ref[1].startswith("test"):
    dataset.check_split_allowed(ref[1], "eval")

_, _, ref_by, _ = dataset.load_dataset(Path(ref[0]))
window = analysis.density_window(analysis.root_density(ref_by[ref[1]]), lo, hi)
window["ref"] = "%s:%s" % (Path(ref[0]).name, ref[1])
man, _, by_split, _ = dataset.load_dataset(manifest_path)
rows, flags = by_split[split], None
if flags_path:
    dataset.check_split_allowed(split, "eval")
    payload = json.loads(Path(flags_path).read_text())
    if payload["content_digest"] != man["content_digest"] or payload["split"] != split:
        raise SystemExit("flags come from a different dataset build or split")
    rows = arms.eval_samples(rows, payload["eval_root_graphs"])
    flags = bootstrap.decode_flags(rows, payload)
res = analysis.density_matched(rows, window, flags)
res["split"] = "%s:%s" % (manifest_path.name, split)
if out_path:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n")
print("reference %s: %d roots, edges/node mean %.3f, window [%.3f, %.3f] (q %.2f-%.2f)" % (
    window["ref"], window["ref_roots"], window["ref_mean"], window["lo"], window["hi"], lo, hi))
print("%s: %d roots, mean %.3f; matched %d (mean %s), below %d, above %d" % (
    res["split"], res["roots"], res["mean_density_all"], res["matched_roots"],
    "n/a" if res["mean_density_matched"] is None else "%.3f" % res["mean_density_matched"], res["below"], res["above"]))
for subset, per_task in res.get("accuracy", {}).items():
    print("%-8s %s" % (subset, "  ".join("%s %.3f" % (t, c if isinstance(c, float) else c["accuracy"])
                                         for t, c in sorted(per_task.items()) if c is not None)))
