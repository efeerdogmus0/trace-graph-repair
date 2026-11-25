"""Root-graph bootstrap intervals from saved per-sample flags (CPU only).

Usage: python scripts/bootstrap_ci.py <manifest> <flags_a.json> [<flags_b.json>] [--out file.json] [--boot 2000] [--roots N]
With two flag files the paired difference a - b is reported as well. --roots N restricts every
file to the first N evaluated root graphs (to pair a 200-root run with a 100-root run)."""
import json, sys
from pathlib import Path
from trace_vr import arms, bootstrap, dataset

args = sys.argv[1:]
out_path, n_boot, n_roots = None, 2000, None
if "--out" in args:
    i = args.index("--out"); out_path = Path(args[i + 1]); del args[i:i + 2]
if "--boot" in args:
    i = args.index("--boot"); n_boot = int(args[i + 1]); del args[i:i + 2]
if "--roots" in args:
    i = args.index("--roots"); n_roots = int(args[i + 1]); del args[i:i + 2]
manifest, files = args[0], [Path(p) for p in args[1:]]
man, _, by_split, _ = dataset.load_dataset(Path(manifest))
payloads = [json.loads(p.read_text()) for p in files]
for k, p in enumerate(payloads):
    if p["content_digest"] != man["content_digest"]:
        raise SystemExit("flags come from a different dataset build")
    if n_roots is not None and p["eval_root_graphs"] != n_roots:
        if p["eval_root_graphs"] < n_roots:
            raise SystemExit("%s covers fewer than %d root graphs" % (files[k], n_roots))
        big = arms.eval_samples(by_split[p["split"]], p["eval_root_graphs"])
        payloads[k] = dict(bootstrap.subset_flags(big, p, arms.eval_samples(by_split[p["split"]], n_roots)),
                           eval_root_graphs=n_roots)
    if (payloads[k]["split"], payloads[k]["eval_root_graphs"]) != (payloads[0]["split"], payloads[0]["eval_root_graphs"]):
        raise SystemExit("the two flag files cover different samples")
rows = arms.eval_samples(by_split[payloads[0]["split"]], payloads[0]["eval_root_graphs"])
flags = [bootstrap.decode_flags(rows, p) for p in payloads]
res = bootstrap.cluster_bootstrap(rows, flags[0], flags[1] if len(flags) > 1 else None, n_boot=n_boot)
res["inputs"] = [{"file": str(f), "run": p.get("run"), "predictions_sha256": p.get("predictions_sha256"),
                  "images_digest": p.get("images_digest"), "missing_responses": p.get("missing_responses")}
                 for f, p in zip(files, payloads)]
text = json.dumps(res, indent=1)
if out_path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")


def show(label, block):
    for k, e in block.items():
        if e:
            extra = ("  share>0 %.3f" % e["share_above_zero"]) if "share_above_zero" in e else ""
            print("%-8s %-26s %+.4f  [%+.4f, %+.4f]%s" % (label, k, e["point"], e["lo"], e["hi"], extra))


print("root graphs %d, replicates %d, %d%% percentile intervals" % (res["root_graphs"], res["n_boot"], round(100 * res["level"])))
show("a", res["a"])
if "b" in res:
    show("b", res["b"]); show("a-b", res["a_minus_b"])
