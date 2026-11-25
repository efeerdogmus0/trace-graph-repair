"""How arm E's score depends on extraction errors, with no model involved.

Feeds the exact solver a TRUE graph with k randomly dropped edges (or k flipped directions)
per image and scores the derived answers against the truth. Usage:
python scripts/parse_solve_sensitivity.py <manifest> <split> <n_roots> [out_json]"""
import json, random, sys
from pathlib import Path
from trace_vr import arms, dataset, metrics, parse_solve as ps, tasks

manifest, split, n_roots = sys.argv[1], sys.argv[2], int(sys.argv[3])
dataset.check_split_allowed(split, "eval")
_, _, by_split, _ = dataset.load_dataset(Path(manifest))
rows = arms.eval_samples(by_split[split], n_roots)
truth = {s["image_id"]: s["private_truth"] for s in rows}
out = {"manifest": manifest, "split": split, "eval_root_graphs": n_roots, "samples": len(rows), "seed": "ps-sens-v1", "conditions": {}}
for kind in ("drop", "flip"):
    for k in (0, 1, 2):
        rng = random.Random("ps-sens-v1:%s:%d" % (kind, k))
        ext = {}
        for img in sorted(truth):
            t = truth[img]
            edges = [list(e) for e in t["edges"]]
            idx = rng.sample(range(len(edges)), min(k, len(edges)))
            if kind == "drop":
                edges = [e for i, e in enumerate(edges) if i not in idx]
            else:
                edges = [e[::-1] if i in idx else e for i, e in enumerate(edges)]
            ext[img] = json.dumps({"nodes": t["nodes"], "edges": edges})
        resp, reasons = ps.responses_for_samples(rows, ext)
        res = metrics.evaluate(rows, resp)
        key = "%s_%d" % (kind, k)
        out["conditions"][key] = {"per_task_accuracy": {t: round(e["accuracy"], 4) for t, e in res["per_task"].items()},
                                  "non_edit_macro": round(res["non_edit_macro_accuracy"], 4), "reasons": reasons}
        if kind == "flip" and k == 0:
            continue
        print(key, json.dumps(out["conditions"][key]), flush=True)
if len(sys.argv) > 4:
    Path(sys.argv[4]).write_text(json.dumps(out, indent=1) + "\n")
