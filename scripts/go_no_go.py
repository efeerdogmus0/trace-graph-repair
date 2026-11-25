"""Apply the pre-registered go/no-go rule (src/trace_vr/go_no_go.py) to collected flags (CPU only).

Usage: python scripts/go_no_go.py <manifest> <B_colab_flags> <B_kaggle_flags> <D_colab_flags> <D2_kaggle_flags> [--out file.json]
All four files must cover the same dev samples (200 roots)."""
import json, sys
from pathlib import Path
from trace_vr import arms, bootstrap, dataset, go_no_go

args = sys.argv[1:]
out_path = None
if "--out" in args:
    i = args.index("--out"); out_path = Path(args[i + 1]); del args[i:i + 2]
manifest, b_colab, b_kaggle, d_colab, d2_kaggle = args[0], *[Path(p) for p in args[1:5]]
man, _, by_split, _ = dataset.load_dataset(Path(manifest))
pay = {}
for name, p in (("B_colab", b_colab), ("B_kaggle", b_kaggle), ("D_colab", d_colab), ("D2_kaggle", d2_kaggle)):
    d = json.loads(p.read_text())
    if d["content_digest"] != man["content_digest"]:
        raise SystemExit("%s comes from a different dataset build" % p)
    pay[name] = d
keys = {(d["split"], d["eval_root_graphs"]) for d in pay.values()}
if len(keys) != 1:
    raise SystemExit("flag files cover different samples: %s" % sorted(keys))
split, n_roots = keys.pop()
rows = arms.eval_samples(by_split[split], n_roots)
fl = {k: bootstrap.decode_flags(rows, v) for k, v in pay.items()}
M = bootstrap.MACRO


def diff(a, b):
    return bootstrap.cluster_bootstrap(rows, fl[a], fl[b])["a_minus_b"][M]


entries = {"machine": diff("B_kaggle", "B_colab"), "d2_vs_b": diff("D2_kaggle", "B_kaggle"),
           "d_vs_b": diff("D_colab", "B_colab")}
res = dict(go_no_go.decide(**entries), entries=entries, split=split, root_graphs=n_roots,
           inputs={k: str(p) for k, p in (("B_colab", b_colab), ("B_kaggle", b_kaggle),
                                          ("D_colab", d_colab), ("D2_kaggle", d2_kaggle))})
text = json.dumps(res, indent=1)
if out_path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")
for k, e in entries.items():
    print("%-8s %+.4f [%+.4f, %+.4f]" % (k, e["point"], e["lo"], e["hi"]))
print("decision:", res["decision"], "-", res["reason"])
