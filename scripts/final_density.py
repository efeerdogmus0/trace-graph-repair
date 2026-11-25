"""Density-matched slice of the locked test from the collected final flags (CPU only, no model).

Usage: python scripts/final_density.py [--out-json f] [--out-md f] [--out-tex f] [--queue f]

Inputs are the files written by scripts/kaggle/collect_final.py: for every entry of
scripts/kaggle/final_queue.json and each of the six locked splits, <dev_dir>/flags_<split>.json (main samples).

What is reported (paper/sections/analysis.tex "Slices", fixed before the main runs): layout rejection leaves
large graphs sparser, so each locked split is also scored on the roots whose edges per node lie inside the
central 5-95% (nearest rank) of the 512 pilot training roots (analysis.density_window on manifests/pilot.json
train; the window is read from drawing metadata, never from labels or model output). Per split and arm: the
non-edit macro on all roots and on the matched roots, as the mean over the three training seeds, with the
per-seed values kept. All six splits are computed so that no split is picked after the fact; the paper's claim
concerns size-OOD. Descriptive only: no interval, no reading rule and no reference arm, because the
pre-registered decision uses the main table (scripts/final_table.py) alone. The slice is shown next to the full
split, never instead of it.

All-or-nothing like scripts/final_table.py: while any queue entry lacks any main flags file the script stops,
writes nothing and prints no number."""
import importlib.util, json, sys
from pathlib import Path
from trace_vr import analysis, arms, bootstrap, dataset

ROOT = Path(__file__).resolve().parents[1]
REF = ("pilot", "train")          # density reference: the training roots every arm saw
LO_Q, HI_Q = 0.05, 0.95           # as in analysis.tex and scripts/density_slice.py
MACRO = "non_edit_macro"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


final_table = _load("final_table", "scripts/final_table.py")
SPLITS = final_table.SPLITS
SEEDS = tuple(final_table.seed_table.seeds.PREREGISTERED_SEEDS)


def window():
    _, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (REF[0] + ".json"))
    w = analysis.density_window(analysis.root_density(by_split[REF[1]]), LO_Q, HI_Q)
    w["ref"] = "%s.json:%s" % REF
    return w


def _mean(values):
    return None if any(v is None for v in values) else sum(values) / len(values)


def build(jobs, splits, root=ROOT, win=None):
    root = Path(root)
    gaps = final_table.missing(jobs, splits, root)
    if gaps:
        raise SystemExit("locked-test density slice refused, %d of %d final runs not collected: %s"
                         % (len(gaps), len(jobs), ", ".join(j for j, _ in gaps)))
    seen = {}
    for e in jobs:
        seen.setdefault(e["arm"], []).append(e["seed"])
    bad = sorted(a for a, s in seen.items() if sorted(s) != sorted(SEEDS))
    if bad:
        raise SystemExit("every arm needs exactly the three pre-registered seeds; check: %s" % bad)
    win = window() if win is None else win
    tables = {}
    for manifest, split, roots in splits:
        man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
        rows = arms.select_samples(by_split[split], roots, "main")
        shape, per_arm = None, {}
        for e in jobs:
            p = root / e["dev_dir"] / ("flags_%s.json" % split)
            got = json.loads(p.read_text())
            if got.get("split") != split or got.get("eval_root_graphs") != roots or got.get("sample_set", "main") != "main":
                raise SystemExit("%s is not the main %s evaluation on %d roots" % (p, split, roots))
            if got.get("content_digest") != man["content_digest"]:
                raise SystemExit("%s comes from a different dataset build" % p)
            res = analysis.density_matched(rows, win, bootstrap.decode_flags(rows, got))
            acc = res.pop("accuracy")
            res.pop("window")
            shape = shape or res
            if res != shape:   # same roots for every run, so the slice itself cannot differ
                raise SystemExit("%s: matched roots differ between runs" % p)
            per_arm.setdefault(e["arm"], {})[str(e["seed"])] = {
                sub: (acc[sub][MACRO] if sub in acc else None) for sub in ("all", "matched")}
        out_arms = []
        for arm, per_seed in per_arm.items():
            out_arms.append({"label": arm, "seeds": sorted(int(s) for s in per_seed), "per_seed": per_seed,
                             "mean": {sub: _mean([per_seed[s][sub] for s in per_seed]) for sub in ("all", "matched")}})
        tables[split] = dict(shape, manifest=manifest, content_digest=man["content_digest"], arms=out_arms)
    return {"window": win, "jobs": [e["job"] for e in jobs], "splits": [s for _, s, _ in splits],
            "labels": list(seen), "tables": tables}


def _fmt(v):
    return "n/a" if v is None else "%.3f" % v


def markdown(t):
    labels, w = t["labels"], t["window"]
    lines = ["Locked test, density-matched slice: roots whose edges per node lie in [%.3f, %.3f], the central %d-%d%% "
             "of the %d training roots (%s). Per arm: non-edit macro on all roots / on the matched roots, mean over three "
             "seeds (%d runs). Descriptive, no intervals; the full split is the reported result."
             % (w["lo"], w["hi"], round(100 * w["lo_q"]), round(100 * w["hi_q"]), w["ref_roots"], w["ref"], len(t["jobs"])), "",
             "| Split | Roots | Matched | Edges/node all | Edges/node matched | " + " | ".join(labels) + " |",
             "|---|---|---|---|---|" + "---|" * len(labels)]
    for s in t["splits"]:
        tab = t["tables"][s]
        rows = {a["label"]: a for a in tab["arms"]}
        cells = ["%s / %s" % (_fmt(rows[l]["mean"]["all"]), _fmt(rows[l]["mean"]["matched"])) for l in labels]
        lines.append("| %s | %d | %d | %s | %s | %s |" % (s, tab["roots"], tab["matched_roots"], _fmt(tab["mean_density_all"]),
                                                       _fmt(tab["mean_density_matched"]), " | ".join(cells)))
    return "\n".join(lines) + "\n"


def latex(t):
    """Booktabs tabular (\\input into a table float): one row per split; per arm all / matched non-edit macro."""
    labels = t["labels"]
    esc = lambda x: str(x).replace("_", "\\_")
    out = ["% Generated by scripts/final_density.py; do not edit by hand.",
           "\\begin{tabular}{lrr%s}" % ("c" * len(labels)), "\\toprule",
           "Split & Roots & Matched & " + " & ".join(esc(l) for l in labels) + " \\\\", "\\midrule"]
    for s in t["splits"]:
        tab = t["tables"][s]
        rows = {a["label"]: a for a in tab["arms"]}
        cells = ["%s / %s" % (_fmt(rows[l]["mean"]["all"]), _fmt(rows[l]["mean"]["matched"])) for l in labels]
        out.append("%s & %d & %d & %s \\\\" % (esc(s), tab["roots"], tab["matched_roots"], " & ".join(cells)))
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def main(argv):
    opts = {"out-json": None, "out-md": None, "out-tex": None, "queue": str(ROOT / "scripts/kaggle/final_queue.json")}
    args = list(argv)
    for key in list(opts):
        if "--" + key in args:
            i = args.index("--" + key); opts[key] = args[i + 1]; del args[i:i + 2]
    if args:
        raise SystemExit(__doc__)
    t = build(json.loads(Path(opts["queue"]).read_text())["jobs"], SPLITS)
    md = markdown(t)
    for key, text in (("out-json", json.dumps(t, indent=1) + "\n"), ("out-md", md), ("out-tex", latex(t))):
        if opts[key]:
            Path(opts[key]).parent.mkdir(parents=True, exist_ok=True)
            Path(opts[key]).write_text(text)
    print(md, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
