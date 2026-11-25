"""Slice breakdown of the locked test from the collected final flags (CPU only, no model).

Usage: python scripts/final_slices.py [--out-json f] [--out-md f] [--out-tex f] [--queue f]

Inputs are the files written by scripts/kaggle/collect_final.py: for every entry of
scripts/kaggle/final_queue.json and each of the six locked splits, <dev_dir>/flags_<split>.json (main samples).

What is reported (paper/sections/analysis.tex "Slices", fixed before the main runs): accuracy by true answer,
image variant (root or edited), graph family, size bucket (<= 8, 9-12, >= 13 nodes), layout and style, per task.
This is analysis.breakdown (the function behind scripts/breakdown.py on dev) applied unchanged to each locked
split and run; the facets are analysis.FACETS and are read from labels and drawing metadata, never from model
output. Per split, task, facet value and arm: right answers and questions summed over the three training seeds
(the same questions in every seed), with the per-seed accuracy kept. Every facet and every value is printed, so
no slice is picked after the fact. Descriptive only: no interval, no reading rule and no reference arm, because
the pre-registered decision uses the main table (scripts/final_table.py) alone.

All-or-nothing like scripts/final_table.py: while any queue entry lacks any main flags file the script stops,
writes nothing and prints no number."""
import importlib.util, json, sys
from pathlib import Path
from trace_vr import analysis, arms, bootstrap, dataset

ROOT = Path(__file__).resolve().parents[1]
FACETS = tuple(analysis.FACETS)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


final_table = _load("final_table", "scripts/final_table.py")
SPLITS = final_table.SPLITS
SEEDS = tuple(final_table.seed_table.seeds.PREREGISTERED_SEEDS)


def build(jobs, splits, root=ROOT):
    root = Path(root)
    gaps = final_table.missing(jobs, splits, root)
    if gaps:
        raise SystemExit("locked-test slice breakdown refused, %d of %d final runs not collected: %s"
                         % (len(gaps), len(jobs), ", ".join(j for j, _ in gaps)))
    seen = {}
    for e in jobs:
        seen.setdefault(e["arm"], []).append(e["seed"])
    bad = sorted(a for a, s in seen.items() if sorted(s) != sorted(SEEDS))
    if bad:
        raise SystemExit("every arm needs exactly the three pre-registered seeds; check: %s" % bad)
    tables = {}
    for manifest, split, roots in splits:
        man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
        rows = arms.select_samples(by_split[split], roots, "main")
        cells = {}        # task -> facet -> value -> {"n": questions per seed, "arms": {arm: {...}}}
        for e in jobs:
            p = root / e["dev_dir"] / ("flags_%s.json" % split)
            got = json.loads(p.read_text())
            if got.get("split") != split or got.get("eval_root_graphs") != roots or got.get("sample_set", "main") != "main":
                raise SystemExit("%s is not the main %s evaluation on %d roots" % (p, split, roots))
            if got.get("content_digest") != man["content_digest"]:
                raise SystemExit("%s comes from a different dataset build" % p)
            bd = analysis.breakdown(rows, bootstrap.decode_flags(rows, got))
            for task, facets in bd.items():
                for facet, values in facets.items():
                    for value, c in values.items():
                        cell = cells.setdefault(task, {}).setdefault(facet, {}).setdefault(value, {"n": c["n"], "arms": {}})
                        if cell["n"] != c["n"]:   # same questions for every run, so the slice sizes cannot differ
                            raise SystemExit("%s: slice sizes differ between runs" % p)
                        a = cell["arms"].setdefault(e["arm"], {"right": 0, "n": 0, "per_seed": {}})
                        a["right"] += c["right"]
                        a["n"] += c["n"]
                        a["per_seed"][str(e["seed"])] = c["accuracy"]
        for facets in cells.values():
            for values in facets.values():
                for cell in values.values():
                    for a in cell["arms"].values():
                        a["accuracy"] = a["right"] / a["n"]
        tables[split] = {"manifest": manifest, "content_digest": man["content_digest"], "roots": roots, "tasks": cells}
    return {"facets": list(FACETS), "jobs": [e["job"] for e in jobs], "splits": [s for _, s, _ in splits],
            "labels": list(seen), "seeds": sorted(SEEDS), "tables": tables}


def _rows(t):
    """(split, task, facet, value, n, {arm: accuracy}) in a fixed order: splits as queued, then sorted."""
    for s in t["splits"]:
        tasks_ = t["tables"][s]["tasks"]
        for task in sorted(tasks_):
            for facet in t["facets"]:
                for value in sorted(tasks_[task].get(facet, {})):
                    cell = tasks_[task][facet][value]
                    yield s, task, facet, value, cell["n"], {l: cell["arms"][l]["accuracy"] for l in t["labels"]}


def markdown(t):
    labels = t["labels"]
    lines = ["Locked test, slice breakdown: accuracy per task by %s. Per arm: right answers over questions, summed over "
             "three seeds (%d runs); n = questions per seed. Descriptive, no intervals; every facet value is listed."
             % (", ".join(t["facets"]), len(t["jobs"])), "",
             "| Split | Task | Facet | Value | n | " + " | ".join(labels) + " |",
             "|---|---|---|---|---|" + "---|" * len(labels)]
    for s, task, facet, value, n, acc in _rows(t):
        lines.append("| %s | %s | %s | %s | %d | %s |" % (s, task, facet, value, n, " | ".join("%.3f" % acc[l] for l in labels)))
    return "\n".join(lines) + "\n"


def latex(t):
    """Booktabs longtable body (needs \\usepackage{longtable}): one row per split, task, facet value."""
    labels = t["labels"]
    esc = lambda x: str(x).replace("_", "\\_").replace("<=", "$\\le$").replace(">=", "$\\ge$")
    out = ["% Generated by scripts/final_slices.py; do not edit by hand.",
           "\\begin{longtable}{llllr%s}" % ("c" * len(labels)), "\\toprule",
           "Split & Task & Facet & Value & $n$ & " + " & ".join(esc(l) for l in labels) + " \\\\", "\\midrule", "\\endhead"]
    for s, task, facet, value, n, acc in _rows(t):
        out.append("%s & %s & %s & %s & %d & %s \\\\" % (esc(s), esc(task), esc(facet), esc(value), n,
                                                       " & ".join("%.3f" % acc[l] for l in labels)))
    out += ["\\bottomrule", "\\end{longtable}"]
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
