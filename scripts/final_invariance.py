"""Invariance-pair table for the locked test from the collected final flags (CPU only, no model).

Usage: python scripts/final_invariance.py [--out-json f] [--out-md f] [--out-tex f] [--queue f]

Inputs are the files written by scripts/kaggle/collect_final.py: for every entry of
scripts/kaggle/final_queue.json and each of the six locked splits, <dev_dir>/flags_<split>.json (root
answers, main run) and <dev_dir>/flags_<split>_inv.json (relabelled / restyled copies of the same roots).

What is reported (paper/sections/analysis.tex "Invariance pairs", fixed before the main runs): per split,
arm and pair kind, the share of pairs with both answers right and the share whose two answers have the
same correctness, next to root-right and copy-right. Counts are summed over the two non-edit tasks and
over the three training seeds (every seed scores the same pairs, so the pooled rate is the mean of the
per-seed rates); the per-seed range of the both-right rate is kept. Descriptive only: no interval, no
reading rule and no reference arm, because the pre-registered decision uses the main table alone.

All-or-nothing like scripts/final_table.py: while any queue entry lacks any main or invariance flags file
the script stops, writes nothing and prints no number."""
import importlib.util, json, sys
from pathlib import Path
from trace_vr import analysis, arms as arms_mod, bootstrap, dataset

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ("pairs", "both_right", "root_right", "copy_right", "same_correctness")
RATES = COUNTS[1:]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


final_table = _load("final_table", "scripts/final_table.py")
SPLITS = final_table.SPLITS
SEEDS = tuple(final_table.seed_table.seeds.PREREGISTERED_SEEDS)


def missing(jobs, splits, root=ROOT):
    """Queue entries without every main and invariance flags file: [(job, [missing file names])]."""
    out = []
    for e in jobs:
        d = Path(root) / e["dev_dir"]
        names = ["final_summary.json"] + ["flags_%s%s.json" % (s, x) for _, s, _ in splits for x in ("", "_inv")]
        gone = [n for n in names if not (d / n).exists()]
        if gone:
            out.append((e["job"], gone))
    return out


def _payload(path, man, split, roots, sample_set):
    p = json.loads(Path(path).read_text())
    if (p.get("content_digest"), p.get("split"), p.get("eval_root_graphs"), p.get("sample_set", "main")) != (
            man["content_digest"], split, roots, sample_set):
        raise SystemExit("%s is not the %s %s evaluation on %d roots of this build" % (path, sample_set, split, roots))
    if p.get("missing_responses"):
        raise SystemExit("%s has %d missing responses" % (path, p["missing_responses"]))
    return p


def _rates(cell):
    return dict(cell, **{k + "_rate": cell[k] / cell["pairs"] for k in RATES})


def build(jobs, splits, root=ROOT):
    root = Path(root)
    gaps = missing(jobs, splits, root)
    if gaps:
        raise SystemExit("locked-test invariance table refused, %d of %d final runs not collected: %s"
                         % (len(gaps), len(jobs), ", ".join(j for j, _ in gaps)))
    seen = {}
    for e in jobs:
        seen.setdefault(e["arm"], []).append(e["seed"])
    bad = sorted(a for a, s in seen.items() if sorted(s) != sorted(SEEDS))
    if bad:
        raise SystemExit("every arm needs exactly the three pre-registered seeds; check: %s" % bad)
    labels = list(seen)
    tables = {}
    for manifest, split, roots in splits:
        man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
        main_rows = arms_mod.eval_samples(by_split[split], roots)
        inv_rows = arms_mod.invariance_samples(by_split[split], roots)
        per_arm = {}
        for e in jobs:
            d = root / e["dev_dir"]
            main = bootstrap.decode_flags(main_rows, _payload(d / ("flags_%s.json" % split), man, split, roots, "main"))
            inv = bootstrap.decode_flags(inv_rows, _payload(d / ("flags_%s_inv.json" % split), man, split, roots, "invariance"))
            per_arm.setdefault(e["arm"], {})[e["seed"]] = analysis.invariance_table(main_rows, main, inv_rows, inv)
        kinds = sorted({k for by_seed in per_arm.values() for t in by_seed.values() for k in t})
        rows = []
        for label in labels:
            by_seed, row = per_arm[label], {"label": label, "seeds": sorted(per_arm[label]), "kinds": {}}
            for kind in kinds:
                per_seed, per_task = {}, {}
                for s in sorted(by_seed):
                    tot = dict.fromkeys(COUNTS, 0)
                    for task, cell in by_seed[s].get(kind, {}).items():
                        acc = per_task.setdefault(task, dict.fromkeys(COUNTS, 0))
                        for k in COUNTS:
                            tot[k] += int(cell[k]); acc[k] += int(cell[k])
                    if not tot["pairs"]:
                        raise SystemExit("%s seed %s has no %s pairs on %s" % (label, s, kind, split))
                    per_seed[str(s)] = _rates(tot)
                if len({v["pairs"] for v in per_seed.values()}) != 1:
                    raise SystemExit("%s: seeds cover different %s pairs on %s" % (label, kind, split))
                pooled = _rates({k: sum(v[k] for v in per_seed.values()) for k in COUNTS})
                both = [v["both_right_rate"] for v in per_seed.values()]
                row["kinds"][kind] = {"pairs_per_seed": next(iter(per_seed.values()))["pairs"], "pooled": pooled,
                                      "both_right_seed_range": {"min": min(both), "max": max(both)},
                                      "per_seed": per_seed, "per_task": {t: _rates(c) for t, c in sorted(per_task.items())}}
            rows.append(row)
        tables[split] = {"manifest": manifest, "content_digest": man["content_digest"], "root_graphs": roots,
                         "invariance_copies": len(inv_rows), "kinds": kinds, "arms": rows}
    return {"jobs": [e["job"] for e in jobs], "splits": [s for _, s, _ in splits], "labels": labels, "tables": tables}


def _short(kind):
    return kind[len("invariant_"):] if kind.startswith("invariant_") else kind


def markdown(t):
    labels = t["labels"]
    lines = ["Locked test, invariance pairs (root question repeated on a relabelled / restyled copy of the same graph, same "
             "answer). Per arm: both answers right / same correctness, as shares of pairs, counts summed over the two "
             "non-edit tasks and three seeds (%d runs). Descriptive, no intervals." % len(t["jobs"]), "",
             "| Split | Pair kind | Pairs | " + " | ".join(labels) + " |", "|---|---|---|" + "---|" * len(labels)]
    for s in t["splits"]:
        tab = t["tables"][s]
        rows = {a["label"]: a for a in tab["arms"]}
        for kind in tab["kinds"]:
            cells = ["%.3f / %.3f" % (rows[l]["kinds"][kind]["pooled"]["both_right_rate"],
                                      rows[l]["kinds"][kind]["pooled"]["same_correctness_rate"]) for l in labels]
            lines.append("| %s | %s | %d | %s |" % (s, _short(kind), rows[labels[0]]["kinds"][kind]["pairs_per_seed"], " | ".join(cells)))
    return "\n".join(lines) + "\n"


def latex(t):
    """Booktabs tabular (\\input into a table float): one row per split and pair kind; per arm both right / same correctness."""
    labels = t["labels"]
    esc = lambda x: str(x).replace("_", "\\_")
    out = ["% Generated by scripts/final_invariance.py; do not edit by hand.",
           "\\begin{tabular}{llr%s}" % ("c" * len(labels)), "\\toprule",
           "Split & Pair kind & Pairs & " + " & ".join(esc(l) for l in labels) + " \\\\", "\\midrule"]
    for s in t["splits"]:
        tab = t["tables"][s]
        rows = {a["label"]: a for a in tab["arms"]}
        for kind in tab["kinds"]:
            cells = ["%.3f / %.3f" % (rows[l]["kinds"][kind]["pooled"]["both_right_rate"],
                                      rows[l]["kinds"][kind]["pooled"]["same_correctness_rate"]) for l in labels]
            out.append("%s & %s & %d & %s \\\\" % (esc(s), esc(_short(kind)), rows[labels[0]]["kinds"][kind]["pairs_per_seed"],
                                                   " & ".join(cells)))
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
