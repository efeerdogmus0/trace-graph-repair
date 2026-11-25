"""Twin-pair and answer-bias table for the locked test from the collected final flags (CPU only, no model).

Usage: python scripts/final_pairs.py [--out-json f] [--out-md f] [--out-tex f] [--queue f]

Inputs are the files written by scripts/kaggle/collect_final.py: for every entry of
scripts/kaggle/final_queue.json and each of the six locked splits, <dev_dir>/flags_<split>.json (main samples;
answer-flipping twins are main samples).

What is reported (paper/sections/analysis.tex "Twin pairs and answer bias", fixed before the main runs): per
split, arm and task (repair included), how many twin pairs have both images right / one right / none right, and for the
yes/no tasks how often the model said true next to how often true was correct. Counts are summed over the three
training seeds (every seed scores the same pairs, so the summed rate is the mean of the per-seed rates); the
per-seed both-right counts are kept. This is scripts/seed_pairs.py applied per locked split. Descriptive only:
no interval, no reading rule and no reference arm, because the pre-registered decision uses the main table
(scripts/final_table.py) alone.

All-or-nothing like scripts/final_table.py: while any queue entry lacks any main flags file the script stops,
writes nothing and prints no number."""
import importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


final_table = _load("final_table", "scripts/final_table.py")
seed_pairs = _load("seed_pairs", "scripts/seed_pairs.py")
SPLITS = final_table.SPLITS
SEEDS = tuple(final_table.seed_table.seeds.PREREGISTERED_SEEDS)


def build(jobs, splits, root=ROOT):
    root = Path(root)
    gaps = final_table.missing(jobs, splits, root)
    if gaps:
        raise SystemExit("locked-test twin-pair table refused, %d of %d final runs not collected: %s"
                         % (len(gaps), len(jobs), ", ".join(j for j, _ in gaps)))
    seen = {}
    for e in jobs:
        seen.setdefault(e["arm"], []).append(e["seed"])
    bad = sorted(a for a, s in seen.items() if sorted(s) != sorted(SEEDS))
    if bad:
        raise SystemExit("every arm needs exactly the three pre-registered seeds; check: %s" % bad)
    tables = {}
    for manifest, split, roots in splits:
        entries = [(e["arm"], e["seed"], root / e["dev_dir"] / ("flags_%s.json" % split)) for e in jobs]
        for _, _, p in entries:
            got = json.loads(p.read_text())
            if got.get("split") != split or got.get("eval_root_graphs") != roots or got.get("sample_set", "main") != "main":
                raise SystemExit("%s is not the main %s evaluation on %d roots" % (p, split, roots))
        t = seed_pairs.build(str(ROOT / "manifests" / (manifest + ".json")), entries)
        for f in t["files"]:
            f["file"] = str(Path(f["file"]).relative_to(root)) if Path(f["file"]).is_relative_to(root) else f["file"]
        tables[split] = dict(t, manifest=manifest)
    return {"jobs": [e["job"] for e in jobs], "splits": [s for _, s, _ in splits], "labels": list(seen), "tables": tables}


def _tasks(tab, key):
    out = []
    for a in tab["arms"]:
        out += [t for t in a["summed"][key] if t not in out]
    return out


def _pair_cell(arm, task):
    tot = arm["summed"]["pairs"].get(task)
    return "n/a" if tot is None else "%d of %d (%.3f)" % (tot["both_right"], tot["pairs"], tot["both_right_rate"])


def _bias_cell(arm, task):
    tot = arm["summed"]["answer_bias"].get(task)
    return "n/a" if tot is None or tot["said_true_rate"] is None else "%.3f" % tot["said_true_rate"]


def markdown(t):
    labels = t["labels"]
    lines = ["Locked test, answer-flipping twin pairs (two images that differ by one edge and have opposite answers). "
             "Per arm: pairs with both images right, counts summed over three seeds (%d runs). Descriptive, no intervals."
             % len(t["jobs"]), "",
             "| Split | Task | Pairs per seed | " + " | ".join(labels) + " |", "|---|---|---|" + "---|" * len(labels)]
    for s in t["splits"]:
        tab = t["tables"][s]
        rows = {a["label"]: a for a in tab["arms"]}
        for task in _tasks(tab, "pairs"):
            n = rows[labels[0]]["per_seed"][str(rows[labels[0]]["seeds"][0])]["pairs"][task]["pairs"]
            lines.append("| %s | %s | %d | %s |" % (s, task, n, " | ".join(_pair_cell(rows[l], task) for l in labels)))
    lines += ["", "Share of answered yes/no questions answered true, summed over seeds (last column: share where true is correct).",
              "", "| Split | Task | " + " | ".join(labels) + " | True is correct |", "|---|---|" + "---|" * (len(labels) + 1)]
    for s in t["splits"]:
        tab = t["tables"][s]
        rows = {a["label"]: a for a in tab["arms"]}
        for task in _tasks(tab, "answer_bias"):
            truth = rows[labels[0]]["summed"]["answer_bias"][task]["truth_true_rate"]
            lines.append("| %s | %s | %s | %.3f |" % (s, task, " | ".join(_bias_cell(rows[l], task) for l in labels), truth))
    return "\n".join(lines) + "\n"


def latex(t):
    """Booktabs tabular (\\input into a table float): one row per split and task; per arm the both-right rate of
    twin pairs summed over seeds."""
    labels = t["labels"]
    esc = lambda x: str(x).replace("_", "\\_")
    out = ["% Generated by scripts/final_pairs.py; do not edit by hand.",
           "\\begin{tabular}{llr%s}" % ("c" * len(labels)), "\\toprule",
           "Split & Task & Pairs & " + " & ".join(esc(l) for l in labels) + " \\\\", "\\midrule"]
    for s in t["splits"]:
        tab = t["tables"][s]
        rows = {a["label"]: a for a in tab["arms"]}
        for task in _tasks(tab, "pairs"):
            n = rows[labels[0]]["per_seed"][str(rows[labels[0]]["seeds"][0])]["pairs"][task]["pairs"]
            cells = ["n/a" if task not in rows[l]["summed"]["pairs"] else "%.3f" % rows[l]["summed"]["pairs"][task]["both_right_rate"]
                     for l in labels]
            out.append("%s & %s & %d & %s \\\\" % (esc(s), esc(task), n, " & ".join(cells)))
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
