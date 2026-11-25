"""Seed-pooled locked-test table from the collected final flags (CPU only, no model).

Usage: python scripts/final_table.py [--out-json f] [--out-md f] [--out-tex f] [--boot 2000] [--queue f] [--reference-file f]

Inputs are the files written by scripts/kaggle/collect_final.py: for every entry of
scripts/kaggle/final_queue.json, <dev_dir>/final_summary.json and <dev_dir>/flags_<split>.json for the six
locked splits (main samples). The reference arm is read from reports/pilot/reference_baseline.json (chosen on
dev before the test was opened); it is not an option of this script.

All-or-nothing (DECISIONS 2026-11-13T22:15Z): pooled test numbers are reported only once all three seed rounds
are in, so the script stops, writes nothing and prints no number while any queue entry is missing or incomplete.
Pooling, intervals and the reading rule are those of the dev seed table (src/trace_vr/seeds.py, scripts/seed_table.py),
applied per split. Invariance-pair scores (flags_<split>_inv.json) are not part of this table."""
import importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACRO = "non_edit_macro_accuracy"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


seed_table = _load("seed_table", "scripts/seed_table.py")
SPLITS = _load("collect_final", "scripts/kaggle/collect_final.py").SPLITS


def reference(path):
    rec = json.loads(Path(path).read_text())
    if not str(rec.get("split", "")).startswith("dev"):
        raise SystemExit("%s was not chosen on dev" % path)
    return rec["chosen"]


def missing(jobs, splits, root=ROOT):
    """Queue entries whose final output is not fully collected: [(job, [missing file names])]."""
    out = []
    for e in jobs:
        d = Path(root) / e["dev_dir"]
        names = ["final_summary.json"] + ["flags_%s.json" % s for _, s, _ in splits]
        gone = [n for n in names if not (d / n).exists()]
        if gone:
            out.append((e["job"], gone))
    return out


def build(jobs, splits, ref, n_boot=2000, root=ROOT):
    root = Path(root)
    gaps = missing(jobs, splits, root)
    if gaps:
        raise SystemExit("locked-test table refused, %d of %d final runs not collected: %s"
                         % (len(gaps), len(jobs), ", ".join(j for j, _ in gaps)))
    arms = {}
    for e in jobs:
        arms.setdefault(e["arm"], set()).add(e["seed"])
    full = set(seed_table.seeds.PREREGISTERED_SEEDS)
    bad = sorted(a for a, s in arms.items() if s != full)
    if bad or ref not in arms:
        raise SystemExit("every arm needs exactly the three pre-registered seeds and the reference %s; check: %s" % (ref, bad))
    tables = {}
    for manifest, split, roots in splits:
        entries = [(e["arm"], e["seed"], root / e["dev_dir"] / ("flags_%s.json" % split)) for e in jobs]
        for _, _, p in entries:
            got = json.loads(p.read_text())
            if got.get("split") != split or got.get("eval_root_graphs") != roots or got.get("sample_set", "main") != "main":
                raise SystemExit("%s is not the main %s evaluation on %d roots" % (p, split, roots))
        t = seed_table.build(str(ROOT / "manifests" / (manifest + ".json")), entries, ref, n_boot)
        for f in t["files"]:
            f["file"] = str(Path(f["file"]).relative_to(root)) if Path(f["file"]).is_relative_to(root) else f["file"]
        tables[split] = dict(t, manifest=manifest)
    return {"reference": ref, "n_boot": n_boot, "jobs": [e["job"] for e in jobs],
            "splits": [s for _, s, _ in splits], "tables": tables}


def markdown(t):
    ref, splits = t["reference"], t["splits"]
    labels = [a["label"] for a in t["tables"][splits[0]]["arms"]]
    rows = {s: {a["label"]: a for a in t["tables"][s]["arms"]} for s in splits}
    lines = ["Locked test, final evaluation (run once per trained run; %d runs, three seeds per arm). Non-edit macro "
             "accuracy pooled over seeds, 95%% interval over root graphs (%d replicates). Rules: `src/trace_vr/seeds.py`."
             % (len(t["jobs"]), t["n_boot"]), "",
             "| Split | Roots | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for s in splits:
        cells = ["%.3f [%.3f, %.3f]" % tuple(rows[s][l]["pooled"][MACRO][k] for k in ("point", "lo", "hi")) for l in labels]
        lines.append("| %s | %d | %s |" % (s, t["tables"][s]["root_graphs"], " | ".join(cells)))
    others = [l for l in labels if l != ref]
    lines += ["", "Pooled difference to %s (reading by the pre-registered rule):" % ref, "",
              "| Split | " + " | ".join("%s - %s" % (l, ref) for l in others) + " |", "|---|" + "---|" * len(others)]
    for s in splits:
        cells = []
        for l in others:
            d = rows[s][l]["vs_ref"][MACRO]
            cells.append("%+.3f [%+.3f, %+.3f] %s" % (d["point"], d["lo"], d["hi"], rows[s][l]["reading"]))
        lines.append("| %s | %s |" % (s, " | ".join(cells)))
    return "\n".join(lines) + "\n"


def latex(t):
    """Booktabs tabular for the paper (\\input into a table float): one row per locked split, pooled non-edit macro
    with its interval per arm, then the pooled difference to the reference with its interval for every other arm."""
    ref, splits = t["reference"], t["splits"]
    labels = [a["label"] for a in t["tables"][splits[0]]["arms"]]
    others = [l for l in labels if l != ref]
    esc = lambda x: str(x).replace("_", "\\_")
    out = ["% Generated by scripts/final_table.py; do not edit by hand.",
           "\\begin{tabular}{lr%s}" % ("c" * (len(labels) + len(others))), "\\toprule",
           "Split & Roots & " + " & ".join([esc(l) for l in labels] + ["%s $-$ %s" % (esc(l), esc(ref)) for l in others]) + " \\\\",
           "\\midrule"]
    for s in splits:
        rows = {a["label"]: a for a in t["tables"][s]["arms"]}
        cells = []
        for l in labels:
            p = rows[l]["pooled"][MACRO]
            cells.append("%.3f {\\scriptsize[%.3f, %.3f]}" % (p["point"], p["lo"], p["hi"]))
        for l in others:
            d = rows[l]["vs_ref"][MACRO]
            cells.append("$%+.3f$ {\\scriptsize[$%+.3f$, $%+.3f$]}" % (d["point"], d["lo"], d["hi"]))
        out.append("%s & %d & %s \\\\" % (esc(s), t["tables"][s]["root_graphs"], " & ".join(cells)))
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def main(argv):
    opts = {"out-json": None, "out-md": None, "out-tex": None, "boot": 2000, "queue": str(ROOT / "scripts/kaggle/final_queue.json"),
            "reference-file": str(ROOT / "reports/pilot/reference_baseline.json")}
    args = list(argv)
    for key in list(opts):
        if "--" + key in args:
            i = args.index("--" + key); opts[key] = args[i + 1]; del args[i:i + 2]
    if args:
        raise SystemExit(__doc__)
    jobs = json.loads(Path(opts["queue"]).read_text())["jobs"]
    t = build(jobs, SPLITS, reference(opts["reference-file"]), int(opts["boot"]))
    md = markdown(t)
    for key, text in (("out-json", json.dumps(t, indent=1) + "\n"), ("out-md", md), ("out-tex", latex(t))):
        if opts[key]:
            Path(opts[key]).parent.mkdir(parents=True, exist_ok=True)
            Path(opts[key]).write_text(text)
    print(md, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
