"""Per-task detail table for the locked test from the collected final metrics (CPU only, no model).

Usage: python scripts/final_metrics.py [--out-json f] [--out-md f] [--out-tex f] [--queue f]

Inputs are the files written by scripts/kaggle/collect_final.py: for every entry of
scripts/kaggle/final_queue.json, <dev_dir>/final_summary.json (the kernel's metrics files, copied) and
<dev_dir>/flags_<split>.json for the six locked splits (main samples).

What is reported (paper/sections/analysis.tex "Cells and task means" and "Perception or reasoning" (ii), fixed
before the main runs): per split, arm and task the strict accuracy, the format-failure rate and the lenient
accuracy; for repair also status accuracy, valid-edit rate and minimal-repair accuracy; responses cut at the
token limit; and for parse-and-solve arms the extraction quality (exact, edge precision / recall, direction
accuracy, node recall; strict parse). Every value is the mean over the three training seeds of the value in
that seed's metrics file (every seed scores the same samples); per-seed values are kept. Descriptive only: no
interval, no reading rule and no reference arm, because the pre-registered decision uses the main table
(scripts/final_table.py) alone. The intervals of the task means are in that table's JSON.

Consistency: every strict per-task accuracy taken from a metrics file must equal the accuracy recomputed from
the tracked flags file of the same evaluation, otherwise the script stops.

All-or-nothing like scripts/final_table.py: while any queue entry lacks final_summary.json or any main flags
file the script stops, writes nothing and prints no number."""
import importlib.util, json, sys
from pathlib import Path

from trace_vr import arms as arms_mod, bootstrap, dataset, metrics as metrics_mod, tasks as tasks_mod

ROOT = Path(__file__).resolve().parents[1]
TASK_FIELDS = ("accuracy", "format_failure_rate", "lenient_accuracy")
REPAIR_FIELDS = ("status_accuracy", "valid_edit_rate", "minimal_repair_accuracy")
EXTRACTION_FIELDS = ("exact", "format_ok", "edge_precision", "edge_recall", "direction_accuracy", "node_recall")
TOL = 1e-9


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


final_table = _load("final_table", "scripts/final_table.py")
SPLITS = final_table.SPLITS
SEEDS = tuple(final_table.seed_table.seeds.PREREGISTERED_SEEDS)


def _mean(values):
    return None if not values or any(v is None for v in values) else sum(values) / len(values)


def flag_accuracy(rows, payload):
    """Strict accuracy per task on the main samples, recomputed from a flags file: {task: (n, accuracy)}."""
    flags = bootstrap.decode_flags(rows, payload)
    out = {}
    for task in tasks_mod.ALL_TASKS:
        got = [flags[s["sample_id"]] for s in rows
               if s["task"] == task and s["metadata"]["image_variant"] in metrics_mod.MAIN_VARIANTS]
        if got:
            out[task] = (len(got), sum(got) / len(got))
    return out


def run_cells(m, split, roots, where):
    """One metrics file -> the cells this table shows. m is final_summary["metrics"][<split>]."""
    if m.get("split") != split or m.get("eval_root_graphs") != roots or m.get("sample_set", "main") != "main":
        raise SystemExit("%s is not the main %s evaluation on %d roots" % (where, split, roots))
    strict = m["metrics"]["per_task"]
    lenient = (m.get("metrics_lenient_parse") or {}).get("per_task")   # parse-and-solve: lenient parse of the extraction
    per_task = {}
    for task, cell in strict.items():
        row = {"n": cell["n"], "accuracy": cell.get("accuracy"), "format_failure_rate": cell.get("format_failure_rate"),
               "lenient_accuracy": lenient[task].get("accuracy") if lenient is not None else cell.get("lenient_accuracy")}
        if task == tasks_mod.REPAIR:
            row.update({k: cell.get(k) for k in REPAIR_FIELDS})
        per_task[task] = row
    out = {"per_task": per_task, "samples": m["metrics"]["samples"], "missing_responses": m["metrics"].get("missing_responses", 0),
           "truncated_at_limit": m.get("truncated_at_limit"), "max_new_tokens": m.get("max_new_tokens")}
    if "extraction_strict" in m:
        out["extraction"] = {k: m["extraction_strict"].get(k) for k in EXTRACTION_FIELDS}
        out["images"] = m.get("images")
    return out


def build(jobs, splits, root=ROOT):
    root = Path(root)
    gaps = final_table.missing(jobs, splits, root)
    if gaps:
        raise SystemExit("locked-test per-task table refused, %d of %d final runs not collected: %s"
                         % (len(gaps), len(jobs), ", ".join(j for j, _ in gaps)))
    seen = {}
    for e in jobs:
        seen.setdefault(e["arm"], []).append(e["seed"])
    bad = sorted(a for a, s in seen.items() if sorted(s) != sorted(SEEDS))
    if bad:
        raise SystemExit("every arm needs exactly the three pre-registered seeds; check: %s" % bad)
    summaries = {e["job"]: json.loads((root / e["dev_dir"] / "final_summary.json").read_text()) for e in jobs}
    tables = {}
    for manifest, split, roots in splits:
        man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
        rows = arms_mod.select_samples(by_split[split], roots, "main")
        per_arm = {}
        for e in jobs:
            where = "%s/final_summary.json [%s]" % (e["dev_dir"], split)
            m = summaries[e["job"]].get("metrics", {}).get(split)
            if m is None:
                raise SystemExit("%s has no metrics for this split" % where)
            cells = run_cells(m, split, roots, where)
            payload = json.loads((root / e["dev_dir"] / ("flags_%s.json" % split)).read_text())
            if payload.get("split") != split or payload.get("eval_root_graphs") != roots or payload.get("sample_set", "main") != "main":
                raise SystemExit("%s/flags_%s.json is not the main %s evaluation on %d roots" % (e["dev_dir"], split, split, roots))
            if payload.get("content_digest") != man["content_digest"] or m.get("content_digest") != man["content_digest"]:
                raise SystemExit("%s comes from a different dataset build" % where)
            direct = flag_accuracy(rows, payload)
            if set(direct) != set(cells["per_task"]):
                raise SystemExit("%s: tasks %s differ from the flags file's %s" % (where, sorted(cells["per_task"]), sorted(direct)))
            for task, (n, acc) in direct.items():
                c = cells["per_task"][task]
                if c["n"] != n or c["accuracy"] is None or abs(c["accuracy"] - acc) > TOL:
                    raise SystemExit("%s: %s accuracy in the metrics file differs from the flags file" % (where, task))
            per_arm.setdefault(e["arm"], {})[e["seed"]] = cells
        arms_out = []
        for label, by_seed in per_arm.items():
            order = sorted(by_seed)
            task_names = list(by_seed[order[0]]["per_task"])
            if any(list(by_seed[s]["per_task"]) != task_names for s in order):
                raise SystemExit("%s %s: seeds report different tasks" % (split, label))
            mean = {}
            for task in task_names:
                fields = TASK_FIELDS + (REPAIR_FIELDS if task == tasks_mod.REPAIR else ())
                ns = {by_seed[s]["per_task"][task]["n"] for s in order}
                if len(ns) != 1:
                    raise SystemExit("%s %s %s: seeds scored different sample counts" % (split, label, task))
                mean[task] = dict({k: _mean([by_seed[s]["per_task"][task].get(k) for s in order]) for k in fields}, n=ns.pop())
            row = {"label": label, "seeds": order, "per_seed": {str(s): by_seed[s] for s in order}, "mean": mean,
                   "truncated_at_limit": {str(s): by_seed[s]["truncated_at_limit"] for s in order},
                   "missing_responses": sum(by_seed[s]["missing_responses"] for s in order)}
            if all("extraction" in by_seed[s] for s in order):
                row["extraction_mean"] = {k: _mean([by_seed[s]["extraction"].get(k) for s in order]) for k in EXTRACTION_FIELDS}
            elif any("extraction" in by_seed[s] for s in order):
                raise SystemExit("%s %s: only some seeds report extraction quality" % (split, label))
            arms_out.append(row)
        tables[split] = {"manifest": manifest, "split": split, "eval_root_graphs": roots, "content_digest": man["content_digest"],
                         "arms": arms_out}
    return {"jobs": [e["job"] for e in jobs], "splits": [s for _, s, _ in splits], "labels": list(seen), "tables": tables}


def _fmt(x):
    return "n/a" if x is None else "%.3f" % x


def _task_names(t):
    out = []
    for s in t["splits"]:
        for a in t["tables"][s]["arms"]:
            out += [k for k in a["mean"] if k not in out]
    return out


def markdown(t):
    labels = t["labels"]
    lines = ["Locked test, per-task detail (%d runs, three seeds per arm). Every cell is the mean over the three seeds of the "
             "value in that seed's metrics file. Descriptive, no intervals. Cell: strict accuracy / lenient accuracy / "
             "format-failure rate." % len(t["jobs"]), "",
             "| Split | Task | Questions | " + " | ".join(labels) + " |", "|---|---|---|" + "---|" * len(labels)]
    for s in t["splits"]:
        rows = {a["label"]: a for a in t["tables"][s]["arms"]}
        for task in _task_names(t):
            if not any(task in rows[l]["mean"] for l in labels):
                continue
            n = next(rows[l]["mean"][task]["n"] for l in labels if task in rows[l]["mean"])
            cells = ["n/a" if task not in rows[l]["mean"] else " / ".join(_fmt(rows[l]["mean"][task][k]) for k in TASK_FIELDS)
                     for l in labels]
            lines.append("| %s | %s | %d | %s |" % (s, task, n, " | ".join(cells)))
    lines += ["", "Repair detail. Cell: status accuracy / valid-edit rate / minimal-repair accuracy (mean over seeds).", "",
              "| Split | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for s in t["splits"]:
        rows = {a["label"]: a for a in t["tables"][s]["arms"]}
        cells = ["n/a" if tasks_mod.REPAIR not in rows[l]["mean"]
                 else " / ".join(_fmt(rows[l]["mean"][tasks_mod.REPAIR][k]) for k in REPAIR_FIELDS) for l in labels]
        lines.append("| %s | %s |" % (s, " | ".join(cells)))
    lines += ["", "Responses cut at the token limit, per seed (%s)." % ", ".join(str(x) for x in sorted(SEEDS)), "",
              "| Split | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for s in t["splits"]:
        rows = {a["label"]: a for a in t["tables"][s]["arms"]}
        cells = [", ".join("n/a" if v is None else str(v) for _, v in sorted(rows[l]["truncated_at_limit"].items())) for l in labels]
        lines.append("| %s | %s |" % (s, " | ".join(cells)))
    ext = [l for l in labels if all("extraction_mean" in a for s in t["splits"] for a in t["tables"][s]["arms"] if a["label"] == l)]
    for l in ext:
        lines += ["", "Extraction quality of arm %s (strict parse, mean over seeds)." % l, "",
                  "| Split | " + " | ".join(k.replace("_", " ") for k in EXTRACTION_FIELDS) + " |",
                  "|---|" + "---|" * len(EXTRACTION_FIELDS)]
        for s in t["splits"]:
            a = next(a for a in t["tables"][s]["arms"] if a["label"] == l)
            lines.append("| %s | %s |" % (s, " | ".join(_fmt(a["extraction_mean"][k]) for k in EXTRACTION_FIELDS)))
    return "\n".join(lines) + "\n"


def latex(t):
    """Booktabs tabular (\\input into a table float): one row per split and task; per arm the strict accuracy with the
    format-failure rate in small type, both means over seeds."""
    labels = t["labels"]
    esc = lambda x: str(x).replace("_", "\\_")
    out = ["% Generated by scripts/final_metrics.py; do not edit by hand.",
           "\\begin{tabular}{llr%s}" % ("c" * len(labels)), "\\toprule",
           "Split & Task & Questions & " + " & ".join(esc(l) for l in labels) + " \\\\", "\\midrule"]
    for s in t["splits"]:
        rows = {a["label"]: a for a in t["tables"][s]["arms"]}
        for task in _task_names(t):
            if not any(task in rows[l]["mean"] for l in labels):
                continue
            n = next(rows[l]["mean"][task]["n"] for l in labels if task in rows[l]["mean"])
            cells = ["n/a" if task not in rows[l]["mean"] else "%s {\\scriptsize(%s)}"
                     % (_fmt(rows[l]["mean"][task]["accuracy"]), _fmt(rows[l]["mean"][task]["format_failure_rate"])) for l in labels]
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
