"""Frozen-base floor (arm A, jobF13) on the locked test, from the collected final files (CPU only, no model).

Usage: python scripts/final_floor.py [--out-json f] [--out-md f] [--out-tex f] [--boot 2000] [--queue f]

Arm A is a reported floor, not a compared arm (paper/sections/setup.tex, DECISIONS 2026-11-14T12:15Z): one
untrained run with the prompt chosen on dev (reports/pilot/base_prompt.json). Its files are those written by
scripts/kaggle/collect_final.py for the queue's "base" entry: <dev_dir>/final_summary.json and
<dev_dir>/flags_<split>.json for the six locked splits (main samples).

What is reported per locked split: the non-edit macro accuracy and each task's accuracy with a 95% interval over
root graphs (same resampling stream as the seed tables, src/trace_vr/seeds.py, one run, no seed pooling), and from
the metrics file the lenient accuracy, the format-failure rate, repair status / valid-edit / minimal-repair, and
responses cut at the token limit. No difference to any arm, no reading rule: the pre-registered decision uses
scripts/final_table.py alone.

Checks: the metrics files must report no adapter and the dev-chosen prompt variant and token limit; every strict
per-task accuracy in a metrics file must equal the one recomputed from the flags file; no missing responses.

All-or-nothing: the floor is opened together with the main table, so the script stops, writes nothing and prints no
number while the base run or any of the twelve queue entries is not fully collected."""
import importlib.util, json, sys
from pathlib import Path

from trace_vr import arms as arms_mod, bootstrap, dataset, seeds, tasks as tasks_mod

ROOT = Path(__file__).resolve().parents[1]
MACRO = "non_edit_macro_accuracy"
TOL = 1e-9


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


final_metrics = _load("final_metrics", "scripts/final_metrics.py")
final_table = final_metrics.final_table
collect_final = _load("collect_final", "scripts/kaggle/collect_final.py")
SPLITS = final_table.SPLITS


def build(base, jobs, splits, n_boot=2000, root=ROOT):
    """base: collect_final.base_entry(); jobs: the twelve queue entries (only checked for completeness)."""
    root = Path(root)
    if base is None:
        raise SystemExit("the queue has no base entry")
    gaps = final_table.missing(list(jobs) + [base], splits, root)
    if gaps:
        raise SystemExit("frozen-base floor refused, %d final runs not collected: %s" % (len(gaps), ", ".join(j for j, _ in gaps)))
    d = root / base["dev_dir"]
    summary = json.loads((d / "final_summary.json").read_text())
    tables = {}
    for manifest, split, roots in splits:
        where = "%s/final_summary.json [%s]" % (base["dev_dir"], split)
        m = summary.get("metrics", {}).get(split)
        if m is None:
            raise SystemExit("%s has no metrics for this split" % where)
        bad = sorted(k for k, v in base.get("expect", {}).items() if m.get(k) != v)
        if bad:
            raise SystemExit("%s: %s differ from the dev choice %s" % (where, bad, base.get("expect")))
        cells = final_metrics.run_cells(m, split, roots, where)
        man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
        rows = arms_mod.select_samples(by_split[split], roots, "main")
        payload = json.loads((d / ("flags_%s.json" % split)).read_text())
        if payload.get("split") != split or payload.get("eval_root_graphs") != roots or payload.get("sample_set", "main") != "main":
            raise SystemExit("%s/flags_%s.json is not the main %s evaluation on %d roots" % (base["dev_dir"], split, split, roots))
        if payload.get("content_digest") != man["content_digest"] or m.get("content_digest") != man["content_digest"]:
            raise SystemExit("%s comes from a different dataset build" % where)
        if payload.get("missing_responses") or cells["missing_responses"]:
            raise SystemExit("%s has missing responses" % where)
        direct = final_metrics.flag_accuracy(rows, payload)
        if set(direct) != set(cells["per_task"]):
            raise SystemExit("%s: tasks %s differ from the flags file's %s" % (where, sorted(cells["per_task"]), sorted(direct)))
        for task, (n, acc) in direct.items():
            c = cells["per_task"][task]
            if c["n"] != n or c["accuracy"] is None or abs(c["accuracy"] - acc) > TOL:
                raise SystemExit("%s: %s accuracy in the metrics file differs from the flags file" % (where, task))
        label = base["arm"]
        boot = seeds.pooled_bootstrap(rows, {label: {0: bootstrap.decode_flags(rows, payload)}}, label, n_boot=n_boot)
        interval = boot["arms"][0]["pooled"]
        tables[split] = {"manifest": manifest, "split": split, "eval_root_graphs": roots, "root_graphs": boot["root_graphs"],
                         "content_digest": man["content_digest"], "interval": interval, "per_task": cells["per_task"],
                         "truncated_at_limit": cells["truncated_at_limit"], "max_new_tokens": cells["max_new_tokens"],
                         "predictions_sha256": payload.get("predictions_sha256")}
    return {"job": base["job"], "arm": base["arm"], "run": base.get("run"), "variant": base.get("variant"),
            "limit": base.get("limit"), "n_boot": n_boot, "splits": [s for _, s, _ in splits], "tables": tables}


def _fmt(x):
    return "n/a" if x is None else "%.3f" % x


def _ci(e):
    return "n/a" if e is None else "%.3f [%.3f, %.3f]" % (e["point"], e["lo"], e["hi"])


def markdown(t):
    tasks = [k for k in t["tables"][t["splits"][0]]["interval"] if k != MACRO]
    lines = ["Frozen base (arm %s, %s): a reported floor, not a compared arm. Prompt variant `%s`, %s new tokens, no adapter, "
             "one run. Accuracy with a 95%% interval over root graphs (%d replicates)."
             % (t["arm"], t["job"], t["variant"], t["limit"], t["n_boot"]), "",
             "| Split | Roots | Non-edit macro | " + " | ".join(tasks) + " |", "|---|---|---|" + "---|" * len(tasks)]
    for s in t["splits"]:
        x = t["tables"][s]
        lines.append("| %s | %d | %s | %s |" % (s, x["root_graphs"], _ci(x["interval"][MACRO]),
                                               " | ".join(_ci(x["interval"].get(k)) for k in tasks)))
    lines += ["", "Per task from the metrics file. Cell: strict / lenient / format-failure rate; responses cut at the limit.", "",
              "| Split | " + " | ".join(tasks) + " | Truncated |", "|---|" + "---|" * (len(tasks) + 1)]
    for s in t["splits"]:
        x = t["tables"][s]
        cells = ["n/a" if k not in x["per_task"] else " / ".join(_fmt(x["per_task"][k][f]) for f in final_metrics.TASK_FIELDS)
                 for k in tasks]
        lines.append("| %s | %s | %s |" % (s, " | ".join(cells), "n/a" if x["truncated_at_limit"] is None else x["truncated_at_limit"]))
    lines += ["", "Repair detail. Cell: status accuracy / valid-edit rate / minimal-repair accuracy.", "", "| Split | Repair |", "|---|---|"]
    for s in t["splits"]:
        r = t["tables"][s]["per_task"].get(tasks_mod.REPAIR)
        lines.append("| %s | %s |" % (s, "n/a" if r is None else " / ".join(_fmt(r.get(k)) for k in final_metrics.REPAIR_FIELDS)))
    return "\n".join(lines) + "\n"


def latex(t):
    """Booktabs tabular: one row per locked split, non-edit macro with its interval, then each task's accuracy."""
    tasks = [k for k in t["tables"][t["splits"][0]]["interval"] if k != MACRO]
    esc = lambda x: str(x).replace("_", "\\_")
    out = ["% Generated by scripts/final_floor.py; do not edit by hand.",
           "\\begin{tabular}{lrc%s}" % ("c" * len(tasks)), "\\toprule",
           "Split & Roots & Non-edit macro & " + " & ".join(esc(k) for k in tasks) + " \\\\", "\\midrule"]
    for s in t["splits"]:
        x = t["tables"][s]
        p = x["interval"][MACRO]
        cells = ["%.3f {\\scriptsize[%.3f, %.3f]}" % (p["point"], p["lo"], p["hi"])]
        cells += [_fmt((x["interval"].get(k) or {}).get("point")) for k in tasks]
        out.append("%s & %d & %s \\\\" % (esc(s), x["root_graphs"], " & ".join(cells)))
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def main(argv):
    opts = {"out-json": None, "out-md": None, "out-tex": None, "boot": 2000, "queue": str(ROOT / "scripts/kaggle/final_queue.json")}
    args = list(argv)
    for key in list(opts):
        if "--" + key in args:
            i = args.index("--" + key); opts[key] = args[i + 1]; del args[i:i + 2]
    if args:
        raise SystemExit(__doc__)
    jobs = json.loads(Path(opts["queue"]).read_text())["jobs"]
    t = build(collect_final.base_entry(Path(opts["queue"])), jobs, SPLITS, int(opts["boot"]))
    md = markdown(t)
    for key, text in (("out-json", json.dumps(t, indent=1) + "\n"), ("out-md", md), ("out-tex", latex(t))):
        if opts[key]:
            Path(opts[key]).parent.mkdir(parents=True, exist_ok=True)
            Path(opts[key]).write_text(text)
    print(md, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
