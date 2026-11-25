"""The pre-registered primary statistic on the locked test, seed-pooled (CPU only, no model).

Usage: python scripts/final_primary.py [--out-json f] [--out-md f] [--out-tex f] [--boot 2000] [--queue f] [--reference-file f]

Protocol section 5 / paper Section "Statistic and interpretation rules": the primary statistic is the unweighted mean of
six accuracies, {reachability, critical edge} x {test_id, test_style_ood, test_struct_ood}, on main samples. Only this
statistic decides the hypothesis. final_table.py reports each locked split on its own; this script computes the six-cell
mean itself, which no other final script did (found 2026-11-15T06:10Z, before any locked-test number was read).

Fixed here, before any test output is read (DECISIONS 2026-11-15T06:10Z):
- per arm and seed: cell = accuracy of one task on one split (samples of that split, main images); primary = mean of
  the six cells; task mean = mean of that task's three cells. Pooled over seeds = mean of the per-seed values.
- interval: stratified paired bootstrap; in every replicate each of the three splits resamples its own root graphs
  (they are disjoint sets), with stream Rng("primary-v1", split, roots, n_boot); every arm and seed is scored on the
  same draw. 95% percentile interval.
- reading of an arm against the reference (from reports/pilot/reference_baseline.json, chosen on dev): the rule of
  src/trace_vr/seeds.py (positive_clear / positive / negative / inconclusive) applied to the primary statistic.
- secondary flags, reported, never used to upgrade a reading: `task_drop` = a task-mean pooled point more than 0.02
  below the reference ("general improvement" ruled out); `splits_same_sign` = the three per-split pooled points of the
  difference all have the sign of the pooled primary point (the "not driven by one style" check).
- E - D is reported with the same interval (protocol: "if E beats D, that is reported").
All-or-nothing like final_table.py: refuses, prints no number and writes nothing while any final run is uncollected."""
import importlib.util, json, sys
from pathlib import Path
from trace_vr import arms as arms_mod, bootstrap, dataset, seeds, tasks
from trace_vr.rng import Rng

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SPLITS = ("test_id", "test_style_ood", "test_struct_ood")
TASKS = tuple(tasks.MAIN_NON_EDIT_TASKS)
PRIMARY = "primary_six_cell"
TASK_DROP = 0.02


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


final_table = _load("final_table", "scripts/final_table.py")


def splits_used(all_splits):
    got = [s for s in all_splits if s[1] in PRIMARY_SPLITS]
    if [s[1] for s in got] != list(PRIMARY_SPLITS):
        raise SystemExit("locked splits do not contain the three primary splits in order: %s" % [s[1] for s in all_splits])
    return got


def _values(per_split):
    """per_split: {split: _stats dict} -> cells, task means and the primary statistic (None if any cell is None)."""
    out = {}
    for s in PRIMARY_SPLITS:
        for t in TASKS:
            out["%s/%s" % (s, t)] = per_split[s].get(t)
    cells = list(out.values())
    for s in PRIMARY_SPLITS:
        out["%s/%s" % (s, bootstrap.MACRO)] = per_split[s].get(bootstrap.MACRO)
    if any(v is None for v in cells):
        out.update({t: None for t in TASKS}, **{PRIMARY: None})
        return out
    for t in TASKS:
        out[t] = sum(per_split[s][t] for s in PRIMARY_SPLITS) / len(PRIMARY_SPLITS)
    out[PRIMARY] = sum(cells) / len(cells)
    return out


def compute(flags_by_split, ref, n_boot=2000, level=0.95):
    """flags_by_split: {split: (main sample rows, {arm: {seed: {sample_id: bool}}})}. Pure function (tests use it)."""
    counts, labels = {}, None
    for s in PRIMARY_SPLITS:
        rows, arms = flags_by_split[s]
        labels = labels or list(arms)
        if list(arms) != labels:
            raise ValueError("split %s has arms %s, expected %s" % (s, list(arms), labels))
        groups, task_list = None, None
        for a in arms:
            for sd, fl in arms[a].items():
                g, tl, tab = bootstrap._counts(rows, fl)
                if groups is None:
                    groups, task_list = g, tl
                elif g != groups or tl != task_list:
                    raise ValueError("%s seed %s on %s covers different samples" % (a, sd, s))
                counts[(s, a, sd)] = tab
        if not groups or not set(TASKS) <= set(task_list):
            raise ValueError("split %s lacks main samples of %s" % (s, TASKS))
        counts[s] = (groups, task_list)
    if ref not in labels:
        raise ValueError("reference %s is not an arm" % ref)
    seeds_of = {a: sorted(flags_by_split[PRIMARY_SPLITS[0]][1][a]) for a in labels}

    def score(draws):
        per = {}
        for a in labels:
            per[a] = {sd: _values({s: bootstrap._stats(draws[s], counts[(s, a, sd)], counts[s][1]) for s in PRIMARY_SPLITS})
                      for sd in seeds_of[a]}
        pooled = {a: {k: seeds._mean([per[a][sd][k] for sd in seeds_of[a]]) for k in per[a][seeds_of[a][0]]} for a in labels}
        return per, pooled

    point_per, point = score({s: counts[s][0] for s in PRIMARY_SPLITS})
    keys = list(point[ref])
    pairs = [(a, ref) for a in labels if a != ref] + ([("E", "D")] if "E" in labels and "D" in labels and ref not in ("E", "D") else [])
    rngs = {s: Rng("primary-v1", s, len(counts[s][0]), n_boot) for s in PRIMARY_SPLITS}
    reps = {a: {k: [] for k in keys} for a in labels}
    diffs = {p: {k: [] for k in keys} for p in pairs}
    for _ in range(n_boot):
        draws = {}
        for s in PRIMARY_SPLITS:
            g = counts[s][0]
            draws[s] = [g[rngs[s].below(len(g))] for _ in g]
        _, cur = score(draws)
        for a in labels:
            for k in keys:
                if cur[a][k] is not None:
                    reps[a][k].append(cur[a][k])
        for a, b in pairs:
            for k in keys:
                if cur[a][k] is not None and cur[b][k] is not None:
                    diffs[(a, b)][k].append(cur[a][k] - cur[b][k])

    def entry(p, r):
        if p is None or not r:
            return None
        lo, hi = bootstrap._interval(r, level)
        return {"point": p, "lo": lo, "hi": hi}

    out = {"statistic": PRIMARY, "splits": list(PRIMARY_SPLITS), "tasks": list(TASKS), "reference": ref, "n_boot": n_boot,
           "level": level, "rng": "primary-v1", "root_graphs": {s: len(counts[s][0]) for s in PRIMARY_SPLITS},
           "arms": [], "differences": []}
    for a in labels:
        out["arms"].append({"label": a, "seeds": seeds_of[a],
                            "per_seed": {str(sd): point_per[a][sd] for sd in seeds_of[a]},
                            "pooled": {k: entry(point[a][k], reps[a][k]) for k in keys}})
    for a, b in pairs:
        d = {k: entry(None if point[a][k] is None or point[b][k] is None else point[a][k] - point[b][k], diffs[(a, b)][k])
             for k in keys}
        if d[PRIMARY] is not None:
            d[PRIMARY]["share_above_zero"] = sum(1 for v in diffs[(a, b)][PRIMARY] if v > 0) / len(diffs[(a, b)][PRIMARY])
        matched = {str(sd): point_per[a][sd][PRIMARY] - point_per[b][sd][PRIMARY]
                   for sd in seeds_of[a] if sd in seeds_of[b]
                   and point_per[a][sd][PRIMARY] is not None and point_per[b][sd][PRIMARY] is not None}
        prim = d[PRIMARY]
        split_points = [d["%s/%s" % (s, bootstrap.MACRO)]["point"] for s in PRIMARY_SPLITS]
        row = {"a": a, "b": b, "diff": d, "seed_matched": matched,
               "reading": seeds.reading(prim, matched, seeds_of[a], seeds_of[b]),
               "task_drop": [t for t in TASKS if d[t] is not None and d[t]["point"] < -TASK_DROP],
               "splits_same_sign": prim is not None and prim["point"] != 0
               and all((p > 0) == (prim["point"] > 0) and p != 0 for p in split_points)}
        out["differences"].append(row)
    return out


def build(jobs, all_splits, ref, n_boot=2000, root=ROOT):
    root = Path(root)
    used = splits_used(all_splits)
    gaps = final_table.missing(jobs, used, root)
    if gaps:
        raise SystemExit("primary statistic refused, %d of %d final runs not collected: %s"
                         % (len(gaps), len(jobs), ", ".join(j for j, _ in gaps)))
    arm_seeds = {}
    for e in jobs:
        arm_seeds.setdefault(e["arm"], set()).add(e["seed"])
    bad = sorted(a for a, s in arm_seeds.items() if s != set(seeds.PREREGISTERED_SEEDS))
    if bad or ref not in arm_seeds:
        raise SystemExit("every arm needs exactly the three pre-registered seeds and the reference %s; check: %s" % (ref, bad))
    flags_by_split, files = {}, []
    for manifest, split, roots in used:
        man, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
        rows = arms_mod.eval_samples(by_split[split], roots)
        arms = {}
        for e in jobs:
            p_path = root / e["dev_dir"] / ("flags_%s.json" % split)
            p = json.loads(p_path.read_text())
            if (p.get("split") != split or p.get("eval_root_graphs") != roots or p.get("sample_set", "main") != "main"
                    or p.get("content_digest") != man["content_digest"] or p.get("missing_responses")):
                raise SystemExit("%s is not a complete main %s evaluation on %d roots of this build" % (p_path, split, roots))
            arms.setdefault(e["arm"], {})[e["seed"]] = bootstrap.decode_flags(rows, p)
            files.append(str(p_path.relative_to(root)) if p_path.is_relative_to(root) else str(p_path))
        flags_by_split[split] = (rows, arms)
    t = compute(flags_by_split, ref, n_boot)
    t.update({"jobs": [e["job"] for e in jobs], "files": files})
    return t


def _ci(e, signed=False):
    f = "%+.3f [%+.3f, %+.3f]" if signed else "%.3f [%.3f, %.3f]"
    return "n/a" if e is None else f % (e["point"], e["lo"], e["hi"])


def markdown(t):
    lines = ["Locked test, pre-registered primary statistic: unweighted mean of six accuracies {%s} x {%s}, main samples, "
             "pooled over three seeds; 95%% stratified paired bootstrap over root graphs (%d replicates, %s). "
             "Only this statistic decides the hypothesis." % (", ".join(t["tasks"]), ", ".join(t["splits"]), t["n_boot"], t["rng"]),
             "", "| Arm | Primary | " + " | ".join("%s mean" % x for x in t["tasks"]) + " | Per seed (primary) |",
             "|---|---|" + "---|" * len(t["tasks"]) + "---|"]
    for a in t["arms"]:
        per = ", ".join("%s: %.3f" % (sd, v[PRIMARY]) for sd, v in a["per_seed"].items() if v[PRIMARY] is not None)
        lines.append("| %s | %s | %s | %s |" % (a["label"], _ci(a["pooled"][PRIMARY]),
                                                " | ".join(_ci(a["pooled"][x]) for x in t["tasks"]), per))
    lines += ["", "| Difference | Primary | Reading | Seed-matched | Task drop > %.2f | Splits same sign |" % TASK_DROP,
              "|---|---|---|---|---|---|"]
    for d in t["differences"]:
        sm = ", ".join("%s: %+.3f" % (k, v) for k, v in d["seed_matched"].items())
        lines.append("| %s - %s | %s | %s | %s | %s | %s |" % (d["a"], d["b"], _ci(d["diff"][PRIMARY], True), d["reading"], sm,
                                                             ", ".join(d["task_drop"]) or "none", "yes" if d["splits_same_sign"] else "no"))
    lines += ["", "Rules: `scripts/final_primary.py` docstring, `src/trace_vr/seeds.py`; the E - D row is reported, it does not "
              "change any reading against the reference."]
    return "\n".join(lines) + "\n"


def latex(t):
    out = ["% Generated by scripts/final_primary.py; do not edit by hand.", "\\begin{tabular}{lccc}", "\\toprule",
           "Arm & Primary & Reachability mean & Critical-edge mean \\\\", "\\midrule"]
    for a in t["arms"]:
        cells = [a["pooled"][k] for k in [PRIMARY] + t["tasks"]]
        out.append("%s & %s \\\\" % (a["label"], " & ".join("%.3f {\\scriptsize[%.3f, %.3f]}" % (c["point"], c["lo"], c["hi"]) for c in cells)))
    out += ["\\midrule", "Difference & Primary & Reading & Seed-matched \\\\", "\\midrule"]
    for d in t["differences"]:
        p = d["diff"][PRIMARY]
        sm = ", ".join("$%+.3f$" % v for v in d["seed_matched"].values())
        out.append("%s $-$ %s & $%+.3f$ {\\scriptsize[$%+.3f$, $%+.3f$]} & %s & %s \\\\"
                   % (d["a"], d["b"], p["point"], p["lo"], p["hi"], d["reading"].replace("_", "\\_"), sm))
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
    t = build(jobs, final_table.SPLITS, final_table.reference(opts["reference-file"]), int(opts["boot"]))
    md = markdown(t)
    for key, text in (("out-json", json.dumps(t, indent=1) + "\n"), ("out-md", md), ("out-tex", latex(t))):
        if opts[key]:
            Path(opts[key]).parent.mkdir(parents=True, exist_ok=True)
            Path(opts[key]).write_text(text)
    print(md, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
