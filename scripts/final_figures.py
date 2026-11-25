"""Locked-test figure from the accepted final table (CPU only, no model, no new estimate).

Usage:
  python scripts/final_figures.py [--table reports/final/final_table.json] [--out-dir reports/final/figures]
                                  [--paper-dir paper/figures|none] [--queue scripts/kaggle/final_queue.json] [--check]

Run it after scripts/final_reports.py. It reads one file only: the JSON that scripts/final_table.py wrote once
all twelve final kernels were collected. No flags file, prediction or summary is opened here, so this script
cannot show a locked-test number before final_table.py has accepted the whole queue; while that JSON does not
exist it refuses and writes nothing. Draws

  ood_test   non-edit macro accuracy of every arm on each of the six locked splits: seed-pooled value, its 95%
             root-graph interval, one dot per training seed

as SVG and PNG in --out-dir and as PDF in --paper-dir, and writes `figures_test.json` (every plotted value, the
sha256 of the table) and a README.md with the same values as a table. The accuracy axis runs from 0 to 1.

Every plotted value is a copy of the table's. The only value computed here is `minus_test_id`, the pooled point
on a split minus the same arm's pooled point on test_id. It is a difference of two reported points on different
root graphs: it has no interval and no reading, and no claim may rest on it (the pre-registered comparisons are
those of final_primary.py and final_table.py). Splits are categories, so points are not joined by lines.
Written before any locked-test number was read (DECISIONS 2026-11-19); `--check` redraws nothing and exits 1 if
the stored `figures_test.json` no longer matches the table.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
MACRO = "non_edit_macro_accuracy"
IN_DISTRIBUTION = "test_id"
SPLIT_NAMES = {
    "test_id": "In distribution",
    "test_style_ood": "Style OOD",
    "test_struct_ood": "Structure OOD",
    "test_combined_ood": "Combined OOD",
    "test_family_ood": "Family OOD\n(diagnostic)",
    "test_size_ood": "Size OOD\n(diagnostic)",
}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dev_figures = _load("dev_figures", "scripts/dev_figures.py")
SPLITS = _load("collect_final", "scripts/kaggle/collect_final.py").SPLITS   # (manifest, split, roots)
SEEDS = sorted(_load("seed_table", "scripts/seed_table.py").seeds.PREREGISTERED_SEEDS)


def load_table(path: Path, jobs: List[str], splits=None) -> Dict[str, Any]:
    """The accepted final table, or a refusal. `jobs` are the queue's job names; the table must cover exactly them."""
    splits = SPLITS if splits is None else splits
    if not Path(path).exists():
        raise SystemExit("refused: %s does not exist; scripts/final_reports.py writes it once all final kernels "
                         "are collected" % path)
    t = json.loads(Path(path).read_text())
    if sorted(t.get("jobs", [])) != sorted(jobs):
        raise SystemExit("refused: the table covers %d runs, the queue has %d" % (len(t.get("jobs", [])), len(jobs)))
    if t.get("splits") != [s for _, s, _ in splits]:
        raise SystemExit("refused: the table's splits are %r" % (t.get("splits"),))
    labels = None
    for manifest, split, roots in splits:
        tab = t["tables"][split]
        if tab["root_graphs"] != roots or tab.get("manifest") != manifest or tab.get("split") != split:
            raise SystemExit("refused: table for %s is not %d roots of %s" % (split, roots, manifest))
        got = [a["label"] for a in tab["arms"]]
        if labels is None:
            labels = got
        if got != labels:
            raise SystemExit("refused: arms differ between splits (%s)" % split)
        for a in tab["arms"]:
            if sorted(a["seeds"]) != SEEDS:
                raise SystemExit("refused: arm %s on %s has seeds %r" % (a["label"], split, a["seeds"]))
    if t["reference"] not in labels:
        raise SystemExit("refused: reference %r is not an arm of the table" % t["reference"])
    return t


def points(t: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per split and arm: pooled non-edit macro, its interval, per-seed values. Copied, not recomputed."""
    at_id = {a["label"]: a["pooled"][MACRO]["point"] for a in t["tables"][IN_DISTRIBUTION]["arms"]} \
        if IN_DISTRIBUTION in t["tables"] else {}
    rows = []
    for split in t["splits"]:
        tab = t["tables"][split]
        for a in tab["arms"]:
            p = a["pooled"][MACRO]
            row = {"split": split, "root_graphs": tab["root_graphs"], "arm": a["label"],
                   "point": p["point"], "lo": p["lo"], "hi": p["hi"],
                   "per_seed": {str(s): a["per_seed"][str(s)][MACRO] for s in sorted(a["seeds"])}}
            if a["label"] in at_id:
                row["minus_test_id"] = p["point"] - at_id[a["label"]]
            rows.append(row)
    return rows


def build(table_path: Path, jobs: List[str], splits=None) -> Dict[str, Any]:
    t = load_table(table_path, jobs, splits)
    return {
        "rule": "final-figures-v1",
        "source": dev_figures._rel(Path(table_path)), "source_sha256": dev_figures._sha256(Path(table_path)),
        "reference": t["reference"], "n_boot": t["n_boot"], "runs": len(t["jobs"]),
        "splits": t["splits"], "arms": [a["label"] for a in t["tables"][t["splits"][0]]["arms"]],
        "ood_test": points(t),
        "note": "point, lo, hi and per_seed are copies of the final table. minus_test_id is point minus the same "
                "arm's point on test_id: different root graphs, no interval, no reading.",
    }


def draw_ood(data: Dict[str, Any], plt):
    arms, splits = data["arms"], data["splits"]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        colour, marker = dev_figures.STYLE.get(arm, dev_figures.FALLBACK)
        offset = (i - (len(arms) - 1) / 2) * width
        rows = {r["split"]: r for r in data["ood_test"] if r["arm"] == arm}
        rows = [rows[s] for s in splits]
        xs = [j + offset for j in range(len(splits))]
        err = [[r["point"] - r["lo"] for r in rows], [r["hi"] - r["point"] for r in rows]]
        for x, r in zip(xs, rows):
            vals = list(r["per_seed"].values())
            ax.scatter([x + 0.035] * len(vals), vals, s=7, color=colour, alpha=0.55, linewidths=0, zorder=2)
        ax.errorbar(xs, [r["point"] for r in rows], yerr=err, fmt=marker, color=colour, markersize=5, capsize=2.5,
                    linewidth=1.1, label=dev_figures.ARM_NAMES.get(arm, arm), zorder=3)
    for j in range(1, len(splits)):
        ax.axvline(j - 0.5, color="#dddddd", linewidth=0.6, linestyle=":")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([SPLIT_NAMES.get(s, s) for s in splits], fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Non-edit macro accuracy (strict)")
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=len(arms), loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    return fig


def write_markdown(data: Dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Locked-test figure",
        "",
        "Generated by `scripts/final_figures.py` (rule `%s`) from `%s`; do not edit by hand." % (data["rule"], data["source"]),
        "",
        "![ood](ood_test.png)",
        "",
        "Non-edit macro accuracy on the six locked splits, %d runs. Marker = pooled over the three training seeds, bar = "
        "95%% interval over root graphs (%d replicates), small dots = single seeds. All of these are copies of the final "
        "table. The last column is the pooled point minus the same arm's pooled point on `test_id`: the splits hold "
        "different root graphs, so it has no interval and is not a test of anything." % (data["runs"], data["n_boot"]),
        "",
        "| Split | Roots | Arm | Pooled | 95% interval | Seeds | Minus test_id |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in data["ood_test"]:
        lines.append("| %s | %d | %s | %.3f | [%.3f, %.3f] | %s | %s |" % (
            r["split"], r["root_graphs"], r["arm"], r["point"], r["lo"], r["hi"],
            ", ".join("%s: %.3f" % kv for kv in r["per_seed"].items()),
            "%+.3f" % r["minus_test_id"] if "minus_test_id" in r else "-"))
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: List[str]) -> int:
    args = list(argv)
    opts = {"table": "reports/final/final_table.json", "out-dir": "reports/final/figures", "paper-dir": "paper/figures",
            "queue": str(ROOT / "scripts/kaggle/final_queue.json")}
    for key in list(opts):
        flag = "--" + key
        if flag in args:
            i = args.index(flag); opts[key] = args[i + 1]; del args[i:i + 2]
    check = "--check" in args
    if check:
        args.remove("--check")
    if args:
        raise SystemExit(__doc__)
    jobs = [e["job"] for e in json.loads(Path(opts["queue"]).read_text())["jobs"]]
    out_dir = Path(opts["out-dir"])
    data = build(Path(opts["table"]), jobs)
    record = out_dir / "figures_test.json"
    if check:
        stored = json.loads(record.read_text()) if record.exists() else None
        if stored is None or {k: v for k, v in stored.items() if k != "files"} != data:
            print("figures_test.json is stale: rerun scripts/final_figures.py")
            return 1
        print("figures_test.json matches the final table")
        return 0
    plt = dev_figures._mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = draw_ood(data, plt)
    files = dev_figures._save(fig, out_dir / "ood_test", ("svg", "png"))
    if opts["paper-dir"] != "none":
        files += dev_figures._save(fig, Path(opts["paper-dir"]) / "ood_test", ("pdf",))
    plt.close(fig)
    record.write_text(json.dumps(dict(data, files=files), indent=1) + "\n")
    write_markdown(data, out_dir)
    print("wrote %d files; %d points (values not echoed, see %s)" % (len(files) + 2, len(data["ood_test"]), record))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
