"""Report figures from recorded development-split metrics (CPU only, no model, no new number).

Usage:
  python scripts/dev_figures.py [--seeds reports/pilot/seeds_dev_interim.json] [--out-dir reports/pilot/figures]
                                [--paper-dir paper/figures|none] [--base <repo root>] [--check]

Reads the seed-pooled dev record written by scripts/seed_table.py and, for the cost figure, the
`summary.json` next to each flags file named in that record. Draws

  arms_dev   accuracy per task and arm: pooled value, its 95% root-graph interval, one dot per seed
  cost_dev   non-edit macro accuracy of each run against its measured T4 seconds (training + dev evaluation)
  errors_dev what each run's responses were, per task: right, well-formed but wrong, or a format failure
             (for arm E a critical-edge question the solver could not answer on the transcribed graph)

as SVG and PNG in --out-dir, the first and the third also as PDF in --paper-dir, and writes `figures_dev.json` with
every plotted value and the sha256 of each source file. Both accuracy axes run from 0 to 1 and the
time axis starts at 0. The only value not copied is the well-formed-but-wrong share of errors_dev,
1 - accuracy - format failure rate of the same summary (a format failure always scores as wrong). Only the split `dev` is accepted; `--check` redraws nothing and exits 1 if the
stored `figures_dev.json` no longer matches the sources.
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

TASKS = [
    ("minimal_graph_repair", "Repair"),
    ("reachability", "Reachability"),
    ("critical_edge", "Critical edge"),
    ("non_edit_macro_accuracy", "Non-edit macro"),
]
MACRO = "non_edit_macro_accuracy"
ARM_NAMES = {
    "A": "A frozen base",
    "B": "B question answering",
    "C": "C structural",
    "D": "D repair",
    "E": "E parse and solve",
}
# Colour-blind-safe (Okabe-Ito); markers differ too, so the figure survives greyscale print.
STYLE = {
    "B": ("#0072B2", "o"),
    "C": ("#E69F00", "s"),
    "D": ("#D55E00", "^"),
    "E": ("#009E73", "D"),
    "A": ("#555555", "v"),
}
FALLBACK = ("#000000", "x")
MACRO_TOLERANCE = 1e-9
ERROR_TASKS = TASKS[:3]
REPAIR = "minimal_graph_repair"
REPAIR_DETAIL = ("status_accuracy", "valid_edit_rate", "minimal_repair_accuracy", "n_true_edited", "n_parsed")
PREMISE = "premise_fails_on_predicted_graph"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_record(path: Path) -> Dict[str, Any]:
    rec = json.loads(path.read_text())
    if rec.get("split") != "dev":
        raise SystemExit("refused: figures are drawn from the dev split only, record says %r" % rec.get("split"))
    for arm in rec["arms"]:
        for task, _ in TASKS:
            if task not in arm["pooled"]:
                raise SystemExit("record has no pooled %s for arm %s" % (task, arm["label"]))
    return rec


def arm_points(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per arm and task: pooled point, interval, per-seed values. Copied, not recomputed."""
    rows = []
    for arm in rec["arms"]:
        for task, _ in TASKS:
            p = arm["pooled"][task]
            rows.append({
                "arm": arm["label"], "task": task,
                "point": p["point"], "lo": p["lo"], "hi": p["hi"],
                "per_seed": {str(s): arm["per_seed"][str(s)][task] for s in arm["seeds"]},
            })
    return rows


def cost_points(rec: Dict[str, Any], base: Path) -> List[Dict[str, Any]]:
    """One row per run: measured training and dev-evaluation seconds from its summary.json.

    The macro accuracy on the y axis is the seed record's; the summary must report the same value,
    otherwise the two files describe different evaluations and the figure is refused."""
    per_seed = {a["label"]: a["per_seed"] for a in rec["arms"]}
    rows = []
    for f in rec["files"]:
        summary_path = (base / f["file"]).parent / "summary.json"
        if not summary_path.exists():
            raise SystemExit("missing %s (needed for the cost figure)" % _rel(summary_path))
        s = json.loads(summary_path.read_text())
        ev, tr = s["eval_dev"], s["train"]
        macro = per_seed[f["label"]][str(f["seed"])][MACRO]
        if abs(ev["metrics"][MACRO] - macro) > MACRO_TOLERANCE:
            raise SystemExit("refused: %s reports macro %r, seed record %r" % (_rel(summary_path), ev["metrics"][MACRO], macro))
        if tr["seed"] != f["seed"]:
            raise SystemExit("refused: %s was trained with seed %r, record says %r" % (_rel(summary_path), tr["seed"], f["seed"]))
        rows.append({
            "arm": f["label"], "seed": f["seed"], "macro": macro,
            "train_seconds": tr["train_seconds"], "eval_seconds": ev["eval_seconds"],
            "gpu_seconds": tr["train_seconds"] + ev["eval_seconds"],
            "gpu": tr.get("gpu"), "eval_samples": ev["samples"],
            "summary": _rel(summary_path), "summary_sha256": _sha256(summary_path),
        })
    return rows


def error_points(rec: Dict[str, Any], base: Path) -> List[Dict[str, Any]]:
    """One row per run and task: strict accuracy and format-failure rate copied from the run's summary.json,
    plus their remainder (responses that parsed but were wrong).

    Refused if the summary's accuracy is not the seed record's, or if the three shares do not form a
    partition. Repair rows carry the summary's repair detail; arm E rows carry the solver's count of
    critical-edge questions whose premise failed on the transcribed graph, which must be exactly its
    critical-edge format failures (that is how parse_solve reports them)."""
    per_seed = {a["label"]: a["per_seed"] for a in rec["arms"]}
    rows = []
    for f in rec["files"]:
        summary_path = (base / f["file"]).parent / "summary.json"
        if not summary_path.exists():
            raise SystemExit("missing %s (needed for the error figure)" % _rel(summary_path))
        ev = json.loads(summary_path.read_text())["eval_dev"]
        if ev["split"] != "dev":
            raise SystemExit("refused: %s is an evaluation of %r" % (_rel(summary_path), ev["split"]))
        for task, _ in ERROR_TASKS:
            m = ev["metrics"]["per_task"][task]
            acc, ff = m["accuracy"], m["format_failure_rate"]
            if abs(acc - per_seed[f["label"]][str(f["seed"])][task]) > MACRO_TOLERANCE:
                raise SystemExit("refused: %s reports %s %r, seed record %r" % (
                    _rel(summary_path), task, acc, per_seed[f["label"]][str(f["seed"])][task]))
            wrong = 1.0 - acc - ff
            if wrong < -MACRO_TOLERANCE:
                raise SystemExit("refused: %s %s: accuracy %r + format failures %r exceed 1" % (_rel(summary_path), task, acc, ff))
            row = {"arm": f["label"], "seed": f["seed"], "task": task, "n": m["n"],
                   "correct": acc, "format_failure": ff, "wrong_well_formed": max(wrong, 0.0),
                   "summary": _rel(summary_path)}
            if task == REPAIR:
                row["repair_detail"] = {k: m[k] for k in REPAIR_DETAIL}
            if task == "critical_edge" and "reasons" in ev:
                count = ev["reasons"].get(PREMISE, 0)
                if abs(count - ff * m["n"]) > 1e-6:
                    raise SystemExit("refused: %s: %d premise failures but %r x %d critical-edge format failures" % (
                        _rel(summary_path), count, ff, m["n"]))
                row["premise_failures"] = count
            rows.append(row)
    return rows


def build(seeds_path: Path, base: Path = ROOT) -> Dict[str, Any]:
    rec = load_record(seeds_path)
    return {
        "rule": "dev-figures-v2",
        "split": rec["split"], "sample_set": rec.get("sample_set"), "root_graphs": rec["root_graphs"],
        "level": rec.get("level"), "n_boot": rec.get("n_boot"), "content_digest": rec.get("content_digest"),
        "source": _rel(seeds_path), "source_sha256": _sha256(seeds_path),
        "arms": [a["label"] for a in rec["arms"]],
        "arms_dev": arm_points(rec),
        "cost_dev": cost_points(rec, base),
        "errors_dev": error_points(rec, base),
        "note": "Values are copied from the sources; nothing is re-estimated here (errors_dev.wrong_well_formed is "
                "1 - correct - format_failure of the same row). Development split only.",
    }


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "svg.hashsalt": "trace-dev-figures", "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    })
    import matplotlib.pyplot as plt
    return plt


def _save(fig, stem: Path, formats) -> List[str]:
    out = []
    for fmt in formats:
        path = stem.with_suffix("." + fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"svg": {"Date": None}, "pdf": {"CreationDate": None, "ModDate": None}, "png": {"Software": None}}[fmt]
        fig.savefig(path, format=fmt, dpi=200, metadata=meta)
        out.append(_rel(path))
    return out


def draw_arms(data: Dict[str, Any], plt):
    arms = data["arms"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        colour, marker = STYLE.get(arm, FALLBACK)
        offset = (i - (len(arms) - 1) / 2) * width
        rows = [r for r in data["arms_dev"] if r["arm"] == arm]
        xs = [j + offset for j in range(len(TASKS))]
        ys = [r["point"] for r in rows]
        err = [[r["point"] - r["lo"] for r in rows], [r["hi"] - r["point"] for r in rows]]
        for x, r in zip(xs, rows):
            vals = list(r["per_seed"].values())
            ax.scatter([x + 0.035] * len(vals), vals, s=7, color=colour, alpha=0.55, linewidths=0, zorder=2)
        ax.errorbar(xs, ys, yerr=err, fmt=marker, color=colour, markersize=5, capsize=2.5, linewidth=1.1,
                    label=ARM_NAMES.get(arm, arm), zorder=3)
    ax.axvline(len(TASKS) - 1.5, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([name for _, name in TASKS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy (strict)")
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=len(arms), loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    return fig


def draw_cost(data: Dict[str, Any], plt):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for arm in data["arms"]:
        colour, marker = STYLE.get(arm, FALLBACK)
        rows = [r for r in data["cost_dev"] if r["arm"] == arm]
        ax.scatter([r["gpu_seconds"] / 60 for r in rows], [r["macro"] for r in rows], s=26, color=colour,
                   marker=marker, label=ARM_NAMES.get(arm, arm), zorder=3)
    ax.set_xlim(0, max(r["gpu_seconds"] for r in data["cost_dev"]) / 60 * 1.08)
    ax.set_ylim(0, 1)
    ax.set_xlabel("T4 minutes per run (training + dev evaluation)")
    ax.set_ylabel("Non-edit macro accuracy")
    ax.grid(True, color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def draw_errors(data: Dict[str, Any], plt):
    """Three panels (tasks), one stacked bar per run: right / parsed but wrong / format failure."""
    arms = data["arms"]
    runs = [(r["arm"], r["seed"]) for r in data["errors_dev"] if r["task"] == ERROR_TASKS[0][0]]
    fig, axes = plt.subplots(1, len(ERROR_TASKS), figsize=(6.6, 2.9), sharey=True)
    xs, x = {}, 0.0
    for arm in arms:
        for run in [r for r in runs if r[0] == arm]:
            xs[run] = x
            x += 1.0
        x += 0.7
    for ax, (task, name) in zip(axes, ERROR_TASKS):
        for r in [r for r in data["errors_dev"] if r["task"] == task]:
            colour, _ = STYLE.get(r["arm"], FALLBACK)
            pos = xs[(r["arm"], r["seed"])]
            ax.bar(pos, r["correct"], width=0.86, color=colour, linewidth=0, zorder=2)
            ax.bar(pos, r["wrong_well_formed"], bottom=r["correct"], width=0.86, color="#d9d9d9", linewidth=0, zorder=2)
            ax.bar(pos, r["format_failure"], bottom=r["correct"] + r["wrong_well_formed"], width=0.86, color="#333333",
                   linewidth=0, zorder=2)
        ax.set_title(name, fontsize=9)
        centres = [sum(xs[r] for r in runs if r[0] == arm) / len([r for r in runs if r[0] == arm]) for arm in arms]
        ax.set_xticks(centres)
        ax.set_xticklabels(arms)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of responses")
    from matplotlib.patches import Patch
    handles = [Patch(color=STYLE.get(a, FALLBACK)[0], label=a + " right") for a in arms]
    handles += [Patch(color="#d9d9d9", label="parsed, wrong"), Patch(color="#333333", label="format failure")]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def write_markdown(data: Dict[str, Any], out_dir: Path) -> None:
    n = data["root_graphs"]
    lines = [
        "# Development-split figures",
        "",
        "Generated by `scripts/dev_figures.py` from `%s` (rule `%s`). Development split, %s root graphs; "
        "no locked-test value is used. Plotted values and source hashes: `figures_dev.json`." % (data["source"], data["rule"], n),
        "",
        "## Arm comparison",
        "",
        "![arms](arms_dev.png)",
        "",
        "Marker = accuracy pooled over the training seeds, bar = its 95% root-graph bootstrap interval, small dots = single seeds. "
        "The y axis runs from 0 to 1. The frozen base (arm A) is not drawn: it was evaluated on 20 of the dev roots only "
        "(`reports/pilot/base_prompt.md`).",
        "",
        "## Cost against accuracy",
        "",
        "![cost](cost_dev.png)",
        "",
        "One point per run. x = measured training seconds + dev evaluation seconds on a Kaggle T4 (kernel setup and data "
        "generation not counted), y = that run's non-edit macro accuracy. Arm E's seconds are its parser's training plus the "
        "transcribe-and-solve evaluation.",
        "",
        "| Arm | Seed | Train s | Dev eval s | Total min | Non-edit macro |",
        "|---|---|---|---|---|---|",
    ]
    for r in data["cost_dev"]:
        lines.append("| %s | %d | %.1f | %.1f | %.1f | %.3f |" % (
            r["arm"], r["seed"], r["train_seconds"], r["eval_seconds"], r["gpu_seconds"] / 60, r["macro"]))
    lines += [
        "",
        "## Error types",
        "",
        "![errors](errors_dev.png)",
        "",
        "One bar per run (three training seeds per arm, in the order of the table below). Coloured = strict accuracy, "
        "light grey = the response parsed but was wrong, dark = format failure (strict parser; counts as wrong). "
        "Accuracy and format-failure rate are copied from each run's `summary.json`; the grey share is their remainder. "
        "For arm E a dark segment on the critical-edge task is a question whose premise failed on the transcribed graph, "
        "so the solver gave no answer; the count is in the last column.",
        "",
        "| Arm | Seed | Task | n | Right | Parsed, wrong | Format failure | E: premise failed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    names = dict(ERROR_TASKS)
    for r in data["errors_dev"]:
        lines.append("| %s | %d | %s | %d | %.4f | %.4f | %.4f | %s |" % (
            r["arm"], r["seed"], names[r["task"]], r["n"], r["correct"], r["wrong_well_formed"], r["format_failure"],
            r.get("premise_failures", "")))
    lines += [
        "",
        "Repair detail (copied from the summaries). Status = the response's `status` field equals the solver's, "
        "over all repair samples; valid edit and minimal repair are over the samples whose reference "
        "answer changes the graph.",
        "",
        "| Arm | Seed | Status right | Valid edit | Minimal repair | Samples needing an edit | Responses parsed |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in data["errors_dev"]:
        if "repair_detail" in r:
            d = r["repair_detail"]
            lines.append("| %s | %d | %.4f | %.4f | %.4f | %d | %d of %d |" % (
                r["arm"], r["seed"], d["status_accuracy"], d["valid_edit_rate"], d["minimal_repair_accuracy"],
                d["n_true_edited"], d["n_parsed"], r["n"]))
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: List[str]) -> int:
    args = list(argv)
    opts = {"seeds": "reports/pilot/seeds_dev_interim.json", "out-dir": "reports/pilot/figures", "paper-dir": "paper/figures",
            "base": str(ROOT)}
    for key in list(opts):
        flag = "--" + key
        if flag in args:
            i = args.index(flag); opts[key] = args[i + 1]; del args[i:i + 2]
    check = "--check" in args
    if check:
        args.remove("--check")
    if args:
        raise SystemExit(__doc__)
    seeds_path, out_dir = Path(opts["seeds"]), Path(opts["out-dir"])
    data = build(seeds_path, base=Path(opts["base"]))
    record = out_dir / "figures_dev.json"
    if check:
        stored = json.loads(record.read_text()) if record.exists() else None
        if stored is None or {k: v for k, v in stored.items() if k != "files"} != data:
            print("figures_dev.json is stale: rerun scripts/dev_figures.py")
            return 1
        print("figures_dev.json matches its sources")
        return 0
    plt = _mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    fig = draw_arms(data, plt)
    files += _save(fig, out_dir / "arms_dev", ("svg", "png"))
    if opts["paper-dir"] != "none":
        files += _save(fig, Path(opts["paper-dir"]) / "arms_dev", ("pdf",))
    plt.close(fig)
    fig = draw_cost(data, plt)
    files += _save(fig, out_dir / "cost_dev", ("svg", "png"))
    plt.close(fig)
    fig = draw_errors(data, plt)
    files += _save(fig, out_dir / "errors_dev", ("svg", "png"))
    if opts["paper-dir"] != "none":
        files += _save(fig, Path(opts["paper-dir"]) / "errors_dev", ("pdf",))
    plt.close(fig)
    record.write_text(json.dumps(dict(data, files=files), indent=1) + "\n")
    write_markdown(data, out_dir)
    print("wrote %d files; %d arm points, %d runs, %d error rows" % (
        len(files) + 2, len(data["arms_dev"]), len(data["cost_dev"]), len(data["errors_dev"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
