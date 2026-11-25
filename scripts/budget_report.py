"""Compute report from the budget log: what was measured, what failed, what was never measured.

usage: python scripts/budget_report.py [--out reports/budget.md] [--out-json reports/budget.json] [--check]

Reads only `runs/budget.jsonl` (one line per run or run part, appended by hand on Colab and by the
collectors on Kaggle). It reads no prediction, flags or metrics file, so it can be run while the locked
test is closed: the seconds of a final kernel are a cost, not a result.

What the numbers are: `gpu_seconds` is training time plus evaluation time as printed by the job itself.
Setup, model download and data generation on the GPU machine are not in it, so it undercounts the
session time a provider bills against a free quota. Lines without `gpu_seconds` are runs whose time was
never read back (lost with the VM) or placeholder lines written when a job was launched; they are listed
one by one and counted as zero, which makes every total a lower bound. Nothing is estimated here.

Every line must match exactly one group rule below; a line that matches none stops the script, so a new
kind of run has to be named before it is added to a total. `--check` compares the existing files with
what would be written and exits 1 on a difference (rerun the script after every new budget line).
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = "budget-report-v1"

# (group key, title, pattern on run_id); first match wins, order is the report's order.
GROUPS = (
    ("smoke", "Smoke and overfit checks (M3)", r"^(a0_smoke|overfit)_"),
    ("pilot1", "One-epoch pilot, arms A, B, D", r"^pilot_[ABD]_(train|eval)$"),
    ("final", "Locked-test evaluation kernels", r"_jobF\d+$"),
    ("vgcure", "External evaluation on VGCure images", r"^vgcure_"),
    ("base_prompt", "Frozen-base prompt scan (dev, 20 roots)", r"^base_prompt_"),
    ("repro", "GPU rerun of a recorded dev run from a clean clone", r"_jobR\d+$"),
    ("pilot3", "Three-epoch pilot on dev: training, evaluation and oracle condition, all arms and seeds", r"^pilot3_"),
)
FAILED = ("failed", "partial")
PLACEHOLDER = ("running",)


def load(path: Path):
    """Budget lines in file order; a line holding only a `note` is returned separately."""
    rows, notes = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "run_id" not in row:
            if set(row) != {"note"}:
                raise SystemExit("line %d: no run_id and not a note line" % number)
            notes.append(row["note"])
            continue
        row["line"] = number
        rows.append(row)
    return rows, notes


def group_of(run_id: str) -> str:
    for key, _title, pattern in GROUPS:
        if re.search(pattern, run_id):
            return key
    raise SystemExit("run_id %r matches no group rule; add one to GROUPS before reporting" % run_id)


def kind_of(row) -> str:
    """measured / measured_failed / placeholder / unmeasured, from the line's own fields."""
    status = str(row["status"])
    if row.get("gpu_seconds") is None:
        return "placeholder" if status in PLACEHOLDER else "unmeasured"
    return "measured_failed" if status.startswith(FAILED) else "measured"


def summarise(rows):
    for row in rows:
        seconds = row.get("gpu_seconds")
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0):
            raise SystemExit("line %d: gpu_seconds is not a non-negative number" % row["line"])
        if row.get("cash_cost") is None:
            raise SystemExit("line %d: no cash_cost" % row["line"])
    measured = [r for r in rows if r.get("gpu_seconds") is not None]

    def block(subset):
        return {"lines": len(subset), "gpu_seconds": round(sum(r["gpu_seconds"] for r in subset), 1)}

    groups = []
    for key, title, _pattern in GROUPS:
        subset = [r for r in rows if group_of(r["run_id"]) == key]
        if not subset:
            continue
        entry = block([r for r in subset if r.get("gpu_seconds") is not None])
        entry.update(key=key, title=title, lines_total=len(subset),
                     lines_without_seconds=sum(1 for r in subset if r.get("gpu_seconds") is None))
        groups.append(entry)
    platforms = {}
    for name in sorted({r["platform"] for r in rows}):
        subset = [r for r in rows if r["platform"] == name]
        entry = block([r for r in subset if r.get("gpu_seconds") is not None])
        entry["lines_total"] = len(subset)
        entry["device_recorded"] = dict(sorted(
            (device, sum(1 for r in subset if (r.get("device") or "not recorded") == device))
            for device in {r.get("device") or "not recorded" for r in subset}))
        platforms[name] = entry
    return {
        "rule": RULE,
        "source": "runs/budget.jsonl",
        "lines": len(rows),
        "cash_cost": sum(r["cash_cost"] for r in rows),
        "measured": block(measured),
        "measured_failed": block([r for r in rows if kind_of(r) == "measured_failed"]),
        "groups": groups,
        "platforms": platforms,
        "failed_lines": [{"line": r["line"], "run_id": r["run_id"], "gpu_seconds": r["gpu_seconds"],
                          "status": r["status"]} for r in rows if kind_of(r) == "measured_failed"],
        "unmeasured_lines": [{"line": r["line"], "run_id": r["run_id"], "platform": r["platform"],
                              "status": r["status"], "started_utc": r.get("started_utc") or r.get("date_utc")}
                             for r in rows if kind_of(r) == "unmeasured"],
        "placeholder_lines": [{"line": r["line"], "run_id": r["run_id"]} for r in rows if kind_of(r) == "placeholder"],
    }


def hours(seconds: float) -> str:
    return "%.2f" % (seconds / 3600.0)


def markdown(summary, notes) -> str:
    total = summary["measured"]
    out = ["# Compute budget", "",
           "Generated by `scripts/budget_report.py` (rule `%s`) from `%s`; do not edit by hand." % (RULE, summary["source"]),
           "",
           "- Cash spent: %s." % summary["cash_cost"],
           "- Measured GPU time: %.1f s = %s h over %d log lines (of %d)."
           % (total["gpu_seconds"], hours(total["gpu_seconds"]), total["lines"], summary["lines"]),
           "- Of that, in runs that failed or crashed: %.1f s over %d lines."
           % (summary["measured_failed"]["gpu_seconds"], summary["measured_failed"]["lines"]),
           "- Runs whose time was never read back: %d lines, counted as zero (listed below)." % len(summary["unmeasured_lines"]),
           "",
           "This is a lower bound. `gpu_seconds` is training plus evaluation time printed by each job; setup, model",
           "download and data generation on the GPU machine are not in it, and lost runs contribute nothing. The",
           "session time a provider counts against its free quota is therefore larger. Free allocation is not zero",
           "economic cost: no electricity, subscription or person time is priced here.",
           "",
           "## By purpose", "",
           "| Purpose | Lines with time | GPU seconds | GPU hours | Lines without time |",
           "|---|---:|---:|---:|---:|"]
    for g in summary["groups"]:
        out.append("| %s | %d | %.1f | %s | %d |" % (g["title"], g["lines"], g["gpu_seconds"], hours(g["gpu_seconds"]),
                                                   g["lines_without_seconds"]))
    out += ["| Total | %d | %.1f | %s | %d |" % (total["lines"], total["gpu_seconds"], hours(total["gpu_seconds"]),
                                               summary["lines"] - total["lines"]),
            "", "## By platform", "",
            "| Platform | Lines with time | GPU seconds | GPU hours | Device as logged |", "|---|---:|---:|---:|---|"]
    for name, p in summary["platforms"].items():
        devices = ", ".join("%s (%d lines)" % item for item in p["device_recorded"].items())
        out.append("| %s | %d | %.1f | %s | %s |" % (name, p["lines"], p["gpu_seconds"], hours(p["gpu_seconds"]), devices))
    out += ["", "## Failed or crashed runs with measured time", ""]
    out += ["- line %d, `%s`: %.1f s, status `%s`" % (f["line"], f["run_id"], f["gpu_seconds"], f["status"])
            for f in summary["failed_lines"]] or ["None."]
    out += ["", "## Runs without a measured time", "",
            "Their GPU time is unknown and is not in any total above.", ""]
    out += ["- line %d, `%s` (%s, started %s): %s" % (u["line"], u["run_id"], u["platform"], u["started_utc"], u["status"])
            for u in summary["unmeasured_lines"]] or ["None."]
    out += ["", "Placeholder lines written at launch (status `running`, replaced by later lines): %s."
            % (", ".join("`%s`" % p["run_id"] for p in summary["placeholder_lines"]) or "none")]
    if notes:
        out += ["", "## Notes in the log", ""] + ["- %s" % note for note in notes]
    return "\n".join(out) + "\n"


def main(argv):
    args = list(argv)
    check = "--check" in args
    if check:
        args.remove("--check")
    out_md, out_json = ROOT / "reports" / "budget.md", ROOT / "reports" / "budget.json"
    while args:
        flag = args.pop(0)
        if flag == "--out" and args:
            out_md = Path(args.pop(0))
        elif flag == "--out-json" and args:
            out_json = Path(args.pop(0))
        else:
            raise SystemExit(__doc__)
    rows, notes = load(ROOT / "runs" / "budget.jsonl")
    summary = summarise(rows)
    texts = {out_md: markdown(summary, notes), out_json: json.dumps(summary, indent=1, sort_keys=True) + "\n"}
    if check:
        stale = [str(p) for p, text in texts.items() if not p.exists() or p.read_text() != text]
        if stale:
            print("stale: %s" % ", ".join(stale))
            return 1
        print("ok")
        return 0
    for path, text in texts.items():
        path.write_text(text)
    print("%s: %.1f s = %s h measured over %d lines, %d lines without time"
          % (out_md, summary["measured"]["gpu_seconds"], hours(summary["measured"]["gpu_seconds"]),
             summary["measured"]["lines"], summary["lines"] - summary["measured"]["lines"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
