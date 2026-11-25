"""Cost plan for the locked-test (final) evaluation queue; reads no test file, runs nothing on a GPU.

Usage: python scripts/kaggle/final_plan.py [--now 2026-11-13T22:00:00Z] [--quota-hours 30] [--running JOB]
                                           [--out reports/pilot/final_plan] [--json | --gate]

--gate prints one line, the push verdict for the next pending final kernel (use this, not quota.py, which
reads queue.json and does not know the final kernels' calibrated estimates). It also runs eval_freeze.py:
if the evaluation code at HEAD differs from what jobF01 ran, the verdict is "stop".

For every entry of scripts/kaggle/final_queue.json the session time of its final kernel is estimated as

    dev_eval_seconds * (final_roots / dev_roots) * (1 + inv_per_main) + setup_seconds

- dev_eval_seconds: the run's measured dev evaluation (summary.json eval_dev.eval_seconds, 200 roots).
- final_roots: roots of the six locked splits that final_eval.sh scores, read from the split metadata of
  manifests/pilot_test.json and manifests/pilot_diag.json (counts only; no sample file is opened).
- inv_per_main: invariance samples per main sample on dev (212 / 1200), because final_eval.sh also scores the
  relabelled/restyled copies; measured from data/pilot when present, else the recorded value.
- setup_seconds: 2 x the largest non-model time of a past Kaggle session (session - train - eval), since the
  final kernel regenerates pilot, pilot_test and pilot_diag. An assumption, labelled as such.
Kernels are packed in queue order into the remaining weekly quota (quota.py's accounting, 2 h margin).
Kaggle stops a session at 12 h; an estimate above 11 h is flagged.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quota  # noqa: E402

FINAL_SPLITS = {"pilot_test": ["test_id", "test_style_ood", "test_struct_ood", "test_combined_ood"],
                "pilot_diag": ["test_family_ood", "test_size_ood"]}
DEV_ROOTS = 200
INV_PER_MAIN_RECORDED = 212 / 1200   # data/pilot dev, first 200 roots, measured 2026-11-13
SESSION_LIMIT_S = 12 * 3600
FLAG_S = 11 * 3600


def final_roots(root=ROOT):
    n = 0
    for man, splits in FINAL_SPLITS.items():
        meta = json.loads((Path(root) / "manifests" / ("%s.json" % man)).read_text())["splits"]
        n += sum(meta[s]["roots"] for s in splits)
    return n


def inv_per_main(root=ROOT):
    path = Path(root) / "data/pilot/dev.samples.jsonl"
    if not path.exists():
        return INV_PER_MAIN_RECORDED, "recorded"
    sys.path.insert(0, str(Path(root) / "src"))
    from trace_vr import arms
    samples = [json.loads(line) for line in path.open()]
    main = arms.eval_samples(samples, DEV_ROOTS)
    inv = arms.invariance_samples(samples, DEV_ROOTS)
    return len(inv) / len(main), "measured"


def setup_seconds(kaggle_dir, root=ROOT):
    """2 x the largest (session - train - eval) over past sessions whose run summaries are local."""
    over = []
    for s in quota.sessions(kaggle_dir):
        model = 0.0
        log = (Path(kaggle_dir) / s["job"] / ("%s.log" % s["job"])).read_text()
        for line in dict.fromkeys(log.splitlines()):   # a resumed job repeats identical summary lines
            for tag, keys in (("TRAIN-SUMMARY ", ("train_seconds",)), ("EVAL-SUMMARY ", ("seconds", "eval_seconds"))):
                if line.startswith(tag):
                    try:
                        d = json.loads(line[len(tag):])
                    except ValueError:
                        continue
                    model += float(next((d[k] for k in keys if k in d), 0.0))
        if model > 0:
            over.append(s["session_seconds"] - model)
    return 2 * max(over) if over else 1800.0


def base_row(queue_path, root, scale, setup, choice=None):
    """Cost row of the frozen base's final kernel (arm A, the queue's "base" job), outside the packed queue.

    Before the dev choice exists the row only says so. Afterwards the basis is the chosen variant's dev evaluation,
    which ran at the long scan limit; the final kernel runs at the chosen limit, so when that is smaller the estimate
    is an upper bound (generation stops earlier), and it is labelled as one."""
    base = json.loads(Path(queue_path).read_text()).get("base")
    if not base:
        return None
    row = {"job": base["job"], "arm": base["arm"], "done": (Path(root) / "runs/kaggle" / base["job"]).exists()}
    choice = Path(choice or Path(root) / base["choice"])
    if not choice.exists():
        return dict(row, state="no dev choice yet (%s missing): cost unknown" % base["choice"])
    c = json.loads(choice.read_text())
    run = "base_prompt_%s" % c["chosen"]
    summ = Path(root) / "runs" / run / "summary.json"
    if not summ.exists():
        return dict(row, run=run, state="runs/%s/summary.json missing (collect_base_prompt.py): cost unknown" % run)
    dev = json.loads(summ.read_text())["eval_dev"]
    dev_s = float(dev["eval_seconds"]) * DEV_ROOTS / float(dev["eval_root_graphs"])
    upper = c["chosen_limit"] < dev["max_new_tokens"]
    return dict(row, run=run, variant=c["chosen"], limit=c["chosen_limit"], dev_limit=dev["max_new_tokens"],
                dev_eval_root_graphs=dev["eval_root_graphs"], dev_eval_seconds_per_200_roots=round(dev_s, 1),
                estimate_hours=round((dev_s * scale + setup) / 3600, 2), upper_bound=upper,
                over_11h=dev_s * scale + setup > FLAG_S,
                state="estimate from %d dev roots at %d new tokens%s" % (
                    dev["eval_root_graphs"], dev["max_new_tokens"],
                    "; upper bound, the final kernel stops at %d" % c["chosen_limit"] if upper else ""))


def measured_final_seconds(root=ROOT):
    """Measured locked-test eval seconds per collected final kernel, from runs/budget.jsonl (run_id <run>_<job>).
    A duration, not a test outcome."""
    out, path = {}, Path(root) / "runs/budget.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                if "final locked-test" in d.get("status", ""):
                    out[d["run_id"].rsplit("_", 1)[-1]] = float(d["gpu_seconds"])
    return out


def plan(queue_path, kaggle_dir, root, now, quota_hours=30.0, running=None, choice=None):
    jobs = json.loads(Path(queue_path).read_text())["jobs"]
    q = quota.check(kaggle_dir, Path(root) / "scripts/kaggle/queue.json", root, now, quota_hours, running)
    roots = final_roots(root)
    ipm, ipm_src = inv_per_main(root)
    setup = setup_seconds(kaggle_dir, root)
    scale = roots / DEV_ROOTS * (1 + ipm)
    measured = measured_final_seconds(root)
    rows, cum, fits_open = [], 0.0, True
    ratios = {}   # arm -> measured / estimated eval seconds of its collected final kernels
    for j in jobs:
        if j["job"] in measured and (Path(root) / j["dev_dir"] / "summary.json").exists():
            dev = json.loads((Path(root) / j["dev_dir"] / "summary.json").read_text())["eval_dev"]
            d = float(dev["eval_seconds"]) * DEV_ROOTS / float(dev.get("eval_root_graphs", DEV_ROOTS))
            ratios.setdefault(j["arm"], []).append(measured[j["job"]] / (d * scale))
    for j in jobs:
        done = (Path(root) / "runs/kaggle" / j["job"]).exists()
        summ = Path(root) / j["dev_dir"] / "summary.json"
        if summ.exists():
            dev = json.loads(summ.read_text())["eval_dev"]
            dev_s, basis = float(dev["eval_seconds"]) * DEV_ROOTS / float(dev.get("eval_root_graphs", DEV_ROOTS)), j["dev_dir"]
        else:   # run not collected yet (job22): same arm's slowest measured dev eval
            same = [json.loads((Path(root) / k["dev_dir"] / "summary.json").read_text())["eval_dev"]["eval_seconds"]
                    for k in jobs if k["arm"] == j["arm"] and (Path(root) / k["dev_dir"] / "summary.json").exists()]
            dev_s, basis = float(max(same)), "max of arm %s (run not collected)" % j["arm"]
        est = dev_s * scale + setup
        ratio = max(ratios.get(j["arm"], [1.0]) + [1.0])   # never below the plain estimate
        cal = dev_s * scale * ratio + setup
        rows.append({"job": j["job"], "arm": j["arm"], "seed": j["seed"], "source": j["source"], "run": j["run"],
                     "parser": j.get("parser"), "dev_eval_seconds": round(dev_s, 1), "basis": basis,
                     "estimate_hours": round(est / 3600, 2), "arm_ratio": round(ratio, 3),
                     "calibrated_hours": round(cal / 3600, 2), "over_11h": cal > FLAG_S, "done": done,
                     "fits_this_week": False, "_cal_s": cal})
    # A running final kernel is counted with its own calibrated estimate (quota.py only knows queue.json and would
    # use the arm's longest past session) and is not packed again.
    run_row = next((r for r in rows if r["job"] == running and not r["done"]), None)
    run_s = run_row["_cal_s"] if run_row else q["running_estimate_hours"] * 3600
    left = quota_hours * 3600 - quota.MARGIN_S - q["used_hours"] * 3600 - run_s
    for r in rows:
        cal = r.pop("_cal_s")
        if r["done"] or r is run_row:
            continue
        if fits_open and cum + cal <= left:
            cum += cal
            r["fits_this_week"] = True
        else:
            fits_open = False
    nxt = next((r for r in rows if not r["done"] and r is not run_row), None)
    total = sum(r["calibrated_hours"] for r in rows if not r["done"])
    return {"now_utc": q["now_utc"], "quota_hours_assumed": quota_hours, "used_hours": q["used_hours"],
            "running": running, "running_estimate_hours": round(run_s / 3600, 2),
            "next": nxt["job"] if nxt else None,
            "verdict": ("ok" if nxt["fits_this_week"] else "wait") if nxt else "done",
            "left_this_week_hours": round(left / 3600, 2), "final_roots": roots, "inv_per_main": round(ipm, 4),
            "inv_per_main_source": ipm_src, "setup_seconds_assumed": round(setup, 1), "scale": round(scale, 3),
            "total_estimate_hours": round(total, 2),
            "arm_ratios": {a: [round(x, 3) for x in v] for a, v in sorted(ratios.items())}, "jobs": rows, "base": base_row(queue_path, root, scale, setup, choice)}


def to_md(p):
    out = ["# Locked-test evaluation plan (cost estimate, no test output)", "",
           "Generated by `scripts/kaggle/final_plan.py` at %s. Estimate per kernel = dev eval seconds x %.3f "
           "(%d final roots / %d dev roots x (1 + %.4f invariance per main sample, %s)) + %.0f s setup (assumed: "
           "2 x largest past non-model session time). Quota %.0f h is an assumption (quota.py); used %.2f h, running "
           "%s ~%.2f h, left this week %.2f h after the 2 h margin." % (
               p["now_utc"], p["scale"], p["final_roots"], DEV_ROOTS, p["inv_per_main"], p["inv_per_main_source"],
               p["setup_seconds_assumed"], p["quota_hours_assumed"], p["used_hours"], p["running"],
               p["running_estimate_hours"], p["left_this_week_hours"]), "",
           "Calibrated h = dev eval seconds x scale x the arm's largest measured/estimated eval-time ratio over its "
           "collected final kernels (runs/budget.jsonl; at least 1) + setup; it drives packing and the 11 h flag. "
           "Ratios: %s." % (json.dumps(p.get("arm_ratios", {})),), "",
           "| Job | Arm | Seed | Source | Run | Parser | Dev eval s | Estimate h | Calibrated h | This week | Note |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in p["jobs"]:
        note = "; ".join(x for x in ["done" if r["done"] else "", "over 11 h (12 h session limit)" if r["over_11h"] else "",
                                     "" if r["basis"].startswith("runs/") else r["basis"]] if x)
        out.append("| %s | %s | %d | %s | %s | %s | %.0f | %.2f | %.2f | %s | %s |" % (
            r["job"], r["arm"], r["seed"], r["source"], r["run"], r["parser"] or "-", r["dev_eval_seconds"],
            r["estimate_hours"], r["calibrated_hours"], "yes" if r["fits_this_week"] else "no", note or "-"))
    out += ["", "Total for pending kernels: ~%.2f h of Kaggle session time." % p["total_estimate_hours"], ""]
    b = p.get("base")
    if b:
        est = "~%.2f h%s" % (b["estimate_hours"], " (over 11 h: 12 h session limit)" if b["over_11h"] else "") if "estimate_hours" in b else "unknown"
        out += ["Frozen base (arm A, %s, reported floor; not in the total or the weekly packing above, runs after jobF08): %s; %s%s."
                % (b["job"], est, b["state"], "; done" if b["done"] else ""), ""]
    return "\n".join(out)


def main(argv):
    args = list(argv)

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    now = quota.parse_time(opt("--now", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
    p = plan(opt("--queue", ROOT / "scripts/kaggle/final_queue.json"), opt("--kaggle-dir", ROOT / "runs/kaggle"),
             ROOT, now, float(opt("--quota-hours", 30)), opt("--running", None))
    out = opt("--out", None)
    if out:
        Path(out + ".json").write_text(json.dumps(p, indent=1) + "\n")
        Path(out + ".md").write_text(to_md(p))
    if "--gate" in args:   # one line for the autopilot: push the next final kernel only on "ok"
        import eval_freeze   # a final kernel may only run the evaluation code jobF01 ran
        frozen, why = eval_freeze.check("HEAD")
        if not frozen:
            p["verdict"] = "stop"
        print("next %s; left this week %.2f h after the 2 h margin -> %s" % (p["next"], p["left_this_week_hours"], p["verdict"]))
        if not frozen:
            print(why)
    else:
        print(json.dumps(p, indent=1) if "--json" in args else to_md(p))
    return p


if __name__ == "__main__":
    main(sys.argv[1:])
