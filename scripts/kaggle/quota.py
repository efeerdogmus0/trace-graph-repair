"""Kaggle GPU quota check from downloaded kernel logs (no network).

Usage: python scripts/kaggle/quota.py [--now 2026-11-13T00:00:00Z] [--quota-hours 30] [--running JOB] [--json]

Kaggle bills GPU quota by session wall time, which includes setup and data generation, so
budget.jsonl's gpu_seconds (train + eval only) undercounts it. This script reads, for every
runs/kaggle/<job>/trace-<job>.log, the largest "time" field (seconds since the session started)
and the first JOB-START timestamp of runs/kaggle/<job>/<job>.log. Sessions that started since the
last weekly reset, Saturday 00:00 UTC, are counted. The boundary is not exposed by the CLI; it was checked
against the account's quota page on 2026-11-14 (28 h 12 m of 30 h available: jobV1's 1.65 h after the reset
counted, jobV0 pushed 22:15Z the evening before did not). Kernels whose log has no start timestamp take it
from scripts/kaggle/session_starts.json (push times), else from the log file's modification time. The next queue entry is estimated as the longest past session
of the same arm (else the longest session overall); a kernel still running (--running) is counted
with the same estimate. The quota itself (default 30 h) is an assumption, not read from Kaggle.

Verdict: "ok" if used + running + next <= quota - margin (2 h), else "wait".
"""
import json, re, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import next_job  # noqa: E402

MARGIN_S = 2 * 3600
STARTS = Path(__file__).resolve().parent / "session_starts.json"
OTHERS = Path(__file__).resolve().parent / "other_sessions.json"   # non-TRACE kernels on the account (other_usage.py)


def week_start(now):
    """Most recent Saturday 00:00 UTC at or before `now` (Kaggle's weekly GPU quota reset)."""
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return day - timedelta(days=(day.weekday() - 5) % 7)
START_RE = re.compile(r"^JOB-START (\S+) arm=(\S+).*?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s*$", re.M)


def parse_time(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def sessions(kaggle_dir, starts=None):
    """One entry per downloaded kernel. Kernels without a JOB-START line or with a differently cased log name
    (the VGCure kernels: jobV0/trace-jobv0.log) still used quota: they are counted with arm None and the log
    file's modification time (download time, later than the real start, so they stay in the window longer)."""
    out = []
    if starts is None:
        starts = json.loads(STARTS.read_text()) if STARTS.exists() else {}
    for d in sorted(Path(kaggle_dir).glob("job*")):
        job = d.name
        logs = sorted(p for p in d.glob("trace-*.log") if p.name.lower() == "trace-%s.log" % job.lower())
        if not logs:
            continue
        times = [e["time"] for e in json.loads(logs[0].read_text()) if "time" in e]
        if not times:
            continue
        job_log = d / ("%s.log" % job)
        m = START_RE.search(job_log.read_text(errors="replace")) if job_log.exists() else None
        if m is not None:
            arm, start = m.group(2), m.group(3)
        elif job in starts:
            arm, start = None, starts[job]
        else:
            arm = None
            start = datetime.fromtimestamp(logs[0].stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append({"job": job, "arm": arm, "start_utc": start, "session_seconds": round(max(times), 1)})
    return out


def other_sessions(path=None):
    path = OTHERS if path is None else Path(path)
    return json.loads(path.read_text()) if path.exists() else []


def check(kaggle_dir, queue_path, root, now, quota_hours=30.0, running=None, starts=None, others=None):
    past = sessions(kaggle_dir, starts)
    window = [s for s in past if week_start(now) <= parse_time(s["start_utc"]) <= now]
    # Other projects' kernels share the weekly quota; counted in `used`, never used for TRACE estimates.
    other_window = [s for s in (other_sessions() if others is None else others)
                    if week_start(now) <= parse_time(s["start_utc"]) <= now]
    other_used = sum(s["session_seconds"] for s in other_window)
    used = sum(s["session_seconds"] for s in window) + other_used

    def estimate(arm):
        same = [s["session_seconds"] for s in past if arm is not None and s["arm"] == arm]
        pool = same or [s["session_seconds"] for s in past]
        return max(pool) if pool else 2 * 3600.0

    jobs = next_job.load(queue_path)
    pending = [j for j in jobs if not (Path(root) / j["out"] / "summary.json").exists()]
    by_name = {j["job"]: j for j in jobs}
    run_est = 0.0
    if running is not None and running not in {s["job"] for s in past}:
        run_est = estimate(by_name[running]["arm"]) if running in by_name else estimate(None)
    nxt = next((j for j in pending if j["job"] != running), None)
    next_est = estimate(nxt["arm"]) if nxt else 0.0
    quota = quota_hours * 3600
    need = used + run_est + next_est
    rest = [j for j in pending if j["job"] != running]
    queue_total = used + run_est + sum(estimate(j["arm"]) for j in rest)
    return {"now_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "quota_hours_assumed": quota_hours,
            "week_start_utc": week_start(now).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sessions_in_window": window, "other_sessions_in_window": len(other_window),
            "other_used_hours": round(other_used / 3600, 2), "used_hours": round(used / 3600, 2),
            "running": running, "running_estimate_hours": round(run_est / 3600, 2),
            "next": nxt["job"] if nxt else None, "next_estimate_hours": round(next_est / 3600, 2),
            "remaining_queue_hours_if_all_this_week": round(queue_total / 3600, 2),
            "verdict": "ok" if need <= quota - MARGIN_S else "wait"}


def main(argv):
    args = list(argv)

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    now = parse_time(opt("--now", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
    res = check(opt("--kaggle-dir", ROOT / "runs/kaggle"), opt("--queue", ROOT / "scripts/kaggle/queue.json"),
                opt("--root", ROOT), now, float(opt("--quota-hours", 30)), opt("--running", None))
    if "--json" in args:
        print(json.dumps(res, indent=1))
    else:
        print("used %.2f h of an assumed %.0f h since the weekly reset %s, as of %s (%d sessions; incl. %.2f h"
              " from %d other-project kernels)" % (
            res["used_hours"], res["quota_hours_assumed"], res["week_start_utc"], res["now_utc"],
            len(res["sessions_in_window"]), res["other_used_hours"], res["other_sessions_in_window"]))
        print("running %s ~%.2f h; next %s ~%.2f h; whole queue this week ~%.2f h -> %s" % (
            res["running"], res["running_estimate_hours"], res["next"], res["next_estimate_hours"],
            res["remaining_queue_hours_if_all_this_week"], res["verdict"]))
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
