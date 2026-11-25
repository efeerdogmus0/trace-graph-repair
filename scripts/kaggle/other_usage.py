"""Record GPU sessions of the account's other (non-TRACE) Kaggle kernels, so quota.py counts them.

Usage: python scripts/kaggle/other_usage.py [--dry-run]

The Kaggle account is shared with other projects (since 2026-11-19 the owner's belirledim.tv voice jobs push
GPU kernels too). Kaggle bills all of them against the same weekly GPU quota, but quota.py only read TRACE's
own logs in runs/kaggle, so it undercounted. This script lists the account's kernels (`kaggle kernels list
--mine`), and for every kernel whose slug does not start with "trace-" downloads only its session log
(`--file-pattern '\\.log$'`, a few kB) and records the largest "time" field (session seconds) under
(kernel, lastRunTime). Entries are merged into scripts/kaggle/other_sessions.json and never removed, so a
kernel re-run later in the week adds a new entry. Limitations: only a kernel's latest version is visible, so a
version that was replaced before this script ran is missed (run it before every gate check); every non-TRACE
kernel is counted as a GPU session (conservative); lastRunTime is taken as the session start.
"""
import json, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "other_sessions.json"
KAGGLE = str(Path.home() / ".kaggle/venv/bin/kaggle")
TIME_RE = re.compile(r'"time"\s*:\s*([0-9.]+)')


def session_seconds(log_text):
    ts = [float(x) for x in TIME_RE.findall(log_text)]
    return round(max(ts), 1) if ts else None


def to_utc(last_run):
    # CLI prints "2026-11-20 08:53:41.173000" (UTC)
    return last_run.split(".")[0].replace(" ", "T") + "Z"


def load(path=OUT):
    return json.loads(Path(path).read_text()) if Path(path).exists() else []


def merge(old, new):
    keys = {(e["kernel"], e["start_utc"]) for e in old}
    return old + [e for e in new if (e["kernel"], e["start_utc"]) not in keys]


def list_kernels():
    r = subprocess.run([KAGGLE, "kernels", "list", "--mine", "--page-size", "100", "--csv"],
                       capture_output=True, text=True, check=True)
    rows = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 4 and "/" in parts[0]:
            rows.append((parts[0], parts[-2]))
    return rows


def fetch(ref, last_run):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([KAGGLE, "kernels", "output", ref, "-p", d, "--file-pattern", r"\.log$", "-q"],
                       capture_output=True, text=True)
        logs = sorted(Path(d).glob("*.log"))
        secs = session_seconds(logs[0].read_text(errors="replace")) if logs else None
    return {"kernel": ref, "start_utc": to_utc(last_run), "session_seconds": secs, "source": "kaggle log"}


def main(argv):
    old = load()
    seen = {(e["kernel"], e["start_utc"]) for e in old}
    new = []
    for ref, last_run in list_kernels():
        if ref.split("/")[-1].startswith("trace-"):
            continue
        if (ref, to_utc(last_run)) in seen:
            continue
        e = fetch(ref, last_run)
        if e["session_seconds"] is None:   # no log yet (still running?): not stored, picked up on a later run
            print("%s %s no log yet, not recorded" % (ref, e["start_utc"]))
            continue
        new.append(e)
    for e in new:
        print("%s %s %.0f s (%s)" % (e["kernel"], e["start_utc"], e["session_seconds"], e["source"]))
    print("%d new, %d total" % (len(new), len(old) + len(new)))
    if "--dry-run" not in argv:
        OUT.write_text(json.dumps(merge(old, new), indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
