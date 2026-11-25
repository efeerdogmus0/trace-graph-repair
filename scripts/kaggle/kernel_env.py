"""Read the ENV-SUMMARY line (library versions, CUDA, GPUs) from downloaded Kaggle kernel logs.

Usage: python scripts/kaggle/kernel_env.py <job> [<job> ...] [--kdir runs/kaggle] [--out reports/final/kernel_env.json]

Final kernels built after 2026-11-16 print one ENV-SUMMARY line (make_final_kernel.ENV_PROBE) into the kernel's
own log, `runs/kaggle/<job>/trace-<job>.log`. This script reads only that line: no score, flag or summary is
parsed, so it is safe to run while the locked test is still closed. Kernels built earlier have no such line and
are listed as `not recorded`; nothing is guessed for them. Exit status 1 if the recorded environments differ
from each other in any field (a Kaggle image update between kernels), 0 otherwise.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAG = "ENV-SUMMARY "


def read_env(log_path):
    """The ENV-SUMMARY dict of one kernel log, or None if the kernel did not print one."""
    text = Path(log_path).read_text(errors="replace")
    try:  # `kaggle kernels output` saves the log as a JSON list of {stream_name, time, data}
        lines = "".join(e.get("data", "") for e in json.loads(text)).splitlines()
    except ValueError:
        lines = text.splitlines()
    found = [json.loads(l[len(TAG):]) for l in lines if l.startswith(TAG)]
    if len(found) > 1:
        raise SystemExit("%s: %d ENV-SUMMARY lines, expected one" % (log_path, len(found)))
    return found[0] if found else None


def differing_fields(envs):
    """Fields whose value is not the same in every recorded environment (None entries are skipped)."""
    rec = [e for e in envs.values() if e is not None]
    keys = sorted(set().union(*rec)) if rec else []
    return [k for k in keys if len({json.dumps(e.get(k), sort_keys=True) for e in rec}) > 1]


def main(argv):
    kdir, out, jobs, it = ROOT / "runs/kaggle", None, [], iter(argv)
    for a in it:
        if a == "--kdir":
            kdir = Path(next(it))
        elif a == "--out":
            out = Path(next(it))
        elif a.startswith("--"):
            raise SystemExit(__doc__)
        else:
            jobs.append(a)
    if not jobs:
        raise SystemExit(__doc__)
    envs = {}
    for job in jobs:
        log = kdir / job / ("trace-%s.log" % job.lower())
        if not log.exists():
            raise SystemExit("missing %s (run `kaggle kernels output` first)" % log)
        envs[job] = read_env(log)
    diff = differing_fields(envs)
    result = {"jobs": envs, "not_recorded": [j for j, e in envs.items() if e is None], "differing_fields": diff}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for job, e in envs.items():
        print(job, "not recorded" if e is None else json.dumps(e, sort_keys=True))
    print("differing fields:", diff or "none")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
