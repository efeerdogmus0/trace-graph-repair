"""Run the locked-test report scripts in their fixed order (CPU only, no model).

Usage: python scripts/final_reports.py [--out-dir reports/final] [--queue f] [--dry-run]

Order (RUNBOOK): final_table, final_primary (the deciding six-cell statistic, added 2026-11-15T06:10Z), final_invariance, final_pairs, final_density, final_slices, final_metrics, final_floor.
Each script is all-or-nothing on its own; this runner stops at the first one that refuses (non-zero exit) and
does not start the next, so nothing after the main table is opened while the main table is refused. Each script
writes <out-dir>/<name>.{json,md,tex}; their stdout (the markdown tables) is not echoed by this runner, it only
prints which script ran and where the files are. --dry-run prints the commands and runs nothing.
Fixed before any locked-test number was read (DECISIONS 2026-11-15T01:30Z)."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["final_table", "final_primary", "final_invariance", "final_pairs", "final_density", "final_slices", "final_metrics", "final_floor"]


def commands(out_dir, queue=None):
    out = []
    for name in ORDER:
        cmd = [sys.executable, str(ROOT / "scripts" / (name + ".py"))]
        for ext in ("json", "md", "tex"):
            cmd += ["--out-" + ext, str(Path(out_dir) / ("%s.%s" % (name, ext)))]
        if queue:
            cmd += ["--queue", str(queue)]
        out.append((name, cmd))
    return out


def run(out_dir, queue=None, dry_run=False):
    """Returns (finished script names, name of the script that refused or None)."""
    done = []
    for name, cmd in commands(out_dir, queue):
        if dry_run:
            print(" ".join(cmd)); done.append(name); continue
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
            print("%s refused (exit %d): %s" % (name, r.returncode, tail[0]))
            print("stopped; not run: %s" % ", ".join(ORDER[len(done) + 1:]) if len(done) + 1 < len(ORDER) else "stopped")
            return done, name
        print("%s ok -> %s/%s.{json,md,tex}" % (name, out_dir, name))
        done.append(name)
    return done, None


def main(argv):
    args = list(argv)
    opts = {"out-dir": str(ROOT / "reports/final"), "queue": None}
    for key in list(opts):
        if "--" + key in args:
            i = args.index("--" + key); opts[key] = args[i + 1]; del args[i:i + 2]
    dry = "--dry-run" in args
    if dry:
        args.remove("--dry-run")
    if args:
        raise SystemExit(__doc__)
    _, refused = run(opts["out-dir"], opts["queue"], dry)
    if refused:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
