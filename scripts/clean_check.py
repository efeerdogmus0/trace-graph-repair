"""Clean-checkout reproduction check (release checklist row "reproduced in a clean environment").

Clones the committed HEAD into a fresh directory (no untracked data, runs or adapters),
rebuilds the datasets the tests read from their configs, compares the rebuilt content
digest with the tracked manifest, restores the tracked manifests, and runs the full test
suite with TRACE_REQUIRE_DATA=1 so no data-dependent test can pass by skipping.

    python scripts/clean_check.py --workdir <empty scratch dir> [--out reports/clean_check.json]

The content digest is the reproducibility criterion (it covers samples, graphs, answers and
image metadata). The images digest covers PNG bytes, which differ between machines through
font rasterisation; it is reported, not required. CPU only; run under nice.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("smoke", "pilot", "pilot_test", "pilot_diag")


def run(cmd, cwd, env=None):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, errors="replace")
    return p, round(time.time() - t0, 1)


def manifest_digests(path):
    m = json.loads(Path(path).read_text())
    return m["content_digest"], m["images_digest"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="empty directory for the clone")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    work = Path(a.workdir)
    if work.exists() and any(work.iterdir()):
        sys.exit("workdir must be empty: %s" % work)
    clone = work / "clone"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    p, _ = run(["git", "clone", "-q", str(ROOT), str(clone)], cwd=ROOT)
    if p.returncode:
        sys.exit("clone failed: " + p.stderr)
    subprocess.check_call(["git", "checkout", "-q", head], cwd=clone)

    env = dict(os.environ, PYTHONPATH="src")
    report = {"head": head, "python": platform.python_version(), "platform": platform.platform(),
              "datasets": {}, "ok": True}
    for name in a.datasets.split(","):
        tracked = manifest_digests(clone / "manifests" / (name + ".json"))
        p, secs = run([a.python, "-m", "trace_vr", "generate", "--config", "configs/data_%s.yaml" % name], clone, env)
        rebuilt = manifest_digests(clone / "manifests" / (name + ".json")) if p.returncode == 0 else (None, None)
        row = {"generate_exit": p.returncode, "seconds": secs,
               "content_digest_tracked": tracked[0], "content_digest_rebuilt": rebuilt[0],
               "content_match": rebuilt[0] == tracked[0],
               "images_match": rebuilt[1] == tracked[1]}
        report["datasets"][name] = row
        report["ok"] &= row["generate_exit"] == 0 and row["content_match"]
    # The generator rewrites manifests; tests must see the tracked ones.
    subprocess.check_call(["git", "checkout", "-q", "--", "manifests"], cwd=clone)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=clone, text=True)
    report["tracked_files_unchanged"] = dirty.strip() == ""
    report["ok"] &= report["tracked_files_unchanged"]

    env_t = dict(env, TRACE_REQUIRE_DATA="1")
    p, secs = run([a.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs"], clone, env_t)
    tail = [l for l in p.stdout.splitlines() if l.strip()][-1:] or [""]
    report["pytest"] = {"exit": p.returncode, "seconds": secs, "summary": tail[0],
                        "skipped_lines": [l for l in p.stdout.splitlines() if l.startswith("SKIPPED")]}
    report["ok"] &= p.returncode == 0

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
