"""Check that the code a locked-test (final) kernel runs is the code the first final kernel ran.

Usage: python scripts/kaggle/eval_freeze.py [--rev REV] [--record]   -> prints one line, exit 1 on a change

The twelve final kernels are built from HEAD at different times (two quota weeks), so their payloads
differ (new reporting scripts and tests are added in between). That is harmless only if every file that
can change a test prediction or its scoring is byte-identical across the kernels. This script hashes the
git blob ids of exactly those paths (FROZEN_PATHS) at a revision and compares the digest with the one
recorded in scripts/kaggle/eval_freeze.json, which was taken at the commit jobF01's payload was built from
(found by rebuilding payloads and matching jobF01's recorded payload sha256; DECISIONS 2026-11-15T06:00Z).
`--record` writes the file; it refuses to overwrite an existing record.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "scripts" / "kaggle" / "eval_freeze.json"
# everything final_eval.sh executes or reads: the package, the data/eval configs and manifests, the
# evaluation entry points, and the kernel templates (version pins, adapter check)
FROZEN_PATHS = ["pyproject.toml", "src/trace_vr", "configs", "manifests", "scripts/colab/final_eval.sh",
                "scripts/eval_arm.py", "scripts/eval_parse_solve.py", "scripts/export_flags.py",
                "scripts/kaggle/make_kernel.py"]


def tree_lines(rev: str, root: Path = ROOT) -> list:
    out = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", rev, "--"] + FROZEN_PATHS,
                         check=True, capture_output=True, text=True).stdout
    lines = sorted("%s %s" % (l.split("\t", 1)[1], l.split()[2]) for l in out.splitlines() if l.strip())
    if not lines:
        raise SystemExit("no frozen files at %s" % rev)
    return lines


def digest(rev: str, root: Path = ROOT) -> str:
    return hashlib.sha256("\n".join(tree_lines(rev, root)).encode()).hexdigest()


def changed_paths(ref: str, rev: str, root: Path = ROOT) -> list:
    a, b = dict(l.rsplit(" ", 1) for l in tree_lines(ref, root)), dict(l.rsplit(" ", 1) for l in tree_lines(rev, root))
    return sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p))


def check(rev: str = "HEAD", record: Path = RECORD, root: Path = ROOT):
    rec = json.loads(Path(record).read_text())
    d = digest(rev, root)
    if d == rec["digest"]:
        return True, "eval code frozen: %s matches %s (%s) -> ok" % (rev, rec["commit"], d[:12])
    diff = changed_paths(rec["commit"], rev, root)
    return False, "eval code CHANGED since %s: %s -> stop" % (rec["commit"], ", ".join(diff[:8]) or "digest differs")


def main(argv):
    rev = argv[argv.index("--rev") + 1] if "--rev" in argv else "HEAD"
    if "--record" in argv:
        if RECORD.exists():
            raise SystemExit("%s exists; the freeze is recorded once" % RECORD)
        full = subprocess.run(["git", "-C", str(ROOT), "rev-parse", rev], check=True, capture_output=True, text=True).stdout.strip()
        RECORD.write_text(json.dumps({"commit": full, "digest": digest(full), "paths": FROZEN_PATHS,
                                      "note": "commit jobF01's payload was built from; see eval_freeze.py"}, indent=1) + "\n")
        print("recorded", full, digest(full))
        return 0
    ok, msg = check(rev)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
