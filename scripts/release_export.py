"""Build the public code release as a clean snapshot of HEAD (local only; it pushes nothing).

usage: python scripts/release_export.py --out <dir> [--author "Name <email>"] [--test] [--into <clone>]

Writes <out>/trace-graph-repair/ from `git archive HEAD`, minus the files listed in PRIVATE: personal or
working notes that are not part of the research record (autopilot state, plans, the original task prompt, notes
about earlier chats, a drafted personal message, the log of the owner's other Kaggle projects). DECISIONS.md and
RUNBOOK.md stay: they are the dated decision log and the reproduction instructions the paper and README rely on.
The snapshot is committed as a single fresh commit (so `scripts/kaggle/eval_freeze.py`, which hashes blob ids,
works in it) under an explicit identity: --author, else the anonymous DEFAULT_AUTHOR (also fit for double-blind
review). It never falls back to the machine's git identity, which on this Mac is <user>@<hostname>.local. `--test` then runs the release audit and the full test
suite inside the snapshot at low priority and reports both. The source tree is not modified.
"""
import io, os, re, shutil, subprocess, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "trace-graph-repair"
PRIVATE = ("PROJECT_STATE.md", "PLAN.md", "BLOCKERS.md", "TRACE_ASTRA_MASTER_PROMPT.md", "docs/prior_gpt_work.md",
           "docs/endorsement_request.md", "scripts/kaggle/other_sessions.json")
DEFAULT_AUTHOR = "TRACE authors <trace-authors@example.invalid>"


def git(*args, cwd=ROOT, **kw):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, **kw)


def identity_env(author):
    m = re.fullmatch(r"\s*([^<>]+?)\s*<([^<>@\s]+@[^<>\s]+)>\s*", author)
    if not m:
        raise SystemExit("--author must look like 'Name <email>'")
    name, email = m.groups()
    if email.endswith(".local"):
        raise SystemExit("refusing a machine-local e-mail (%s) as the release author" % email)
    env = dict(os.environ)
    env.update(GIT_AUTHOR_NAME=name, GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_NAME=name, GIT_COMMITTER_EMAIL=email)
    return env


def export(out, author=DEFAULT_AUTHOR):
    dst = out / NAME
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    data = git("archive", "--format=tar", "HEAD").stdout
    kept = 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name not in PRIVATE:
                tf.extract(m, dst)
                kept += 1
    missing = [p for p in PRIVATE if not (ROOT / p).exists()]
    if missing:
        raise SystemExit("PRIVATE lists files that no longer exist: %s" % missing)
    git("init", "-q", "-b", "main", cwd=dst)
    git("add", "-A", cwd=dst)
    head = git("rev-parse", "--short", "HEAD").stdout.decode().strip()
    git("commit", "-q", "-m", "TRACE: verified drawn-graph benchmark, pre-registered repair-transfer study (snapshot of %s)" % head,
        cwd=dst, env=identity_env(author))
    return dst, kept


def commit_into(clone, dst, author):
    """Commit the snapshot `dst` on top of an existing clone of the public repo (no push, no history rewrite).

    The clone's index is matched to the snapshot's tree (added, changed and removed files), then committed under
    `author`. Returns the new commit id, or None when the snapshot equals the clone's HEAD."""
    clone = Path(clone).resolve()
    gd = ["--git-dir=%s" % (clone / ".git"), "--work-tree=%s" % dst]
    git(*gd, "add", "-A", cwd=dst)
    if not git(*gd, "status", "--porcelain", cwd=dst).stdout.strip():
        return None
    snap = git("log", "-1", "--format=%s", cwd=dst).stdout.decode().strip()
    git(*gd, "commit", "-q", "-m", "Update: " + snap, cwd=dst, env=identity_env(author))
    gone = git("diff", "--name-only", "--diff-filter=D", "HEAD~1", "HEAD", cwd=clone).stdout.decode().split("\n")
    for rel in filter(None, gone):                    # files the commit removed; nothing else is touched
        (clone / rel).unlink(missing_ok=True)
    git("checkout", "-q", "HEAD", "--", ".", cwd=clone)  # bring the clone's tracked files to the new commit
    return git("rev-parse", "--short", "HEAD", cwd=clone).stdout.decode().strip()


def main(argv):
    if "--out" not in argv:
        raise SystemExit(__doc__)
    out = Path(argv[argv.index("--out") + 1]).resolve()
    if out == ROOT or ROOT in out.parents and not str(out.relative_to(ROOT)).startswith("dist"):
        raise SystemExit("write the release outside the repo or under dist/")
    author = argv[argv.index("--author") + 1] if "--author" in argv else DEFAULT_AUTHOR
    dst, kept = export(out, author)
    print("release snapshot %s: %d files, author %s (left out: %s)" % (dst, kept, author, ", ".join(PRIVATE)))
    if "--test" in argv:
        py = str(ROOT / ".venv/bin/python")
        env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}
        a = subprocess.run([py, "scripts/release_audit.py", "--out-md", "/dev/null"], cwd=dst, capture_output=True, text=True, env=env)
        print((a.stdout + a.stderr).strip().splitlines()[-1] if (a.stdout + a.stderr).strip() else "audit: no output")
        t = subprocess.run(["nice", "-n", "19", py, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=dst,
                           capture_output=True, text=True, env=env)
        fails = [l for l in t.stdout.splitlines() if l.startswith("FAILED")]
        print("suite in snapshot: exit %d%s" % (t.returncode, "".join("\n  " + f for f in fails)))
        if t.returncode or a.returncode:
            return t.returncode or a.returncode
    if "--into" in argv:
        c = commit_into(argv[argv.index("--into") + 1], dst, author)
        print("committed %s on the clone (not pushed)" % c if c else "clone already equals the snapshot; nothing committed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
