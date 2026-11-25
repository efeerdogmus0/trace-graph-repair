"""Check that the pre-registered rules are still the ones in force when the locked test was opened.

Usage: python scripts/protocol_freeze.py            -> one line per item, exit 1 if a rule changed
       python scripts/protocol_freeze.py --record   -> write docs/protocol_freeze.json (once, needs git)

The locked test was opened by the first final kernel (jobF01), whose payload was built from the commit in
scripts/kaggle/eval_freeze.json. eval_freeze.py already pins the evaluation *code* to that commit. This
script pins the *rules* a reader has to take on trust:

- sections 1-6 of docs/protocol.md (hypothesis, systems, data, contract, metrics and interpretation
  rules, controls): sha256 of the text from "## 1." up to "## 7.";
- the rule and choice files outside the frozen code paths (RULE_FILES): locked-test queue, reference
  baseline choice, frozen-base prompt choice, external subset and its protocol.

The record holds the values at the opening commit. The check recomputes them from the working tree
without git (a git blob id is sha1 of "blob <size>\\0" + bytes), so it also runs on a released copy.
Sections 7 and 8 of the protocol (open items, version record) may be edited; sections 1-6 may not.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "protocol_freeze.json"
PROTOCOL = "docs/protocol.md"
START, END = "## 1. ", "## 7. "
RULE_FILES = ["scripts/kaggle/final_queue.json", "reports/pilot/reference_baseline.json",
              "reports/pilot/base_prompt.json", "reports/external/vgcure_subset.json", "docs/external_eval.md"]


def rules_text(text: str) -> str:
    """Sections 1-6 of the protocol: from the first line starting with START up to the line starting with END."""
    lines = text.splitlines(keepends=True)
    a = [i for i, l in enumerate(lines) if l.startswith(START)]
    b = [i for i, l in enumerate(lines) if l.startswith(END)]
    if len(a) != 1 or len(b) != 1 or a[0] >= b[0]:
        raise SystemExit("protocol sections 1-6 not found exactly once")
    return "".join(lines[a[0]:b[0]])


def rules_sha(text: str) -> str:
    return hashlib.sha256(rules_text(text).encode()).hexdigest()


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git(root: Path, *args) -> bytes:
    return subprocess.run(["git", "-C", str(root)] + list(args), check=True, capture_output=True).stdout


def record_at(commit: str, root: Path = ROOT) -> dict:
    full = _git(root, "rev-parse", commit).decode().strip()
    date = _git(root, "log", "-1", "--format=%cI", full).decode().strip()
    proto = _git(root, "show", "%s:%s" % (full, PROTOCOL))
    return {"commit": full, "commit_date": date,
            "protocol_sections_1_to_6_sha256": rules_sha(proto.decode()),
            "protocol_blob": blob_id(proto),
            "files": {f: blob_id(_git(root, "show", "%s:%s" % (full, f))) for f in RULE_FILES},
            "note": "values at the commit jobF01's payload was built from (scripts/kaggle/eval_freeze.json); "
                    "see scripts/protocol_freeze.py"}


def check(record: Path = RECORD, root: Path = ROOT):
    rec = json.loads(Path(record).read_text())
    lines, ok = [], True
    same = rules_sha((root / PROTOCOL).read_text()) == rec["protocol_sections_1_to_6_sha256"]
    ok &= same
    lines.append("%s sections 1-6: %s" % (PROTOCOL, "unchanged" if same else "CHANGED"))
    for f, blob in sorted(rec["files"].items()):
        p = root / f
        same = p.is_file() and blob_id(p.read_bytes()) == blob
        ok &= same
        lines.append("%s: %s" % (f, "unchanged" if same else ("CHANGED" if p.is_file() else "MISSING")))
    lines.append("rules as at %s (%s) -> %s" % (rec["commit"][:7], rec["commit_date"], "ok" if ok else "stop"))
    return ok, lines


def main(argv):
    if "--record" in argv:
        if RECORD.exists():
            raise SystemExit("%s exists; the freeze is recorded once" % RECORD)
        commit = json.loads((ROOT / "scripts" / "kaggle" / "eval_freeze.json").read_text())["commit"]
        RECORD.write_text(json.dumps(record_at(commit), indent=1) + "\n")
        print("recorded", commit)
        return 0
    ok, lines = check()
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
