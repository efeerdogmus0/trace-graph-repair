"""Record sha256 and size of every downloaded Kaggle artifact (CPU only, reads files, writes one JSON).

Usage: python scripts/kaggle/artifact_manifest.py [--check]
runs/kaggle/ (adapters, predictions, logs) is not tracked by git; this writes the tracked
runs/kaggle_artifacts.json so a later release can prove that the adapter and prediction files it ships are
the ones the reported numbers came from. Checkpoint folders (ckpt/) are skipped.
Each predictions file is cross-checked against the predictions_sha256 in the tracked flags file the
collector wrote (runs/<run>/flags_*.json); a mismatch stops the script. VGCure predictions
(predictions_vgcure*.jsonl, external evaluation) have no flags file and are only hashed.
Locked-test kernels (jobFxx in final_queue.json): once a job's output is downloaded, final_problems() requires
all twelve predictions files, each tied to the flags file collect_final.py wrote in that entry's dev_dir, and the
evaluated adapter copy to be byte-identical to the source training kernel's adapter; anything else stops the script.
The queue's "base" job (arm A, frozen base) is checked the same way once downloaded, against runs/base_prompt_<variant>
from the dev choice file; it has no adapter, and adapter files in its output are a problem.
--check: recompute and compare with the existing manifest instead of writing it (exit 1 on any difference)."""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_final as cf  # noqa: E402  (SPLITS, SAMPLE_SETS, suffix_of: the twelve evaluations of a final kernel)
KAGGLE = ROOT / "runs/kaggle"
OUT = ROOT / "runs/kaggle_artifacts.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flag_hashes(root=ROOT):
    """predictions_sha256 -> tracked flags file, over every runs/*/flags_*.json."""
    out = {}
    for p in sorted((root / "runs").glob("*/flags_*.json")):
        sha = json.loads(p.read_text()).get("predictions_sha256")
        if sha:
            out.setdefault(sha, []).append(str(p.relative_to(root)))
    return out


def build(kaggle=KAGGLE, root=ROOT):
    flags = flag_hashes(root)
    files = []
    for p in sorted(kaggle.rglob("*")):
        rel = p.relative_to(kaggle)
        if not p.is_file() or "ckpt" in rel.parts or "__pycache__" in rel.parts:
            continue
        entry = {"path": str(rel), "bytes": p.stat().st_size, "sha256": sha256(p)}
        if p.name.startswith("predictions_vgcure") and p.suffix == ".jsonl":
            entry["external"] = "vgcure"   # external evaluation: scored by eval_vgcure.py, no flags file exists
        elif p.name.startswith("predictions_") and p.suffix == ".jsonl":
            entry["flags_files"] = flags.get(entry["sha256"], [])
        files.append(entry)
    return {"generated_by": "scripts/kaggle/artifact_manifest.py", "root": "runs/kaggle", "files": files}


def unmatched_predictions(manifest):
    return [f["path"] for f in manifest["files"] if "flags_files" in f and not f["flags_files"]]


def final_problems(manifest, queue, base_entry=None):
    """(downloaded final jobs, problems) for the locked-test queue entries that have any file in the manifest.

    A downloaded final job must be complete: twelve predictions files, each matching flags_<suffix>.json in the
    entry's dev_dir (so it was collected by collect_final.py), and the adapter it evaluated (the parser for arm E)
    must have the sha256 recorded for the source training kernel."""
    by_path = {f["path"]: f for f in manifest["files"]}
    present, problems = [], []
    entries = list(queue["jobs"])
    base = queue.get("base")
    if base and any(p.startswith(base["job"] + "/") for p in by_path):
        # arm A (frozen base): the entry exists only through the dev choice file (base_entry: a ready entry, for tests)
        try:
            entries.append(base_entry or cf.base_entry())
        except SystemExit as err:
            present.append(base["job"])
            problems.append("%s: downloaded, but the dev choice cannot be read (%s)" % (base["job"], err))
    for e in entries:
        job = e["job"]
        if not any(p.startswith(job + "/") for p in by_path):
            continue
        present.append(job)
        for _, split, _ in cf.SPLITS:
            for sample_set in cf.SAMPLE_SETS:
                suffix = cf.suffix_of(split, sample_set)
                pred = by_path.get("%s/runs/%s/predictions_%s.jsonl" % (job, e["run"], suffix))
                want = "%s/flags_%s.json" % (e["dev_dir"], suffix)
                if pred is None:
                    problems.append("%s: predictions_%s.jsonl missing" % (job, suffix))
                elif want not in pred.get("flags_files", []):
                    problems.append("%s: predictions_%s.jsonl does not match %s (run collect_final.py)" % (job, suffix, want))
        if e["arm"] == "A":   # no adapter was evaluated; none may have been produced
            stray = [p for p in by_path if p.startswith(job + "/") and "/adapter/" in p]
            if stray:
                problems.append("%s: frozen-base kernel output contains adapter files: %s" % (job, stray[:3]))
            continue
        adapter_run = e.get("parser", e["run"])
        tail = "runs/%s/adapter/adapter_model.safetensors" % adapter_run
        copy, source = by_path.get("%s/%s" % (job, tail)), by_path.get("%s/%s" % (e["source"], tail))
        if copy is None or source is None:
            problems.append("%s: adapter %s missing in %s" % (job, tail, job if copy is None else e["source"]))
        elif copy["sha256"] != source["sha256"]:
            problems.append("%s: evaluated adapter %s differs from the one trained by %s" % (job, adapter_run, e["source"]))
    return present, problems


def main(argv):
    m = build()
    bad = unmatched_predictions(m)
    if bad:
        raise SystemExit("predictions without a matching tracked flags file: %s" % bad)
    queue = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())
    present, problems = final_problems(m, queue)
    if problems:
        raise SystemExit("locked-test kernel output incomplete or inconsistent:\n  " + "\n  ".join(problems))
    compared = [j for j in present if j in {e["job"] for e in queue["jobs"]}]   # the frozen base is counted apart
    base = (queue.get("base") or {}).get("job")
    print("final (locked-test) kernels downloaded and consistent: %d of %d%s" % (
        len(compared), len(queue["jobs"]), "; frozen base %s %s" % (base, "in" if base in present else "not yet") if base else ""))
    if "--check" in argv:
        old = json.loads(OUT.read_text())
        if old != m:
            print("manifest differs from runs/kaggle_artifacts.json"); return 1
        print("ok: %d files match" % len(m["files"])); return 0
    OUT.write_text(json.dumps(m, indent=1) + "\n")
    print("wrote %s: %d files, %d bytes" % (OUT.relative_to(ROOT), len(m["files"]), sum(f["bytes"] for f in m["files"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
