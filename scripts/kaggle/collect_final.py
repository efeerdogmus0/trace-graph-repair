"""Turn a downloaded locked-test (final) kernel output into tracked flags files (CPU only, no model).

Usage: python scripts/kaggle/collect_final.py <jobFxx> [--kernel-dir DIR] [--out DIR] [--force] [--budget]
The job must be an entry of scripts/kaggle/final_queue.json, or its "base" job (arm A, the frozen base: run and
dev_dir are runs/base_prompt_<variant> from reports/pilot/base_prompt.json, and every metrics file must report no
adapter, that variant and that token limit); run
`kaggle kernels output yiitefeerdomu/trace-<job> -p runs/kaggle/<job>` first, so that
runs/kaggle/<job>/<job>.log and runs/kaggle/<job>/runs/<run>/ exist (runs/kaggle/ is not tracked).

final_eval.sh scores six locked splits, each on its main samples and on the invariance copies: twelve
evaluations. For every one of them this script recomputes the per-sample flags from the downloaded
predictions file with the local manifest and requires them to equal the kernel's FLAGS-SUMMARY line
(same predictions sha256, same bits, same content digest); the per-machine images digest is recorded, not
compared. Nothing is written unless all twelve are present, finished (FINAL-END in the log) and equal.

Writes <out>/flags_<split>[_inv].json and <out>/final_summary.json (metrics copied from files); <out>
defaults to the queue entry's dev_dir, next to the run's dev files. Prints one budget.jsonl line
(sum of the twelve eval_seconds); appends it with --budget.

This reads locked test labels, so it sets TRACE_UNLOCK_TEST itself: it only re-scores a finished final
evaluation and cannot produce a model output."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MACHINE_KEYS = ("images_digest",)
# (manifest, split, roots) exactly as SPLITS in scripts/colab/final_eval.sh (tests/test_kaggle_collect_final.py checks).
SPLITS = [("pilot_test", "test_id", 200), ("pilot_test", "test_style_ood", 200), ("pilot_test", "test_struct_ood", 200),
          ("pilot_test", "test_combined_ood", 100), ("pilot_diag", "test_family_ood", 100), ("pilot_diag", "test_size_ood", 100)]
SAMPLE_SETS = ("main", "invariance")


def suffix_of(split, sample_set):
    return split if sample_set == "main" else split + "_inv"   # arms.run_suffix for image mode


def base_entry(queue=ROOT / "scripts/kaggle/final_queue.json", choice=None):
    """The frozen base's (arm A) final kernel, or None when the queue has no "base" key.

    Run, dev_dir, prompt variant and token limit come from the dev choice file only (make_final_kernel.base_choice,
    the same reader the kernel builder uses); SystemExit while that file is missing or malformed."""
    base = json.loads(Path(queue).read_text()).get("base")
    if not base:
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_final_kernel as mfk
    variant, limit = mfk.base_choice(choice or ROOT / base["choice"])
    run = "base_prompt_%s" % variant
    return {"job": base["job"], "arm": base["arm"], "run": run, "dev_dir": "runs/" + run, "variant": variant, "limit": limit,
            "expect": {"adapter": "none", "prompt_variant": variant, "max_new_tokens": limit}}


def queue_entry(job, queue=ROOT / "scripts/kaggle/final_queue.json", choice=None):
    q = json.loads(Path(queue).read_text())
    for e in q["jobs"]:
        if e["job"] == job:
            return e
    if q.get("base", {}).get("job") == job:
        return base_entry(queue, choice)
    raise SystemExit("%s is not in %s" % (job, queue))


def log_lines(log_text, tag):
    return [json.loads(l[len(tag) + 1:]) for l in log_text.splitlines() if l.startswith(tag + " ")]


def local_flags(manifest, split, roots, run_dir, sample_set, root=ROOT):
    cmd = [sys.executable, str(root / "scripts/export_flags.py"), str(root / "manifests" / (manifest + ".json")), split,
           str(roots), str(run_dir), "image", sample_set]
    res = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                         env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin", "TRACE_UNLOCK_TEST": "final-evaluation"})
    if res.returncode != 0:
        raise SystemExit("export_flags failed for %s %s:\n%s" % (split, sample_set, res.stderr[-2000:]))
    return log_lines(res.stdout, "FLAGS-SUMMARY")[-1]


def collect(job, run, kdir, out, splits=SPLITS, force=False, expect=None):
    """Check all evaluations, then write. Returns (budget_line, written_paths).

    expect: metric values every one of the evaluations must report (arm A: no adapter, the chosen prompt variant and
    token limit), so a kernel run with another prompt or limit than the dev choice is refused."""
    kdir, out = Path(kdir), Path(out)
    run_dir, log_path = kdir / "runs" / run, kdir / (job + ".log")
    if not log_path.exists():
        raise SystemExit("missing %s (download the kernel output first)" % log_path)
    log_text = log_path.read_bytes().decode("utf-8", errors="replace")
    if not any(l.startswith("FINAL-END " + job) for l in log_text.splitlines()):
        raise SystemExit("no FINAL-END line for %s in %s: the final evaluation did not finish" % (job, log_path))
    if (out / "final_summary.json").exists() and not force:
        raise SystemExit("%s/final_summary.json exists; pass --force to overwrite" % out)
    logged_all = log_lines(log_text, "FLAGS-SUMMARY")
    checked, metrics_all, eval_s = [], {}, 0.0
    for manifest, split, roots in splits:
        for sample_set in SAMPLE_SETS:
            suffix = suffix_of(split, sample_set)
            metrics_path = run_dir / ("metrics_%s.json" % suffix)
            if not metrics_path.exists():
                raise SystemExit("missing %s: this evaluation did not finish" % metrics_path)
            logged = [f for f in logged_all if f.get("split") == split and f.get("sample_set", "main") == sample_set
                      and f.get("input_mode", "image") == "image"]
            if not logged:
                raise SystemExit("no FLAGS-SUMMARY line for %s %s in %s" % (split, sample_set, log_path))
            logged = logged[-1]
            if logged.get("eval_root_graphs") != roots:
                raise SystemExit("%s %s scored %s roots, expected %s" % (split, sample_set, logged.get("eval_root_graphs"), roots))
            local = local_flags(manifest, split, roots, run_dir, sample_set)
            local["run"] = logged["run"]
            diff = sorted(k for k in set(local) | set(logged) if k not in MACHINE_KEYS and local.get(k) != logged.get(k))
            if diff:
                raise SystemExit("%s %s: local flags differ from the kernel's FLAGS-SUMMARY in: %s" % (split, sample_set, diff))
            metrics = json.loads(metrics_path.read_text())
            bad = sorted(k for k, x in (expect or {}).items() if metrics.get(k) != x)
            if bad:
                raise SystemExit("%s %s: metrics do not match the queue entry in %s (want %s)"
                                 % (split, sample_set, bad, json.dumps({k: expect[k] for k in bad}, sort_keys=True)))
            metrics_all[suffix] = metrics
            eval_s += float(metrics.get("eval_seconds", 0))
            checked.append((suffix, logged, local.get("images_digest")))
    env = [l for l in log_text.splitlines() if l.startswith("peft ") or "Tesla" in l or l.startswith("FINAL-") or l.startswith("adapter ok")
           or l.startswith("no adapter")][:8]
    summary = {"source": "copied from Kaggle kernel output yiitefeerdomu/trace-%s (files, not transcription); every flags "
                         "file re-scored locally from the downloaded predictions and equal to the log line" % job.lower(),
               "job": job, "run": run, "kernel_environment_lines": env, "required_metric_values": expect or {},
               "evaluations": [s for s, _, _ in checked], "metrics": metrics_all,
               "images_digest": {s: {"kernel": l.get("images_digest"), "local": d} for s, l, d in checked}}
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, logged, _ in checked:
        p = out / ("flags_%s.json" % suffix)
        p.write_text(json.dumps(logged) + "\n"); written.append(p)
    (out / "final_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    written.append(out / "final_summary.json")
    device = "Tesla T4" if any("Tesla T4" in l for l in log_text.splitlines()) else None   # None = not stated in the log
    budget = {"run_id": "%s_%s" % (run, job), "platform": "kaggle", "device": device, "gpu_seconds": round(eval_s, 1),
              "cash_cost": 0, "status": "complete; final locked-test evaluation, gpu_seconds = sum of %d eval_seconds "
                                        "(setup/data generation not counted)" % len(checked)}
    return budget, written


def main(argv):
    opts = {"kernel_dir": None, "out": None, "force": False, "budget": False}
    pos, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("--kernel-dir", "--out"):
            opts[a[2:].replace("-", "_")] = argv[i + 1]; i += 2; continue
        if a in ("--force", "--budget"):
            opts[a[2:]] = True; i += 1; continue
        pos.append(a); i += 1
    if len(pos) != 1:
        raise SystemExit(__doc__)
    job = pos[0]
    entry = queue_entry(job)
    kdir = Path(opts["kernel_dir"]) if opts["kernel_dir"] else ROOT / "runs/kaggle" / job
    out = Path(opts["out"]) if opts["out"] else ROOT / entry["dev_dir"]
    if not out.is_absolute():
        out = ROOT / out
    budget, written = collect(job, entry["run"], kdir, out, force=opts["force"], expect=entry.get("expect"))
    print(json.dumps(budget))
    if opts["budget"]:
        with open(ROOT / "runs/budget.jsonl", "a") as fh:
            fh.write(json.dumps(budget) + "\n")
    print("wrote %d files in %s" % (len(written), out))


if __name__ == "__main__":
    main(sys.argv[1:])
