"""Turn a downloaded Kaggle kernel output into the small tracked files the analysis scripts use (CPU only).

Usage: python scripts/kaggle/collect.py <job> <arm> [--seed N] [--mode image] [--out runs/pilot3_<arm>] [--kernel-dir DIR] [--force] [--budget]
Expects `kaggle kernels output yiitefeerdomu/trace-<job> -p runs/kaggle/<job>` to have been run, so that
runs/kaggle/<job>/<job>.log and runs/kaggle/<job>/runs/pilot3_<arm>/ exist (runs/kaggle/ is not tracked).

Writes <out>/summary.json (train summary + eval metrics, copied from files, not transcribed) and
<out>/flags_<suffix>.json. The flags are recomputed here from the downloaded predictions file with the
local manifest, and must equal the FLAGS-SUMMARY line in the kernel log (same predictions sha256, same
bits); any mismatch stops the script. Prints one budget.jsonl line; appends it with --budget."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MACHINE_KEYS = ("images_digest",)
TRAIN_DIR = {"E": "pilot3_C_desc"}


def parse_args(argv):
    opts = {"seed": "3407", "mode": "image", "out": None, "kernel_dir": None, "force": False, "budget": False}
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--seed", "--mode", "--out", "--kernel-dir"):
            opts[a[2:].replace("-", "_")] = argv[i + 1]; i += 2; continue
        if a in ("--force", "--budget"):
            opts[a[2:]] = True; i += 1; continue
        pos.append(a); i += 1
    if len(pos) != 2:
        raise SystemExit(__doc__)
    return pos[0], pos[1], opts


def log_lines(log_text, tag):
    out = []
    for line in log_text.splitlines():
        if line.startswith(tag + " "):
            out.append(json.loads(line[len(tag) + 1:]))
    return out


def flags_from_predictions(run_dir, roots, mode):
    cmd = [sys.executable, str(ROOT / "scripts/export_flags.py"), str(ROOT / "manifests/pilot.json"), "dev",
           str(roots), str(run_dir), mode]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"),
                         "PATH": "/usr/bin:/bin"})
    if res.returncode != 0:
        raise SystemExit("export_flags failed:\n" + res.stderr[-2000:])
    return log_lines(res.stdout, "FLAGS-SUMMARY")[-1]


def main(argv):
    job, arm, opts = parse_args(argv)
    kdir = Path(opts["kernel_dir"]) if opts["kernel_dir"] else ROOT / "runs/kaggle" / job
    name = "pilot3_" + arm + ("" if opts["seed"] == "3407" else "_s" + opts["seed"])   # same rule as job.sh
    run_dir = kdir / "runs" / name
    # Arm E (job_e.sh) trains the C_desc parser in runs/pilot3_C_desc and evaluates parse-and-solve in
    # runs/pilot3_E, so its train summary lives next to the parser, not next to the predictions.
    seed_suffix = "" if opts["seed"] == "3407" else "_s" + opts["seed"]   # job_e.sh: parser dir carries the seed too
    train_dir = kdir / "runs" / (TRAIN_DIR[arm] + seed_suffix if arm in TRAIN_DIR else name)
    log_path = kdir / (job + ".log")
    for p in (train_dir / "train_summary.json", log_path):
        if not p.exists():
            raise SystemExit("missing %s (download the kernel output first)" % p)
    suffix = "dev" if opts["mode"] == "image" else "dev_" + opts["mode"].replace("+", "_")
    metrics_path = run_dir / ("metrics_%s.json" % suffix)
    if not metrics_path.exists():
        raise SystemExit("missing %s: the evaluation did not finish" % metrics_path)
    out = Path(opts["out"]) if opts["out"] else ROOT / "runs" / name
    if not out.is_absolute():
        out = ROOT / out
    if (out / "summary.json").exists() and not opts["force"]:
        raise SystemExit("%s/summary.json exists; pass --force to overwrite" % out)

    log_text = log_path.read_bytes().decode("utf-8", errors="replace")
    train = json.loads(train_path_text(train_dir))
    metrics = json.loads(metrics_path.read_text())
    roots = metrics.get("eval_root_graphs") or metrics.get("roots")
    logged = [f for f in log_lines(log_text, "FLAGS-SUMMARY") if f.get("input_mode", "image") == opts["mode"]]
    if not logged:
        raise SystemExit("no FLAGS-SUMMARY line for mode %s in %s" % (opts["mode"], log_path))
    logged = logged[-1]
    roots = roots or logged["eval_root_graphs"]
    local = flags_from_predictions(run_dir, roots, opts["mode"])
    local["run"] = logged["run"]
    # Image bytes are rendered per machine (DECISIONS: content digest identical, image bytes differ), so the
    # images digest is recorded, not compared; everything else (content digest, predictions, bits) must match.
    diff = sorted(k for k in set(local) | set(logged) if k not in MACHINE_KEYS and local.get(k) != logged.get(k))
    if diff:
        raise SystemExit("local flags differ from the kernel's FLAGS-SUMMARY in: %s" % diff)

    env = [l for l in log_text.splitlines() if l.startswith("peft ") or "Tesla" in l or "JOB-" in l][:6]
    summary = {"source": "copied from Kaggle kernel output yiitefeerdomu/trace-%s (files, not transcription); "
                         "flags re-scored locally from the downloaded predictions and equal to the log line" % job,
               "kernel_environment_lines": env, "train_dir": train_dir.name, "train": train, "eval_" + suffix: metrics,
               "images_digest": {"kernel": logged.get("images_digest"), "local": local.get("images_digest")}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / ("flags_%s.json" % suffix)).write_text(json.dumps(logged) + "\n")
    gpu_s = round(float(train.get("train_seconds", 0)) + float(metrics.get("eval_seconds", 0)), 1)
    budget = {"run_id": "%s_%s" % (name, job), "platform": "kaggle", "device": train.get("gpu"),
              "gpu_seconds": gpu_s, "cash_cost": 0,
              "status": "complete; gpu_seconds = train_seconds + eval_seconds (setup/data generation not counted)"}
    print(json.dumps(budget))
    if opts["budget"]:
        with open(ROOT / "runs/budget.jsonl", "a") as fh:
            fh.write(json.dumps(budget) + "\n")
    print("wrote", out / "summary.json", out / ("flags_%s.json" % suffix))


def train_path_text(run_dir):
    return (run_dir / "train_summary.json").read_text()


if __name__ == "__main__":
    main(sys.argv[1:])
