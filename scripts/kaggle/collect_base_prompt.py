"""Collect a frozen-base prompt-variant kernel (make_kernel.py <job> A <roots>:image) into tracked files (CPU only).

Usage: python scripts/kaggle/collect_base_prompt.py <job> [--kernel-dir DIR] [--out-root runs] [--force] [--budget]
Expects `kaggle kernels output yiitefeerdomu/trace-<job> -p runs/kaggle/<job>`, i.e. <job>.log and
runs/base_prompt_<variant>/ for every variant of trace_vr/base_prompt.py.

All-or-nothing: every check below runs for all variants before anything is written.
- the log has JOB-START and JOB-END for this job (the kernel finished);
- each variant has predictions, metrics and the token-budget scan; the metrics name that variant, no adapter, split dev,
  the local manifest's content digest, and the same roots / samples / token limit as the other variants;
- strict per-sample flags are recomputed here from the downloaded predictions and must equal the kernel's FLAGS-SUMMARY
  line of that run (images digest recorded, not compared: image bytes differ per machine);
- the token-budget scan describes the same evaluation, and the rule of base_prompt.choose applied to the downloaded
  scans gives the variant, limit and scores of the kernel's BASE-PROMPT line.

Writes <out-root>/base_prompt_<variant>/{summary.json, flags_dev.json, token_budget_dev.json}. The flags are scored at
the long run limit (what export_flags.py sees); the selection uses the scan at each variant's own limit, so the two
macros can differ. Prints one budget.jsonl line (sum of the variants' eval_seconds); appends it with --budget.
Afterwards: python scripts/base_prompt_choice.py runs --out reports/pilot/base_prompt.json --out-md reports/pilot/base_prompt.md
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trace_vr import base_prompt  # noqa: E402

MACHINE_KEYS = ("images_digest",)
SAME_KEYS = ("eval_root_graphs", "samples", "max_new_tokens", "content_digest")


def parse_args(argv):
    opts = {"kernel_dir": None, "out_root": None, "force": False, "budget": False}
    pos, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("--kernel-dir", "--out-root"):
            opts[a[2:].replace("-", "_")] = argv[i + 1]; i += 2; continue
        if a in ("--force", "--budget"):
            opts[a[2:]] = True; i += 1; continue
        pos.append(a); i += 1
    if len(pos) != 1:
        raise SystemExit(__doc__)
    return pos[0], opts


def log_lines(log_text, tag):
    return [json.loads(l[len(tag) + 1:]) for l in log_text.splitlines() if l.startswith(tag + " ")]


def local_flags(run_dir, roots):
    cmd = [sys.executable, str(ROOT / "scripts/export_flags.py"), str(ROOT / "manifests/pilot.json"), "dev",
           str(roots), str(run_dir), "image"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    if res.returncode != 0:
        raise SystemExit("export_flags failed for %s:\n%s" % (run_dir, res.stderr[-2000:]))
    return log_lines(res.stdout, "FLAGS-SUMMARY")[-1]


def check(job, kdir):
    """Everything that will be written, or SystemExit. Nothing is written here."""
    log_path = kdir / (job + ".log")
    if not log_path.exists():
        raise SystemExit("missing %s (download the kernel output first)" % log_path)
    log_text = log_path.read_bytes().decode("utf-8", errors="replace")
    lines = log_text.splitlines()
    if not any(l.startswith("JOB-START %s " % job) and "base_prompt" in l for l in lines):
        raise SystemExit("%s has no base-prompt JOB-START line for %s" % (log_path, job))
    if not any(l.startswith("JOB-END %s " % job) for l in lines):
        raise SystemExit("%s has no JOB-END line: the kernel did not finish" % log_path)
    digest = json.loads((ROOT / "manifests/pilot.json").read_text())["content_digest"]
    logged_flags = {f["run"]: f for f in log_lines(log_text, "FLAGS-SUMMARY")}   # last line per run wins
    per, scans = {}, {}
    for v in base_prompt.VARIANTS:
        name = base_prompt.run_name(v)
        run = kdir / "runs" / name
        paths = {k: run / k for k in ("predictions_dev.jsonl", "metrics_dev.json", "token_budget_dev.json")}
        for p in paths.values():
            if not p.exists():
                raise SystemExit("missing %s: all variants are needed" % p)
        metrics = json.loads(paths["metrics_dev.json"].read_text())
        want = {"prompt_variant": v, "adapter": "none", "split": "dev", "input_mode": "image", "sample_set": "main",
                "content_digest": digest}
        bad = sorted(k for k, x in want.items() if metrics.get(k) != x)
        if bad:
            raise SystemExit("%s/metrics_dev.json is not the %s frozen-base dev run: %s differ" % (name, v, bad))
        scan = json.loads(paths["token_budget_dev.json"].read_text())
        pairs = (("split", "split"), ("eval_root_graphs", "eval_root_graphs"), ("samples", "samples"),
                 ("content_digest", "content_digest"), ("run_limit", "max_new_tokens"))
        bad = sorted(a for a, b in pairs if scan.get(a) != metrics.get(b))
        if bad:
            raise SystemExit("%s: token-budget scan and metrics describe different evaluations: %s" % (name, bad))
        if name not in logged_flags:
            raise SystemExit("no FLAGS-SUMMARY line for %s in %s" % (name, log_path))
        logged = logged_flags[name]
        local = local_flags(run, metrics["eval_root_graphs"])
        diff = sorted(k for k in set(local) | set(logged) if k not in MACHINE_KEYS and local.get(k) != logged.get(k))
        if diff:
            raise SystemExit("%s: local flags differ from the kernel's FLAGS-SUMMARY in: %s" % (name, diff))
        per[v] = {"name": name, "metrics": metrics, "scan": scan, "flags": logged,
                  "images_digest": {"kernel": logged.get("images_digest"), "local": local.get("images_digest")}}
        scans[v] = scan
    for k in SAME_KEYS:
        if len({repr(per[v]["metrics"].get(k)) for v in base_prompt.VARIANTS}) != 1:
            raise SystemExit("variants were not run on the same evaluation: %s differs" % k)
    try:
        choice = base_prompt.choose(scans)
    except ValueError as e:
        raise SystemExit(str(e))
    logged_choice = log_lines(log_text, "BASE-PROMPT")
    if not logged_choice:
        raise SystemExit("no BASE-PROMPT line in %s" % log_path)
    mine = {"chosen": choice["chosen"], "limit": choice["chosen_limit"],
            "macro": {v: e["non_edit_macro"] for v, e in choice["variants"].items()}}
    if logged_choice[-1] != mine:
        raise SystemExit("the kernel's BASE-PROMPT line %s differs from the rule applied to the downloaded scans %s"
                         % (json.dumps(logged_choice[-1]), json.dumps(mine)))
    env = [l for l in lines if l.startswith("peft ") or "Tesla" in l or l.startswith("JOB-")][:6]
    device = "Tesla T4" if any("Tesla T4" in l for l in lines) else None   # None = not stated in the log
    return per, choice, env, device


def main(argv):
    job, opts = parse_args(argv)
    kdir = Path(opts["kernel_dir"]) if opts["kernel_dir"] else ROOT / "runs/kaggle" / job
    out_root = Path(opts["out_root"]) if opts["out_root"] else ROOT / "runs"
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    per, choice, env, device = check(job, kdir)
    existing = [str(out_root / e["name"] / "summary.json") for e in per.values()
                if (out_root / e["name"] / "summary.json").exists()]
    if existing and not opts["force"]:
        raise SystemExit("%s exists; pass --force to overwrite" % ", ".join(existing))
    for v, e in per.items():
        out = out_root / e["name"]
        out.mkdir(parents=True, exist_ok=True)
        summary = {"source": "copied from Kaggle kernel output yiitefeerdomu/trace-%s (files, not transcription); flags "
                             "re-scored locally from the downloaded predictions and equal to the log line" % job.lower(),
                   "arm": "A", "prompt_variant": v, "kernel_environment_lines": env, "eval_dev": e["metrics"],
                   "images_digest": e["images_digest"],
                   "note": "flags_dev.json is scored at the run limit; the variant choice uses token_budget_dev.json "
                           "at the variant's own limit"}
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        (out / "flags_dev.json").write_text(json.dumps(e["flags"]) + "\n")
        (out / "token_budget_dev.json").write_text(json.dumps(e["scan"], indent=2) + "\n")
    gpu_s = round(sum(float(e["metrics"]["eval_seconds"]) for e in per.values()), 1)
    budget = {"run_id": "base_prompt_%s" % job, "platform": "kaggle", "device": device, "gpu_seconds": gpu_s,
              "cash_cost": 0, "status": "complete; sum of eval_seconds of the %d prompt variants of the frozen base "
                                        "(setup/data generation not counted)" % len(per)}
    print(json.dumps(budget))
    if opts["budget"]:
        with open(ROOT / "runs/budget.jsonl", "a") as fh:
            fh.write(json.dumps(budget) + "\n")
    print("chosen %s at %d new tokens; wrote %s" % (choice["chosen"], choice["chosen_limit"],
                                                    ", ".join(str(out_root / e["name"]) for e in per.values())))


if __name__ == "__main__":
    main(sys.argv[1:])
