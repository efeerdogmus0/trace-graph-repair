"""Dress rehearsal of the locked-test reporting path on synthetic flags (no locked-test number is read).

    python scripts/rehearse_final_reports.py --workdir <empty scratch dir> [--out reports/final_rehearsal.json]
                                             [--boot 200] [--tectonic PATH]

final_reports.py has so far only been run while it refuses (not all final kernels are collected), so its
eight scripts have never run end to end on the full queue, the full split list and the real file layout.
This script does that in a throwaway clone of HEAD:

1. clone HEAD into <workdir>/repo and point its data/ at this checkout's data/ (inputs only);
2. for every queue entry and the "base" entry (arm A), DELETE whatever flags_test_*.json and
   final_summary.json the clone holds (the collected real ones included, unread) and write synthetic ones:
   per-sample answers are a fixed hash of (sample, arm, seed) -- true / false / prose -- scored against the
   labels exactly as collect_final.py would, in the same layout (twelve evaluations, metrics keyed by suffix,
   arm E with the parse-and-solve metrics blocks, arm A with the dev-chosen prompt fields);
3. run scripts/final_reports.py, scripts/final_figures.py, scripts/final_style_cards.py, scripts/result_summary.py and
   scripts/paper_finding.py and scripts/paper_conclusion.py in the clone, then compile paper/main.tex there (when tectonic is given) and
   check that no "Pending" line is left in the locked-test section.

Writes only the status record (--out): which scripts succeeded, the output files' sizes, and the compile
result. No table value is copied out, and the numbers inside the clone are synthetic by construction.
CPU only; run under nice."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL_TABLES = ("final_primary", "final_table", "final_pairs", "final_invariance", "final_density",
                "final_slices", "final_metrics", "final_floor")
ANSWERS = ('{"answer": true}', '{"answer": false}', 'I think {"answer": true}')


def answer(sample_id, arm, seed):
    h = int(hashlib.sha256(("%s|%s|%s" % (sample_id, arm, seed)).encode()).hexdigest()[:8], 16)
    return ANSWERS[h % len(ANSWERS)]


def write_run(repo, entry, splits, sample_sets, expect=None):
    """Synthetic collected output for one final kernel in the clone (layout of collect_final.py)."""
    sys.path.insert(0, str(repo / "src"))
    from trace_vr import arms, bootstrap, dataset, metrics, tasks
    d = repo / entry["dev_dir"]
    d.mkdir(parents=True, exist_ok=True)
    for p in list(d.glob("flags_test_*.json")) + [d / "final_summary.json"]:
        if p.exists():
            p.unlink()
    summary = {"synthetic": True, "job": entry["job"], "run": entry.get("run"), "metrics": {}, "evaluations": []}
    cache = {}
    for manifest, split, roots in splits:
        if manifest not in cache:
            cache[manifest] = dataset.load_dataset(repo / "manifests" / (manifest + ".json"))
        man, _, by_split, _ = cache[manifest]
        for sample_set in sample_sets:
            suffix = split if sample_set == "main" else split + "_inv"
            rows = arms.select_samples(by_split[split], roots, sample_set)
            resp = {s["sample_id"]: answer(s["sample_id"], entry["arm"], entry.get("seed", 0)) for s in rows}
            head = {"split": split, "sample_set": sample_set, "eval_root_graphs": roots,
                    "content_digest": man["content_digest"]}
            if entry["arm"] == "E":
                m = dict(head, truncated_at_limit=0, max_new_tokens=512, images=len(rows), eval_seconds=1.0,
                         metrics=metrics.evaluate(rows, resp),
                         metrics_lenient_parse=metrics.evaluate(rows, {k: tasks.lenient_candidate(v) for k, v in resp.items()}),
                         extraction_strict={"exact": 0.5, "format_ok": 1.0, "edge_precision": 0.9, "edge_recall": 0.8,
                                            "direction_accuracy": 0.7, "node_recall": 1.0})
            else:
                m = dict(head, truncated_at_limit=0, max_new_tokens=96, eval_seconds=1.0,
                         metrics=metrics.evaluate(rows, resp, lenient=True))
                m.update(expect or {})
            summary["metrics"][suffix] = m
            summary["evaluations"].append(suffix)
            out = bootstrap.score_flags(rows, resp)
            out.update(head)
            if entry.get("run"):   # predictions file in the kernel-output layout, recorded in the flags (style cards read it)
                pred = repo / "runs" / "kaggle" / entry["job"] / "runs" / entry["run"] / ("predictions_%s.jsonl" % suffix)
                pred.parent.mkdir(parents=True, exist_ok=True)
                raw = "".join(json.dumps({"sample_id": s["sample_id"], "text": resp[s["sample_id"]]}) + "\n" for s in rows)
                pred.write_text(raw)
                out["predictions_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            (d / ("flags_%s.json" % suffix)).write_text(json.dumps(out))
    (d / "final_summary.json").write_text(json.dumps(summary))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tectonic", default=None)
    a = ap.parse_args(argv)
    work = Path(a.workdir).resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit("workdir must be empty: %s" % work)
    work.mkdir(parents=True, exist_ok=True)
    repo = work / "repo"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "clone", "--quiet", str(ROOT), str(repo)], check=True)
    if (repo / "data").exists():
        raise SystemExit("clone already has data/; expected it untracked")
    (repo / "data").symlink_to(ROOT / "data")
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), TRACE_UNLOCK_TEST="final-evaluation")
    os.environ["TRACE_UNLOCK_TEST"] = "final-evaluation"   # labels are needed to score the synthetic answers

    sys.path.insert(0, str(repo / "scripts" / "kaggle"))
    import collect_final
    queue = repo / "scripts/kaggle/final_queue.json"
    jobs = json.loads(queue.read_text())["jobs"]
    t0 = time.time()
    for e in jobs:
        write_run(repo, e, collect_final.SPLITS, collect_final.SAMPLE_SETS)
    base = collect_final.base_entry(queue, repo / json.loads(queue.read_text())["base"]["choice"])
    write_run(repo, base, collect_final.SPLITS, collect_final.SAMPLE_SETS, expect=base["expect"])
    synth_s = round(time.time() - t0, 1)

    t0 = time.time()
    p = subprocess.run([sys.executable, "scripts/final_reports.py", "--out-dir", "reports/final"], cwd=repo, env=env,
                       capture_output=True, text=True, errors="replace")
    rep = {"head": head, "synthetic_runs": len(jobs) + 1, "synthetic_seconds": synth_s,
           "final_reports_exit": p.returncode, "final_reports_seconds": round(time.time() - t0, 1),
           "final_reports_status": [l for l in p.stdout.splitlines() if " ok -> " in l or "refused" in l],
           "final_reports_stderr_tail": p.stderr.splitlines()[-15:], "outputs": {}}
    for f in sorted((repo / "reports/final").glob("*")) if (repo / "reports/final").exists() else []:
        if f.is_file():
            rep["outputs"][f.name] = f.stat().st_size
    # the figure script reads the table final_reports.py just wrote (added 2026-11-19)
    g = subprocess.run([sys.executable, "scripts/final_figures.py"], cwd=repo, env=env, capture_output=True, text=True,
                       errors="replace")
    rep["final_figures_exit"] = g.returncode
    rep["final_figures_stderr_tail"] = g.stderr.splitlines()[-5:]
    rep["figure_outputs"] = {str(f.relative_to(repo)): f.stat().st_size for f in
                             sorted((repo / "reports/final/figures").glob("*")) + [repo / "paper/figures/ood_test.pdf"]
                             if f.exists()}
    # style-pair cards read the predictions files behind the flags (added 2026-11-19)
    s = subprocess.run([sys.executable, "scripts/final_style_cards.py", "--out", "reports/final/style_cards.json",
                        "--out-md", "reports/final/style_cards.md", "--copy-images", "reports/final/style_cards"],
                       cwd=repo, env=env, capture_output=True, text=True, errors="replace")
    rep["style_cards_exit"] = s.returncode
    rep["style_cards_stderr_tail"] = s.stderr.splitlines()[-5:]
    rep["style_cards_images"] = len(list((repo / "reports/final/style_cards").glob("*.png")))
    # announcement draft / Turkish summary read only final_primary.json (added 2026-11-20)
    r = subprocess.run([sys.executable, "scripts/result_summary.py"], cwd=repo, env=env, capture_output=True, text=True,
                       errors="replace")
    rep["result_summary_exit"] = r.returncode
    rep["result_summary_stderr_tail"] = r.stderr.splitlines()[-5:]
    # abstract and intro finding paragraph from final_primary.json only (added 2026-11-20, rule finding-v1)
    f = subprocess.run([sys.executable, "scripts/paper_finding.py"], cwd=repo, env=env, capture_output=True, text=True,
                       errors="replace")
    rep["paper_finding_exit"] = f.returncode
    rep["paper_finding_stderr_tail"] = f.stderr.splitlines()[-5:]
    # conclusion body from final_primary.json only (added 2026-11-20, rule conclusion-v1)
    k = subprocess.run([sys.executable, "scripts/paper_conclusion.py"], cwd=repo, env=env, capture_output=True, text=True,
                       errors="replace")
    rep["paper_conclusion_exit"] = k.returncode
    rep["paper_conclusion_stderr_tail"] = k.stderr.splitlines()[-5:]
    if a.tectonic:
        outdir = work / "pdf"; outdir.mkdir()
        c = subprocess.run([a.tectonic, "-X", "compile", "main.tex", "--outdir", str(outdir)], cwd=repo / "paper",
                           capture_output=True, text=True, errors="replace")
        log = c.stdout + c.stderr
        rep["paper_compile_exit"] = c.returncode
        rep["paper_errors"] = [l for l in log.splitlines() if l.startswith("error") or "! " in l][:10]
        rep["paper_overfull"] = sum("Overfull" in l for l in log.splitlines())
        pdf = outdir / "main.pdf"
        rep["paper_pdf_bytes"] = pdf.stat().st_size if pdf.exists() else 0
        # the pending branch of \\finaltable is chosen at compile time; count it in the log-free way: the .tex
        # inputs exist for every table, so each \\IfFileExists takes the table branch
        rep["paper_pending_lines"] = (sum(not (repo / "reports/final" / (n + ".tex")).exists() for n in FINAL_TABLES)
                                      + (not (repo / "paper/figures/ood_test.pdf").exists()))
    rep["ok"] = (p.returncode == 0 and len(rep["outputs"]) == 24 and g.returncode == 0 and len(rep["figure_outputs"]) == 5
                 and s.returncode == 0 and rep["style_cards_images"] > 0 and r.returncode == 0 and f.returncode == 0 and k.returncode == 0
                 and rep.get("paper_compile_exit", 0) == 0 and not rep.get("paper_pending_lines"))
    text = json.dumps(rep, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text, end="")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
