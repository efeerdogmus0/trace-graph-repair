"""Use the benchmark with any model: export the prompts, then score a predictions file (CPU only).

No model code is needed on either side. The export holds only what a model may see (image path and
instruction text, through dataset.to_model_example); labels never leave the scorer.

  python scripts/benchmark_io.py export <manifest> <split> --out prompts.jsonl [--roots N]
      one JSON line per sample: {"sample_id", "task", "image", "prompt"}; "image" is a path
      relative to the dataset directory printed on stderr (data/<name>/).

  python scripts/benchmark_io.py score <manifest> <split> <predictions.jsonl> [--roots N]
                                       [--name RUN] [--out report.json] [--flags flags.json] [--boot 2000]
      predictions: one JSON line per sample, {"sample_id": ..., "text": <raw model output>}.
      A sample without a line counts as wrong (and is counted as missing); an id that is not in
      the exported set, or a repeated id, is an error. Scoring is strict: the answer must be the
      single JSON object the instruction asks for. The report holds the per-task accuracy, the
      non-edit macro (mean of reachability and critical edge), the lenient diagnostic, twin-pair
      counts (both twins right) and 95% intervals from resampling root graphs. --flags writes the
      per-sample file that scripts/bootstrap_ci.py and scripts/breakdown.py read, so two models
      can be compared as a paired difference.

Locked test splits (names starting with "test") stay refused unless TRACE_UNLOCK_TEST is set, as
everywhere else in this repository; use dev while developing a model.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from trace_vr import analysis, arms, bootstrap, dataset, metrics


def _rows(manifest, split, roots):
    dataset.check_split_allowed(split, "eval")
    man, data_dir, by_split, _ = dataset.load_dataset(Path(manifest))
    if split not in by_split:
        raise SystemExit("split %r not in %s (has: %s)" % (split, manifest, ", ".join(sorted(by_split))))
    total = len({s["graph_group_id"] for s in by_split[split]})
    if roots is None:
        roots = total
    if not 1 <= roots <= total:
        raise SystemExit("--roots must be between 1 and %d for split %r" % (total, split))
    return man, data_dir, arms.eval_samples(by_split[split], roots), roots


def export(args):
    man, data_dir, rows, roots = _rows(args.manifest, args.split, args.roots)
    lines = []
    for s in rows:
        ex = dataset.to_model_example(s, Path("."))
        lines.append(json.dumps({"sample_id": s["sample_id"], "task": s["task"], "image": ex["image"],
                                 "prompt": ex["prompt"]}, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("%d prompts from %d root graphs of %s/%s -> %s; images are relative to %s"
          % (len(lines), roots, man["name"], args.split, out, data_dir), file=sys.stderr)


def read_predictions(path, rows):
    known = {s["sample_id"] for s in rows}
    raw = Path(path).read_bytes()
    responses = {}
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            sid, text = rec["sample_id"], rec["text"]
        except (ValueError, KeyError, TypeError):
            raise SystemExit('%s line %d: expected {"sample_id": ..., "text": ...}' % (path, n))
        if sid not in known:
            raise SystemExit("%s line %d: sample_id %r is not in the exported set" % (path, n, sid))
        if sid in responses:
            raise SystemExit("%s line %d: sample_id %r appears twice" % (path, n, sid))
        if text is not None and not isinstance(text, str):
            raise SystemExit("%s line %d: text must be a string or null" % (path, n))
        responses[sid] = text
    return responses, hashlib.sha256(raw).hexdigest()


def score(args):
    man, _, rows, roots = _rows(args.manifest, args.split, args.roots)
    responses, sha = read_predictions(args.predictions, rows)
    payload = bootstrap.score_flags(rows, responses)
    payload.update({"run": args.name or Path(args.predictions).stem, "split": args.split, "input_mode": "image",
                    "sample_set": "main", "eval_root_graphs": roots, "content_digest": man["content_digest"],
                    "images_digest": man["images_digest"], "predictions_sha256": sha})
    flags = bootstrap.decode_flags(rows, payload)
    report = {
        "dataset": man["name"], "split": args.split, "root_graphs": roots, "run": payload["run"],
        "content_digest": man["content_digest"], "images_digest": man["images_digest"],
        "predictions_sha256": sha,
        "metrics": metrics.evaluate(rows, responses, lenient=True),
        "twin_pairs": analysis.pair_table(rows, flags),
        "intervals": bootstrap.cluster_bootstrap(rows, flags, n_boot=args.boot),
    }
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.flags:
        Path(args.flags).parent.mkdir(parents=True, exist_ok=True)
        Path(args.flags).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    m = report["metrics"]
    print("%s/%s, %d root graphs, %d samples, %d missing" % (man["name"], args.split, roots, m["samples"], m["missing_responses"]))
    for task, e in m["per_task"].items():
        print("  %-22s strict %.3f  lenient %.3f  format failures %.3f  (n=%d)"
              % (task, e["accuracy"], e["lenient_accuracy"], e["format_failure_rate"], e["n"]))
    macro = report["intervals"]["a"].get(bootstrap.MACRO)
    if m["non_edit_macro_accuracy"] is not None and macro:
        print("  non-edit macro %.3f [%.3f, %.3f] (95%%, root-graph bootstrap)"
              % (m["non_edit_macro_accuracy"], macro["lo"], macro["hi"]))
    for task, c in report["twin_pairs"].items():
        print("  twins %-16s both right %d of %d" % (task, c["both_right"], c["pairs"]))
    if not args.out:
        print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("manifest"); e.add_argument("split")
    e.add_argument("--out", required=True); e.add_argument("--roots", type=int)
    e.set_defaults(fn=export)
    s = sub.add_parser("score")
    s.add_argument("manifest"); s.add_argument("split"); s.add_argument("predictions")
    s.add_argument("--roots", type=int); s.add_argument("--name"); s.add_argument("--out")
    s.add_argument("--flags"); s.add_argument("--boot", type=int, default=2000)
    s.set_defaults(fn=score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
