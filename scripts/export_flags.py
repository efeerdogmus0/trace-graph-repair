"""Print per-sample strict scores of a finished evaluation as one short line.

Runs on the training machine, next to predictions_<split>.jsonl; the printed FLAGS-SUMMARY
line is saved as runs/<run>/flags_<split>.json on the analysis machine.
Usage: python scripts/export_flags.py <manifest> <split> <n_roots> <run_dir> [input_mode] [sample_set]"""
import hashlib, json, sys
from pathlib import Path
from trace_vr import arms, bootstrap, dataset

manifest, split, n_roots, run_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
input_mode = sys.argv[5] if len(sys.argv) > 5 else "image"
sample_set = sys.argv[6] if len(sys.argv) > 6 else "main"
dataset.check_split_allowed(split, "eval")
suffix = arms.run_suffix(split, input_mode, sample_set)
man, _, by_split, _ = dataset.load_dataset(Path(manifest))
rows = arms.select_samples(by_split[split], n_roots, sample_set)
pred_path = run_dir / ("predictions_%s.jsonl" % suffix)
raw = pred_path.read_bytes()
responses = {}
for line in raw.decode("utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        responses[r["sample_id"]] = r["text"]
out = bootstrap.score_flags(rows, responses)
out.update({"run": run_dir.name, "split": split, "input_mode": input_mode, "sample_set": sample_set, "eval_root_graphs": n_roots, "content_digest": man["content_digest"],
            "images_digest": man["images_digest"], "predictions_sha256": hashlib.sha256(raw).hexdigest()})
print("FLAGS-SUMMARY " + json.dumps(out), flush=True)
