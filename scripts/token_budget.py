"""Pick max_new_tokens per arm from one long dev run (protocol items 1 and 5).

Usage: python scripts/token_budget.py <manifest> <split> <n_roots> <predictions.jsonl> <run_limit> [out.json]

The predictions must come from scripts/eval_arm.py (they carry token ids). Only the
tokenizer is loaded (from the local Hugging Face cache, no download, no model weights).
"""
import json, sys
from pathlib import Path
from trace_vr import arms, dataset, token_budget

manifest, split, n_roots, pred_path, run_limit = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
out_path = Path(sys.argv[6]) if len(sys.argv) > 6 else pred_path.with_name(pred_path.stem + "_token_budget.json")
dataset.check_split_allowed(split, "eval")
man, _, by_split, _ = dataset.load_dataset(Path(manifest))
samples = arms.eval_samples(by_split[split], n_roots)
rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
from transformers import AutoTokenizer
from trace_vr.model import BASE_ID, BASE_REVISION
tok = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REVISION, local_files_only=True)
res = token_budget.scan(samples, rows, run_limit, lambda ids: tok.decode(list(ids), skip_special_tokens=True))
res.update({"manifest": manifest, "split": split, "eval_root_graphs": n_roots, "samples": len(samples),
            "predictions": str(pred_path), "content_digest": man["content_digest"]})
out_path.write_text(json.dumps(res, indent=2))
print("TOKEN-BUDGET " + json.dumps({"chosen": res["chosen_limit"], "by_limit": {k: [round(v["max_truncation_rate"], 3), v["non_edit_macro"], v["non_edit_macro_lenient"]] for k, v in res["limits"].items()}}))
