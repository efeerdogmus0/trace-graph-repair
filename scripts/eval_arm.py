"""Evaluate base or adapter. Usage: python scripts/eval_arm.py <manifest> <split> <n_roots> <run_dir> <adapter_dir|none> [max_new] [input_mode] [sample_set] [prompt_variant]

input_mode: image (default) | image+edges | edges. The last two give the true graph as text
(oracle diagnostic) and write predictions_<split>_<mode>.jsonl / metrics_<split>_<mode>.json.
sample_set: main (default) | invariance (relabelled/restyled copies; files get suffix _inv).
prompt_variant: common (default) | direct | system; the last two are for the frozen base only
(trace_vr/base_prompt.py) and must be given their own run_dir."""
import json, sys, time
from pathlib import Path
import torch
from trace_vr import arms, base_prompt, dataset, metrics, model as M, tasks

manifest, split, n_roots, run_dir, adapter = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
max_new = int(sys.argv[6]) if len(sys.argv) > 6 else 96
input_mode = sys.argv[7] if len(sys.argv) > 7 else "image"
sample_set = sys.argv[8] if len(sys.argv) > 8 else "main"
prompt_variant = sys.argv[9] if len(sys.argv) > 9 else base_prompt.COMMON
base_prompt.check_allowed(prompt_variant, adapter)
if input_mode not in dataset.INPUT_MODES:
    raise SystemExit("unknown input_mode %r" % input_mode)
dataset.check_split_allowed(split, "eval")
suffix = arms.run_suffix(split, input_mode, sample_set)
run_dir.mkdir(parents=True, exist_ok=True)
man, data_dir, by_split, _ = dataset.load_dataset(Path(manifest))
rows = arms.select_samples(by_split[split], n_roots, sample_set)
model, processor, device = M.load()
if adapter != "none":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
model.eval()
responses, t0, trunc, new_tokens = {}, time.time(), 0, 0
with open(run_dir / ("predictions_%s.jsonl" % suffix), "w") as fh:
    for i, s in enumerate(rows):
        ex = dataset.to_model_example(s, data_dir, input_mode=input_mode)
        system, prompt = base_prompt.apply(prompt_variant, ex["prompt"])
        extra = {} if system is None else {"system": system}
        out = M.generate(model, processor, M.build_inference_inputs(processor, ex["image"], prompt, **extra), max_new)
        responses[s["sample_id"]] = out["text"]
        trunc += out["hit_limit"]; new_tokens += out["new_tokens"]
        fh.write(json.dumps({"sample_id": s["sample_id"], "task": s["task"], "text": out["text"], "new_tokens": out["new_tokens"],
                             "hit_limit": out["hit_limit"], "ids": out["ids"]}) + "\n")
        if i % 100 == 0:
            print("eval %d/%d %.0fs" % (i, len(rows), time.time() - t0), flush=True)
res = metrics.evaluate(rows, responses, lenient=True)
if sample_set == "invariance":  # evaluate() scores root/edit images only; score the copies directly
    inv: dict = {}
    for s in rows:
        inv.setdefault(s["task"], []).append(metrics.score_sample(s, responses.get(s["sample_id"]))["correct"])
    res["invariance_copy_accuracy"] = {t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in inv.items()}
summary = {"manifest": manifest, "split": split, "eval_root_graphs": n_roots, "samples": len(rows), "adapter": adapter, "max_new_tokens": max_new, "input_mode": input_mode, "sample_set": sample_set, "prompt_variant": prompt_variant,
           "content_digest": man["content_digest"], "images_digest": man["images_digest"], "eval_seconds": round(time.time() - t0, 1),
           "truncated_at_limit": int(trunc), "mean_new_tokens": new_tokens / max(1, len(rows)), "metrics": res}
(run_dir / ("metrics_%s.json" % suffix)).write_text(json.dumps(summary, indent=2))
short = {t: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()} for t, e in res["per_task"].items()}
print("EVAL-SUMMARY " + json.dumps({"adapter": adapter, "input_mode": input_mode, "n": len(rows), "truncated": int(trunc), "seconds": summary["eval_seconds"], "per_task": short,
                                    "macro": res["non_edit_macro_accuracy"], "macro_lenient": res.get("non_edit_macro_lenient_accuracy"),
                                    "pairs": res["pair_accuracy"], "sample_set": sample_set, "prompt_variant": prompt_variant,
                                    "invariance_copy_accuracy": res.get("invariance_copy_accuracy")}), flush=True)
