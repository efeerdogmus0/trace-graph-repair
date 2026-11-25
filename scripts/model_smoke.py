"""M3 step 1: load the base model, verify the image path, run frozen-base inference on a few samples.

Usage: python scripts/model_smoke.py <manifest> <split> <n_root_graphs> <run_dir>
"""
import json, sys, time, platform
from pathlib import Path
import torch
from PIL import Image
from trace_vr import dataset, metrics, model as M, tasks

manifest, split, n_roots, run_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
dataset.check_split_allowed(split, "eval")
run_dir.mkdir(parents=True, exist_ok=True)
_, data_dir, samples_by_split, _ = dataset.load_dataset(Path(manifest))
rows = samples_by_split[split]
groups = []
for s in rows:
    if s["graph_group_id"] not in groups:
        groups.append(s["graph_group_id"])
keep = set(groups[:n_roots])
rows = [s for s in rows if s["graph_group_id"] in keep and s["metadata"]["image_variant"] in ("root", "edit")
        and s["task"] in (tasks.REPAIR, tasks.REACHABILITY, tasks.CRITICAL_EDGE, tasks.EDGE_EXISTS, tasks.EXTRACTION)]

t0 = time.time()
model, processor, device = M.load()
load_s = time.time() - t0
print("loaded on", device, model.dtype, "in %.1fs" % load_s, "| class", type(model).__name__, "| processor", type(processor).__name__)

# Check 1-2: image tokens exist, and a different image changes the inputs.
ex = dataset.to_model_example(rows[0], data_dir, with_target=True)
a = M.build_inference_inputs(processor, ex["image"], ex["prompt"])
other = next(s for s in rows if s["image_id"] != rows[0]["image_id"])
b = M.build_inference_inputs(processor, str(data_dir / other["image_path"]), ex["prompt"])
image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
n_img = int((a["input_ids"] == image_token).sum())
pix_key = [k for k in a.keys() if "pixel" in k][0]
checks = {
    "image_tokens_in_prompt": n_img,
    "prompt_tokens": int(a["input_ids"].shape[1]),
    "pixel_tensor_shape": list(a[pix_key].shape),
    "same_image_twice_identical": bool(torch.equal(a[pix_key], M.build_inference_inputs(processor, ex["image"], ex["prompt"])[pix_key])),
    "different_image_changes_pixels": bool(not torch.equal(a[pix_key], b[pix_key])),
    "different_image_same_text_tokens": bool(torch.equal(a["input_ids"], b["input_ids"])),
}
# Check 3: loss mask covers exactly the assistant answer.
full, n_prompt = M.build_training_inputs(processor, ex["image"], ex["prompt"], ex["target"])
supervised = full["input_ids"][0][full["labels"][0] != -100]
checks["supervised_text"] = processor.decode(supervised, skip_special_tokens=False)
checks["supervised_tokens"] = int(supervised.shape[0])
checks["target"] = ex["target"]
print(json.dumps(checks, indent=1))
assert n_img > 0 and checks["different_image_changes_pixels"] and checks["same_image_twice_identical"]
assert ex["target"] in checks["supervised_text"]

responses, log = {}, []
t1 = time.time()
for i, s in enumerate(rows):
    ex = dataset.to_model_example(s, data_dir)
    limit = 192 if s["task"] in (tasks.REPAIR, tasks.EXTRACTION) else 64
    t = time.time()
    out = M.generate(model, processor, M.build_inference_inputs(processor, ex["image"], ex["prompt"]), limit)
    out.update({"sample_id": s["sample_id"], "task": s["task"], "seconds": round(time.time() - t, 3)})
    out["score"] = {k: v for k, v in metrics.score_sample(s, out["text"]).items() if k != "parsed"}
    responses[s["sample_id"]] = out["text"]
    log.append(out)
    if i < 6 or i % 25 == 0:
        print(i, s["task"], "%.1fs" % out["seconds"], repr(out["text"][:110]), "->", out["score"]["correct"])
infer_s = time.time() - t1
with open(run_dir / "predictions.jsonl", "w") as fh:
    for r in log:
        fh.write(json.dumps(r) + "\n")
res = metrics.evaluate(rows, responses)
summary = {
    "purpose": "frozen_base_smoke", "manifest": manifest, "split": split, "root_graphs": len(keep), "samples": len(rows),
    "base_id": M.BASE_ID, "revision": M.BASE_REVISION, "device": device, "dtype": str(model.dtype),
    "torch": torch.__version__, "platform": platform.platform(), "load_seconds": round(load_s, 1),
    "inference_seconds": round(infer_s, 1), "seconds_per_sample": round(infer_s / len(rows), 2),
    "truncated_at_limit": sum(1 for r in log if r["hit_limit"]), "pipeline_checks": checks, "metrics": res,
}
if torch.cuda.is_available():
    summary["gpu"] = torch.cuda.get_device_name(0)
    summary["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    summary["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    print("GPU", summary["gpu"], "peak allocated", summary["peak_allocated_gib"], "GiB, reserved", summary["peak_reserved_gib"], "GiB")
(run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(res, indent=1))
print("s/sample %.2f  truncated %d" % (infer_s / len(rows), summary["truncated_at_limit"]))
