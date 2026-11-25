"""M3 step 2: tiny LoRA overfit. Shows that the training path is wired; it is NOT a research result.

Usage: python scripts/overfit_smoke.py <manifest> <split> <n_samples> <steps> <run_dir>
"""
import json, sys, time, platform
from pathlib import Path
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from trace_vr import dataset, metrics, model as M, tasks

manifest, split, n, steps, run_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
dataset.check_split_allowed(split, "train")
run_dir.mkdir(parents=True, exist_ok=True)
_, data_dir, samples_by_split, _ = dataset.load_dataset(Path(manifest))
rows = [s for s in samples_by_split[split] if s["task"] == tasks.REPAIR][:n]
print("categories", sorted(s["metadata"]["task_info"]["category"] for s in rows))

torch.manual_seed(3407)
model, processor, device = M.load()
TARGETS = r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"
probe_name, probe = next((k, v) for k, v in model.named_parameters() if k.endswith("gate_proj.weight"))
base_before = probe.detach().float().sum().item()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules=TARGETS))
trainable = [p for p in model.parameters() if p.requires_grad]
n_train, n_all = sum(p.numel() for p in trainable), sum(p.numel() for p in model.parameters())
print("trainable %d of %d (%.3f%%), adapter dtype %s" % (n_train, n_all, 100 * n_train / n_all, trainable[0].dtype))
lora_b = next(p for k, p in model.named_parameters() if "lora_B" in k)
opt = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01, eps=1e-6)

batches = []
for s in rows:
    ex = dataset.to_model_example(s, data_dir, with_target=True)
    full, _ = M.build_training_inputs(processor, ex["image"], ex["prompt"], ex["target"])
    batches.append({k: v.to(device) for k, v in full.items()})

model.train()
log, t0 = [], time.time()
for step in range(steps):
    out = model(**batches[step % len(batches)])
    loss = out.loss
    if not torch.isfinite(loss):
        print("non-finite loss at step", step); break
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    log.append({"step": step, "loss": float(loss.detach()), "seconds": round(time.time() - t0, 1)})
    if step % 10 == 0 or step == steps - 1:
        print("step %3d loss %.4f  %.0fs" % (step, float(loss), time.time() - t0))
train_s = time.time() - t0
(run_dir / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))

def run_eval(mdl):
    mdl.eval(); resp = {}
    for s in rows:
        ex = dataset.to_model_example(s, data_dir)
        resp[s["sample_id"]] = M.generate(mdl, processor, M.build_inference_inputs(processor, ex["image"], ex["prompt"]), 192)["text"]
    return resp, metrics.evaluate(rows, resp)["per_task"][tasks.REPAIR]

resp, after = run_eval(model)
print("after training:", json.dumps(after))
with model.disable_adapter():
    _, off = run_eval(model)
print("adapter disabled:", json.dumps(off))
model.save_pretrained(run_dir / "adapter")
base_after = probe.detach().float().sum().item()  # same Parameter object, now wrapped as base_layer.weight
del model
fresh, _, _ = M.load()
reloaded = PeftModel.from_pretrained(fresh, run_dir / "adapter")
resp2, again = run_eval(reloaded)
summary = {
    "purpose": "overfit_smoke_not_a_result", "samples": len(rows), "steps": len(log), "device": device, "torch": torch.__version__,
    "platform": platform.platform(), "base_id": M.BASE_ID, "revision": M.BASE_REVISION, "lora_targets": TARGETS,
    "trainable_params": n_train, "total_params": n_all, "first_loss": log[0]["loss"], "last_loss": log[-1]["loss"],
    "train_seconds": round(train_s, 1), "seconds_per_step": round(train_s / max(1, len(log)), 2),
    "lora_B_abs_sum_after": float(lora_b.detach().abs().sum()), "base_weight_unchanged": base_before == base_after,
    "eval_after": after, "eval_adapter_disabled": off, "eval_after_reload": again, "same_outputs_after_reload": resp == resp2,
}
if torch.cuda.is_available():
    summary["gpu"] = torch.cuda.get_device_name(0)
    summary["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    summary["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    print("GPU", summary["gpu"], "peak allocated", summary["peak_allocated_gib"], "GiB, reserved", summary["peak_reserved_gib"], "GiB")
(run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("first_loss", "last_loss", "seconds_per_step", "lora_B_abs_sum_after", "base_weight_unchanged", "same_outputs_after_reload")}, indent=1))
print("after reload:", json.dumps(again))
