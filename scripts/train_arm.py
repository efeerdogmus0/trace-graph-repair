"""LoRA training of one arm.
Usage: python scripts/train_arm.py <manifest> <split> <arm> <run_dir> [epochs] [accum] [seed] [ckpt_every]
ckpt_every (samples, rounded up to a multiple of accum; 0 = off) writes <run_dir>/ckpt; a rerun
with the same arguments resumes from it (trace_vr.resume)."""
import json, sys, time, platform
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from trace_vr import arms, dataset, model as M, resume
from trace_vr.rng import Rng

manifest, split, arm, run_dir = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 1
accum = int(sys.argv[6]) if len(sys.argv) > 6 else 4
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 3407
ckpt_every = int(sys.argv[8]) if len(sys.argv) > 8 else 0
dataset.check_split_allowed(split, "train")
if ckpt_every:
    ckpt_every = -(-ckpt_every // accum) * accum
LR, WARMUP, R = 2e-4, 0.03, 16
TARGETS = r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"
run_dir.mkdir(parents=True, exist_ok=True)
man, data_dir, by_split, _ = dataset.load_dataset(Path(manifest))
rows = arms.training_samples(by_split[split], arm)
torch.manual_seed(seed)
model, processor, device = M.load()
model = get_peft_model(model, LoraConfig(r=R, lora_alpha=R, lora_dropout=0.0, target_modules=TARGETS))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01, eps=1e-6)
total_updates = (len(rows) * epochs + accum - 1) // accum
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / max(1, int(WARMUP * total_updates))) * max(0.0, 1 - u / total_updates))
order_rng = Rng("train-order", seed, arm)
model.train()
log, t0, seen, tokens_in, tokens_target, prior_s = [], time.time(), 0, 0, 0, 0.0
ckpt_key = {"arm": arm, "manifest": manifest, "split": split, "epochs": epochs, "accum": accum, "seed": seed, "samples": len(rows)}
restored = resume.load(run_dir, model, opt, sched, ckpt_key)
if restored:
    seen, log, prior_s = restored["seen"], restored["log"], restored["elapsed_seconds"]
    tokens_in, tokens_target = restored["tokens_in"], restored["tokens_target"]
    print("resumed from %s at sample %d" % (run_dir / resume.CKPT, seen), flush=True)
skip = seen
print("arm %s: %d samples, %d epochs, accum %d, %d updates, trainable %d" % (arm, len(rows), epochs, accum, total_updates, sum(p.numel() for p in params)), flush=True)
for epoch in range(epochs):
    for s in order_rng.shuffled(rows):  # always drawn, so the order replays exactly on resume
        if skip:
            skip -= 1
            continue
        ex = dataset.to_model_example(s, data_dir, with_target=True)
        full, n_prompt = M.build_training_inputs(processor, ex["image"], ex["prompt"], ex["target"])
        tokens_in += int(full["input_ids"].shape[1]); tokens_target += int(full["input_ids"].shape[1]) - n_prompt
        loss = model(**{k: v.to(device) for k, v in full.items()}).loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss at sample %d" % seen)
        (loss / accum).backward()
        seen += 1
        if seen % accum == 0 or seen == len(rows) * epochs:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        log.append({"sample": seen, "loss": float(loss.detach()), "seconds": round(prior_s + time.time() - t0, 1)})
        if ckpt_every and seen % ckpt_every == 0 and seen < len(rows) * epochs:
            resume.save(run_dir, model, opt, sched, seen, log, {**ckpt_key, "elapsed_seconds": prior_s + time.time() - t0,
                                                               "tokens_in": tokens_in, "tokens_target": tokens_target})
            print("checkpoint at sample %d" % seen, flush=True)
        if seen % 100 == 0:
            recent = sum(r["loss"] for r in log[-100:]) / 100
            print("seen %5d  mean loss(last 100) %.4f  %.0fs" % (seen, recent, time.time() - t0), flush=True)
train_s = prior_s + time.time() - t0
model.save_pretrained(run_dir / "adapter")
(run_dir / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
summary = {"arm": arm, "manifest": manifest, "split": split, "content_digest": man["content_digest"], "images_digest": man["images_digest"],
           "samples": len(rows), "root_graphs": len({s["graph_group_id"] for s in rows}), "images": len({s["image_id"] for s in rows}),
           "epochs": epochs, "accum": accum, "resumed_from_sample": restored["seen"] if restored else 0, "updates": total_updates, "lr": LR, "lora_r": R, "lora_targets": TARGETS, "seed": seed,
           "base_id": M.BASE_ID, "revision": M.BASE_REVISION, "torch": torch.__version__, "platform": platform.platform(),
           "tokens_in_seen": tokens_in, "target_tokens_seen": tokens_target, "train_seconds": round(train_s, 1),
           "mean_loss_first_100": sum(r["loss"] for r in log[:100]) / 100, "mean_loss_last_100": sum(r["loss"] for r in log[-100:]) / 100}
if torch.cuda.is_available():
    summary.update({"gpu": torch.cuda.get_device_name(0), "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)})
(run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
print("TRAIN-SUMMARY " + json.dumps(summary), flush=True)
