"""Arm E on the same eval samples as the other arms. Usage:
python scripts/eval_parse_solve.py <manifest> <split> <n_roots> <run_dir> [adapter_dir|none] [max_new] [sample_set]

One extraction generation per image (root and edit images of the first n_roots graphs),
then every main question on that image is answered by the exact solver on the parsed graph
(src/trace_vr/parse_solve.py) and scored against the true graph with metrics.evaluate.
Writes extraction_<split>.jsonl, predictions_<split>.jsonl (derived responses, same format as
eval_arm.py so export_flags.py works) and metrics_<split>.json; prints EVAL-SUMMARY.
sample_set: main (default) | invariance (relabelled/restyled copies; files get suffix _inv, as eval_arm.py)."""
import json, sys, time
from pathlib import Path
from trace_vr import arms, dataset, metrics, model as M, parse_solve as ps, tasks

manifest, split, n_roots, run_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
adapter = sys.argv[5] if len(sys.argv) > 5 else "none"
max_new = int(sys.argv[6]) if len(sys.argv) > 6 else 512
sample_set = sys.argv[7] if len(sys.argv) > 7 else "main"
suffix = arms.run_suffix(split, "image", sample_set)
dataset.check_split_allowed(split, "eval")
run_dir.mkdir(parents=True, exist_ok=True)
man, data_dir, by_split, _ = dataset.load_dataset(Path(manifest))
rows = arms.select_samples(by_split[split], n_roots, sample_set)
first = {}
for s in rows:
    first.setdefault(s["image_id"], s)
model, processor, device = M.load()
if adapter != "none":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
model.eval()
prompt = ps.extraction_prompt()
ext, t0, trunc, new_tokens, ext_scores = {}, time.time(), 0, 0, []
with open(run_dir / ("extraction_%s.jsonl" % suffix), "w") as fh:
    for i, (img, s) in enumerate(first.items()):
        ex = dataset.to_model_example(s, data_dir)
        out = M.generate(model, processor, M.build_inference_inputs(processor, ex["image"], prompt), max_new)
        ext[img] = out["text"]; trunc += out["hit_limit"]; new_tokens += out["new_tokens"]
        t = s["private_truth"]
        sc = tasks.score_response(ps.extraction_query(), t["nodes"], t["edges"], out["text"])
        scl = tasks.score_response(ps.extraction_query(), t["nodes"], t["edges"], tasks.lenient_candidate(out["text"]))
        ext_scores.append((sc, scl))
        fh.write(json.dumps({"image_id": img, "text": out["text"], "new_tokens": out["new_tokens"]}) + "\n")
        if i % 50 == 0:
            print("extract %d/%d %.0fs" % (i, len(first), time.time() - t0), flush=True)
resp, reasons = ps.responses_for_samples(rows, ext)
resp_l, reasons_l = ps.responses_for_samples(rows, ext, lenient=True)
with open(run_dir / ("predictions_%s.jsonl" % suffix), "w") as fh:
    for s in rows:
        fh.write(json.dumps({"sample_id": s["sample_id"], "task": s["task"], "text": resp[s["sample_id"]], "new_tokens": 0}) + "\n")
res, res_l = metrics.evaluate(rows, resp), metrics.evaluate(rows, resp_l)
if sample_set == "invariance":  # evaluate() scores root/edit images only; score the copies directly
    inv: dict = {}
    for s in rows:
        inv.setdefault(s["task"], []).append(metrics.score_sample(s, resp.get(s["sample_id"]))["correct"])
    res["invariance_copy_accuracy"] = {t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in inv.items()}


def _ext_summary(k):
    vals = [x[k] for x in ext_scores]
    def m(key):
        v = [r[key] for r in vals if r.get(key) is not None]
        return round(sum(v) / len(v), 4) if v else None
    return {"exact": round(sum(r["correct"] for r in vals) / len(vals), 4), "format_ok": round(sum(r["format_ok"] for r in vals) / len(vals), 4),
            **{key: m(key) for key in ("edge_precision", "edge_recall", "direction_accuracy", "node_recall")}}


summary = {"arm": "E", "manifest": manifest, "split": split, "eval_root_graphs": n_roots, "samples": len(rows), "images": len(first), "adapter": adapter,
           "max_new_tokens": max_new, "sample_set": sample_set, "content_digest": man["content_digest"], "images_digest": man["images_digest"], "eval_seconds": round(time.time() - t0, 1),
           "truncated_at_limit": int(trunc), "mean_new_tokens": new_tokens / max(1, len(first)), "extraction_strict": _ext_summary(0),
           "extraction_lenient": _ext_summary(1), "reasons": reasons, "reasons_lenient": reasons_l, "metrics": res, "metrics_lenient_parse": res_l}
(run_dir / ("metrics_%s.json" % suffix)).write_text(json.dumps(summary, indent=2))
acc = lambda r: {t: round(e["accuracy"], 4) for t, e in r["per_task"].items()}
print("EVAL-SUMMARY " + json.dumps({"arm": "E", "adapter": adapter, "n": len(rows), "images": len(first), "truncated": int(trunc), "seconds": summary["eval_seconds"],
                                    "extraction": summary["extraction_strict"], "extraction_lenient": summary["extraction_lenient"], "per_task": acc(res), "macro": res["non_edit_macro_accuracy"],
                                    "per_task_lenient": acc(res_l), "macro_lenient": res_l["non_edit_macro_accuracy"], "reasons": reasons, "reasons_lenient": reasons_l, "sample_set": sample_set,
                                    "invariance_copy_accuracy": res.get("invariance_copy_accuracy")}), flush=True)
