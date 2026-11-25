"""Evaluate one arm on the pre-registered VGCure subset (M7; run once, after the locked test).

usage: python scripts/eval_vgcure.py <work_dir> <run_dir> <adapter_dir|none> [max_new] [--parse-solve]

<work_dir> is the output of scripts/vgcure_prepare.py (subset_full.json + images). Default: the adapter
answers each instruction directly (max_new 96, as on dev). --parse-solve (arm E): the adapter transcribes
each image once (extraction prompt, max_new 512) and the exact solver answers every item on that image.
Writes predictions_vgcure.jsonl and metrics_vgcure.json in <run_dir>; prints EVAL-SUMMARY and
FLAGS-SUMMARY (scores fixed in docs/external_eval.md).
"""
import hashlib, json, os, sys, time
from pathlib import Path

from trace_vr import vgcure


def main(argv):
    parse_solve = "--parse-solve" in argv
    argv = [a for a in argv if a != "--parse-solve"]
    work, run_dir, adapter = Path(argv[0]), Path(argv[1]), argv[2]
    max_new = int(argv[3]) if len(argv) > 3 else (512 if parse_solve else 96)
    sub = json.loads((work / "subset_full.json").read_text())
    rows = sub["samples"]
    record = json.loads((Path(__file__).resolve().parents[1] / os.environ.get("TRACE_VGCURE_RECORD", vgcure.TEST_RECORD)).read_text())
    assert sub["subset_sha256"] == record["subset_sha256"] and [s["sample_id"] for s in rows] == record["sample_ids"]
    from trace_vr import model as M, parse_solve as ps
    model, processor, _ = M.load()
    if adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    run_dir.mkdir(parents=True, exist_ok=True)
    t0, trunc, new_tokens, responses, ext = time.time(), 0, 0, {}, {}
    if parse_solve:
        prompt = ps.extraction_prompt()
        images = list(dict.fromkeys(s["image"] for s in rows))
        with open(run_dir / "extraction_vgcure.jsonl", "w") as fh:
            for i, im in enumerate(images):
                out = M.generate(model, processor, M.build_inference_inputs(processor, str(work / im), prompt), max_new)
                ext[im] = out["text"]; trunc += out["hit_limit"]; new_tokens += out["new_tokens"]
                fh.write(json.dumps({"image": im, "text": out["text"], "new_tokens": out["new_tokens"], "hit_limit": out["hit_limit"]}) + "\n")
                if i % 50 == 0:
                    print("extract %d/%d %.0fs" % (i, len(images), time.time() - t0), flush=True)
        reasons = {}
        for s in rows:
            resp, info = ps.answer_from_extraction(s["query"], ext[s["image"]])
            responses[s["sample_id"]] = resp
            k = info["reason"] or "answered"; reasons[k] = reasons.get(k, 0) + 1
        n_gen = len(images)
    else:
        reasons = None
        for i, s in enumerate(rows):
            out = M.generate(model, processor, M.build_inference_inputs(processor, str(work / s["image"]), s["instruction"]), max_new)
            responses[s["sample_id"]] = out["text"]; trunc += out["hit_limit"]; new_tokens += out["new_tokens"]
            if i % 100 == 0:
                print("eval %d/%d %.0fs" % (i, len(rows), time.time() - t0), flush=True)
        n_gen = len(rows)
    with open(run_dir / "predictions_vgcure.jsonl", "w") as fh:
        for s in rows:
            fh.write(json.dumps({"sample_id": s["sample_id"], "task": s["task"], "text": responses[s["sample_id"]]}) + "\n")
    res = vgcure.evaluate(rows, responses)
    summary = {"arm_mode": "parse_solve" if parse_solve else "direct", "adapter": adapter, "max_new_tokens": max_new,
               "subset_sha256": sub["subset_sha256"], "revision": sub["revision"], "generations": n_gen,
               "truncated_at_limit": int(trunc), "mean_new_tokens": new_tokens / max(1, n_gen),
               "eval_seconds": round(time.time() - t0, 1), "reasons": reasons, "metrics": res}
    (run_dir / "metrics_vgcure.json").write_text(json.dumps(summary, indent=2))
    r4 = lambda v: round(v, 4) if isinstance(v, float) else v
    print("EVAL-SUMMARY " + json.dumps({"split": "vgcure", "adapter": adapter, "mode": summary["arm_mode"], "n": len(rows),
                                        "truncated": int(trunc), "seconds": summary["eval_seconds"], "reasons": reasons,
                                        **{k: r4(v) for k, v in res.items() if k != "per_source"}}), flush=True)
    raw = (run_dir / "predictions_vgcure.jsonl").read_bytes()
    fl = vgcure.flags(rows, responses)
    fl.update({"run": run_dir.name, "split": "vgcure", "subset_sha256": sub["subset_sha256"],
               "predictions_sha256": hashlib.sha256(raw).hexdigest()})
    print("FLAGS-SUMMARY " + json.dumps(fl), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
