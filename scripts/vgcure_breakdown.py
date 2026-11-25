"""Descriptive breakdown of the pre-registered VGCure runs by item class (CPU only, reads local predictions).

usage: python scripts/vgcure_breakdown.py <meta_root> --out <json> <arm>=<job_dir>/<run>,... ...

For every arm and seed: accuracy on true and on false edge claims, on yes and on no path items, the share of
"true" answers per task, the format-failure share, and per-source edge / reachability accuracy. Pooled values are
means over the seeds. No intervals and no decision: this explains the pooled scores of scripts/vgcure_seed_pool.py
(same subset, same scorer `vgcure.score_item`), it does not test anything. Only counts and rates are written; no
VGCure text leaves the machine.
"""
import json, sys
from pathlib import Path

from trace_vr import tasks, vgcure

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("edge_true", "edge_false", "reach_yes", "reach_no")


def run_stats(samples, resp):
    acc = {c: [] for c in CLASSES}
    said_true = {"edge": [], "reach": []}
    fmt, src = [], {}
    for s in samples:
        text = resp[s["sample_id"]]
        r = vgcure.score_item(s, text)
        fmt.append(r["format_ok"])
        edge = s["task"] == tasks.EDGE_EXISTS
        cls = ("edge_" if edge else "reach_") + (("true" if s["answer"] else "false") if edge else ("yes" if s["answer"] else "no"))
        acc[cls].append(r["correct"])
        if r["format_ok"]:
            said_true["edge" if edge else "reach"].append(r["correct"] == bool(s["answer"]))
        p = src.setdefault(s["source"], {"edge": [], "reach": []})
        p["edge" if edge else "reach"].append(r["correct"])
    mean = lambda v: sum(v) / len(v) if v else None
    out = {"accuracy": {c: mean(v) for c, v in acc.items()}, "n": {c: len(v) for c, v in acc.items()},
           "said_true": {k: mean(v) for k, v in said_true.items()}, "format_failure": 1 - mean(fmt),
           "per_source": {k: {"edge": mean(v["edge"]), "reach": mean(v["reach"])} for k, v in sorted(src.items())}}
    return out


def pool(per):
    def avg(get):
        vals = [get(p) for p in per]
        return None if any(v is None for v in vals) else sum(vals) / len(vals)
    return {"accuracy": {c: avg(lambda p, c=c: p["accuracy"][c]) for c in CLASSES},
            "said_true": {k: avg(lambda p, k=k: p["said_true"][k]) for k in ("edge", "reach")},
            "format_failure": avg(lambda p: p["format_failure"]),
            "per_source": {s: {t: avg(lambda p, s=s, t=t: p["per_source"][s][t]) for t in ("edge", "reach")}
                           for s in per[0]["per_source"]}}


def main(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--out")]
    meta, specs = Path(pos[0]), [p.split("=", 1) for p in pos[1:]]
    record = json.load(open(ROOT / vgcure.TEST_RECORD))
    sub = vgcure.build_subset(meta, record["per_source"], record.get("skip", 0))
    if sub["subset_sha256"] != record["subset_sha256"]:
        raise SystemExit("subset digest differs from the selection record")
    samples = sub["samples"]
    res = {"subset_sha256": sub["subset_sha256"], "record": "preregistered", "n": len(samples), "arms": {}}
    for arm, paths in specs:
        per = []
        for path in paths.split(","):
            job, run = path.rsplit("/", 1)
            resp = {r["sample_id"]: r.get("text") for r in map(json.loads, open(Path(job) / "runs" / run / "predictions_vgcure.jsonl"))}
            if any(s["sample_id"] not in resp for s in samples):
                raise SystemExit("%s: responses missing" % path)
            per.append(run_stats(samples, resp))
        res["arms"][arm] = {"runs": paths.split(","), "per_seed": per, "pooled": pool(per)}
        p = res["arms"][arm]["pooled"]
        print(arm, " ".join("%s %.3f" % (c, p["accuracy"][c]) for c in CLASSES),
              "| said true edge %.3f reach %.3f | format fail %.3f" % (p["said_true"]["edge"], p["said_true"]["reach"], p["format_failure"]))
    if out:
        Path(out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
