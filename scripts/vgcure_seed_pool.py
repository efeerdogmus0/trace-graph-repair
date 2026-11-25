"""Seed-pooled VGCure comparison: per-seed scores and pooled differences with a paired graph bootstrap.

usage: python scripts/vgcure_seed_pool.py <meta_root> [--dev] [--out <json>] <arm>=<job_dir>/<run>,<job_dir>/<run>,... ...

The first arm is the reference. Seeds are listed in the same order for every arm. A pooled score is the mean of
the per-seed scores; each of the 10,000 resamples (seed 20261114) draws graphs once and applies the draw to every
seed and arm, so the interval covers the choice of graphs, not the choice of training seed (three seeds).
"""
import json, random, sys
from pathlib import Path

from trace_vr import vgcure

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("external_macro", "edge_exists_accuracy", "reachability_balanced_accuracy")


def main(argv):
    dev = "--dev" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--out")]
    meta, specs = Path(pos[0]), [p.split("=", 1) for p in pos[1:]]
    record = json.load(open(ROOT / (vgcure.DEV_RECORD if dev else vgcure.TEST_RECORD)))
    sub = vgcure.build_subset(meta, record["per_source"], record.get("skip", 0))
    if sub["subset_sha256"] != record["subset_sha256"]:
        raise SystemExit("subset digest differs from the selection record")
    samples = sub["samples"]
    graphs = {}
    for s in samples:
        graphs.setdefault(s["sample_id"].rsplit("/", 1)[0], []).append(s)
    names = sorted(graphs)

    def counts(path):
        job, run = path.rsplit("/", 1)
        resp = {r["sample_id"]: r.get("text") for r in map(json.loads, open(Path(job) / "runs" / run / "predictions_vgcure.jsonl"))}
        if any(s["sample_id"] not in resp for s in samples):
            raise SystemExit("%s: responses missing" % path)
        tab = []
        for g in names:
            c = [0] * 6
            for s in graphs[g]:
                ok = vgcure.score_item(s, resp[s["sample_id"]])["correct"]
                i = 0 if s["task"] != "reachability" else (2 if s["answer"] else 4)
                c[i] += 1; c[i + 1] += ok
            tab.append(c)
        point = vgcure.evaluate(samples, resp)
        st = stats(tab, range(len(names)))
        assert all(abs(st[k] - point[k]) < 1e-12 for k in KEYS), path
        return tab

    def stats(tab, idx):
        t = [sum(tab[i][j] for i in idx) for j in range(6)]
        edge = t[1] / t[0]
        parts = [t[k + 1] / t[k] for k in (2, 4) if t[k]]
        reach = sum(parts) / len(parts)
        return dict(zip(KEYS, ((edge + reach) / 2, edge, reach)))

    def pooled(tabs, idx):
        per = [stats(t, idx) for t in tabs]
        return {k: sum(p[k] for p in per) / len(per) for k in KEYS}

    arms = {name: [counts(p) for p in paths.split(",")] for name, paths in specs}
    order = [name for name, _ in specs]
    ref = order[0]
    full = list(range(len(names)))
    res = {"subset_sha256": sub["subset_sha256"], "record": "dev" if dev else "preregistered", "graphs": len(names),
           "n": len(samples), "reference": ref, "resamples": 10000, "seed": 20261114, "arms": {}}
    for a in order:
        res["arms"][a] = {"runs": dict(specs)[a].split(","), "per_seed": [stats(t, full) for t in arms[a]],
                          "pooled": pooled(arms[a], full)}
    rng = random.Random(20261114)
    diffs = {a: {k: [] for k in KEYS} for a in order[1:]}
    seed_diffs = {a: [{k: [] for k in KEYS} for _ in arms[a]] for a in order[1:]}
    for _ in range(10000):
        idx = [rng.randrange(len(names)) for _ in names]
        base = pooled(arms[ref], idx)
        base_seed = [stats(t, idx) for t in arms[ref]]
        for a in order[1:]:
            m = pooled(arms[a], idx)
            for k in KEYS:
                diffs[a][k].append(m[k] - base[k])
            if len(arms[a]) == len(arms[ref]):
                for i, t in enumerate(arms[a]):
                    st = stats(t, idx)
                    for k in KEYS:
                        seed_diffs[a][i][k].append(st[k] - base_seed[i][k])

    def ci(v, point):
        v = sorted(v)
        return {"point": point, "lo": v[int(0.025 * len(v))], "hi": v[int(0.975 * len(v)) - 1],
                "p_le_0": sum(x <= 0 for x in v) / len(v)}

    for a in order[1:]:
        row = res["arms"][a]
        row["diff_pooled"] = {k: ci(diffs[a][k], row["pooled"][k] - res["arms"][ref]["pooled"][k]) for k in KEYS}
        if len(arms[a]) == len(arms[ref]):
            row["diff_per_seed"] = [{k: ci(sd[k], row["per_seed"][i][k] - res["arms"][ref]["per_seed"][i][k]) for k in KEYS}
                                    for i, sd in enumerate(seed_diffs[a])]
    for a in order:
        r = res["arms"][a]
        print(a, "pooled", {k: round(v, 4) for k, v in r["pooled"].items()}, "seeds macro", [round(p["external_macro"], 4) for p in r["per_seed"]])
        if "diff_pooled" in r:
            for k in KEYS:
                d = r["diff_pooled"][k]
                print("   vs %s %s %+.4f [%+.4f, %+.4f] p<=0 %.4f" % (ref, k, d["point"], d["lo"], d["hi"], d["p_le_0"]))
            for i, sd in enumerate(r.get("diff_per_seed", [])):
                d = sd["external_macro"]
                print("   seed %d macro %+.4f [%+.4f, %+.4f]" % (i, d["point"], d["lo"], d["hi"]))
    if out:
        Path(out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
