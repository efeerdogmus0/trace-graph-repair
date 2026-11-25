"""Paired graph-level bootstrap on a VGCure subset (light CPU; needs the local VGCure metadata, not the images).

usage: python scripts/vgcure_compare.py <meta_root> <job_dir> <ref_run> <run> [<run> ...] [--dev] [--out <json>]

Rebuilds the subset from <meta_root> (understanding/ + reasoning/ test files), refuses unless its digest
equals the selection record, re-scores each run's predictions_vgcure.jsonl with vgcure.evaluate and reports
external macro, edge accuracy and reachability balanced accuracy per run plus the difference to <ref_run>
with a 95% percentile interval (10,000 resamples of graphs, seed 20261114). The default record is the
pre-registered subset; --dev selects the exploratory one.
"""
import json, random, sys
from pathlib import Path

from trace_vr import vgcure

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("external_macro", "edge_exists_accuracy", "reachability_balanced_accuracy")


def load(job: Path, run: str):
    return {r["sample_id"]: r.get("text") for r in map(json.loads, open(job / "runs" / run / "predictions_vgcure.jsonl"))}


def main(argv):
    dev = "--dev" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--out")]
    meta, job, ref, runs = Path(pos[0]), Path(pos[1]), pos[2], pos[3:]
    record = json.load(open(ROOT / (vgcure.DEV_RECORD if dev else vgcure.TEST_RECORD)))
    sub = vgcure.build_subset(meta, record["per_source"], record.get("skip", 0))
    if sub["subset_sha256"] != record["subset_sha256"]:
        raise SystemExit("subset digest differs from the selection record")
    samples = sub["samples"]
    graphs = {}
    for s in samples:
        graphs.setdefault(s["sample_id"].rsplit("/", 1)[0], []).append(s)
    names = sorted(graphs)
    resp = {r: load(job, r) for r in [ref] + runs}
    for r, d in resp.items():
        missing = [s["sample_id"] for s in samples if s["sample_id"] not in d]
        if missing:
            raise SystemExit("%s: %d responses missing" % (r, len(missing)))
    point = {r: vgcure.evaluate(samples, d) for r, d in resp.items()}
    # per-graph sufficient counts, so each resample is a few sums (edge n/correct, reach yes n/correct, reach no n/correct)
    def counts(r):
        tab = []
        for g in names:
            c = [0] * 6
            for s in graphs[g]:
                ok = vgcure.score_item(s, resp[r].get(s["sample_id"]))["correct"]
                i = 0 if s["task"] != "reachability" else (2 if s["answer"] else 4)
                c[i] += 1; c[i + 1] += ok
            tab.append(c)
        return tab

    def stats(tab, idx):
        t = [sum(tab[i][j] for i in idx) for j in range(6)]
        edge = t[1] / t[0] if t[0] else None
        parts = [t[k + 1] / t[k] for k in (2, 4) if t[k]]
        reach = sum(parts) / len(parts) if parts else None
        macro = (edge + reach) / 2 if edge is not None and reach is not None else None
        return dict(zip(KEYS, (macro, edge, reach)))

    tabs = {r: counts(r) for r in [ref] + runs}
    full = list(range(len(names)))
    for r in tabs:   # the fast path must reproduce vgcure.evaluate exactly
        st = stats(tabs[r], full)
        assert all(abs(st[k] - point[r][k]) < 1e-12 for k in KEYS), r
    rng = random.Random(20261114)
    diffs = {r: {k: [] for k in KEYS} for r in runs}
    for _ in range(10000):
        idx = [rng.randrange(len(names)) for _ in names]
        base = stats(tabs[ref], idx)
        for r in runs:
            m = stats(tabs[r], idx)
            for k in KEYS:
                if m[k] is not None and base[k] is not None:
                    diffs[r][k].append(m[k] - base[k])
    res = {"subset_sha256": sub["subset_sha256"], "record": "dev" if dev else "preregistered", "graphs": len(names),
           "n": len(samples), "reference": ref, "resamples": 10000, "seed": 20261114, "runs": {}}
    for r in [ref] + runs:
        row = {k: point[r][k] for k in KEYS}
        row["format_ok"] = point[r]["format_ok"]
        if r != ref:
            for k in KEYS:
                v = sorted(diffs[r][k])
                row["diff_" + k] = {"point": point[r][k] - point[ref][k], "lo": v[int(0.025 * len(v))],
                                    "hi": v[int(0.975 * len(v)) - 1], "p_le_0": sum(x <= 0 for x in v) / len(v)}
        res["runs"][r] = row
        print(r, json.dumps(row))
    if out:
        Path(out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
