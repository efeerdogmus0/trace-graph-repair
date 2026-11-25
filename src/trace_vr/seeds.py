"""Pooling several training seeds of each arm, and the pre-registered seed reading (CPU only).

Fixed 2026-11-12, before any second or third seed was evaluated (protocol section 5,
paper/sections/analysis.tex "Seeds and decision order").

Pooled statistic of an arm: the mean over its seeds of the per-seed statistic, every seed scored
on the same resampled root graphs. The pooled difference A - R is mean_s(A_s) - mean_s(R_s) on the
same draw, so it needs no pairing of seeds across arms. Its interval covers root-graph sampling
only; the spread between seeds is reported separately (per-seed points, min and max), never folded
into the interval.

Seed-matched differences A_s - R_s pair the runs that share a training seed (seed s of each arm).

Reading of an arm A against the reference R on the non-edit macro (`reading`):
  positive_clear  pooled interval above 0, every seed-matched point > 0, pooled point >= 0.05
  positive        pooled interval above 0 and every seed-matched point > 0
  negative        pooled interval below 0 and every seed-matched point < 0
  inconclusive    otherwise (including: pooled interval above 0 but one seed at or below 0)
A positive reading needs all three pre-registered seeds; with fewer it is reported as
`incomplete_seeds` whatever the numbers say.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .bootstrap import MACRO, _counts, _interval, _stats
from .rng import Rng

PREREGISTERED_SEEDS = (3407, 2027, 9173)
CLEAR_MARGIN = 0.05


def _mean(values: List[Optional[float]]) -> Optional[float]:
    return None if any(v is None for v in values) else sum(values) / len(values)


def pooled_bootstrap(samples: Sequence[Dict[str, Any]], arms: Dict[str, Dict[int, Dict[str, bool]]], ref: str,
                     n_boot: int = 2000, seed: str = "bootstrap-v1", level: float = 0.95) -> Dict[str, Any]:
    """arms: {label: {training_seed: {sample_id: correct}}}. Same resampling stream as
    bootstrap.cluster_bootstrap, so one seed per arm reproduces its numbers exactly."""
    if ref not in arms:
        raise ValueError("reference %r is not an arm" % ref)
    tables, groups, task_list = {}, None, None
    for label, by_seed in arms.items():
        if not by_seed:
            raise ValueError("%s has no seeds" % label)
        for s, flags in by_seed.items():
            g, t, tab = _counts(samples, flags)
            if groups is None:
                groups, task_list = g, t
            elif g != groups or t != task_list:
                raise ValueError("%s seed %s covers different samples" % (label, s))
            tables[(label, s)] = tab
    if not groups:
        raise ValueError("no main samples to resample")
    names = list(task_list) + [MACRO]

    def pooled(draw, label):
        per = {s: _stats(draw, tables[(label, s)], task_list) for s in arms[label]}
        return {k: _mean([per[s][k] for s in per]) for k in names}, per

    point = {l: pooled(groups, l) for l in arms}
    rng = Rng(seed, len(groups), n_boot)
    reps = {l: {k: [] for k in names} for l in arms}
    diff = {l: {k: [] for k in names} for l in arms if l != ref}
    for _ in range(n_boot):
        draw = [groups[rng.below(len(groups))] for _ in groups]
        cur = {l: pooled(draw, l)[0] for l in arms}
        for l in arms:
            for k in names:
                if cur[l][k] is None:
                    continue
                reps[l][k].append(cur[l][k])
                if l != ref and cur[ref][k] is not None:
                    diff[l][k].append(cur[l][k] - cur[ref][k])

    def entry(p, r):
        if p is None or not r:
            return None
        lo, hi = _interval(r, level)
        return {"point": p, "lo": lo, "hi": hi}

    out = {"unit": "graph_group_id", "root_graphs": len(groups), "n_boot": n_boot, "seed": seed, "level": level,
           "reference": ref, "tasks": names, "arms": []}
    for l in arms:
        pp, per = point[l]
        row = {"label": l, "seeds": sorted(arms[l]),
               "per_seed": {str(s): {k: per[s][k] for k in names} for s in sorted(arms[l])},
               "pooled": {k: entry(pp[k], reps[l][k]) for k in names}, "vs_ref": None}
        macros = [per[s][MACRO] for s in arms[l] if per[s][MACRO] is not None]
        row["seed_spread"] = {"min": min(macros), "max": max(macros)} if macros else None
        if l != ref:
            vs = {}
            for k in names:
                rp = point[ref][0][k]
                e = entry(None if pp[k] is None or rp is None else pp[k] - rp, diff[l][k])
                if e is not None:
                    e["share_above_zero"] = sum(1 for d in diff[l][k] if d > 0) / len(diff[l][k])
                vs[k] = e
            matched = {str(s): per[s][MACRO] - point[ref][1][s][MACRO]
                       for s in sorted(arms[l]) if s in arms[ref]
                       and per[s][MACRO] is not None and point[ref][1][s][MACRO] is not None}
            row["vs_ref"] = vs
            row["seed_matched_macro_diff"] = matched
            row["reading"] = reading(vs.get(MACRO), matched, sorted(arms[l]), sorted(arms[ref]))
        out["arms"].append(row)
    return out


def reading(pooled_diff: Optional[Dict[str, float]], matched: Dict[str, float],
            seeds_a: Sequence[int], seeds_ref: Sequence[int]) -> str:
    if pooled_diff is None:
        return "inconclusive"
    full = set(PREREGISTERED_SEEDS)
    vals = list(matched.values())
    if pooled_diff["lo"] > 0 and vals and all(v > 0 for v in vals):
        if not (full <= set(seeds_a) and full <= set(seeds_ref) and len(matched) == len(full)):
            return "incomplete_seeds"
        return "positive_clear" if pooled_diff["point"] >= CLEAR_MARGIN else "positive"
    if pooled_diff["hi"] < 0 and vals and all(v < 0 for v in vals):
        return "negative"
    return "inconclusive"


def choose_reference(table: Dict[str, Any], candidates: Sequence[str] = ("B", "C")) -> Dict[str, Any]:
    """Pick the reference baseline on dev (protocol section 5), from a pooled seed table.

    Rule (fixed 2026-11-13, before the locked test is opened): every candidate must be in the table
    with all three pre-registered seeds; the chosen reference is the candidate with the higher pooled
    non-edit macro point; on an exact tie the first candidate (B, the plain QA baseline) is kept.
    The table must be a dev table. The gap between the candidates is reported, not used to decide."""
    if not str(table.get("split", "")).startswith("dev"):
        raise ValueError("the reference baseline is chosen on dev only, got split %r" % table.get("split"))
    rows = {r["label"]: r for r in table["arms"]}
    full = set(PREREGISTERED_SEEDS)
    for c in candidates:
        if c not in rows:
            raise ValueError("candidate %s is not in the seed table" % c)
        if not full <= set(rows[c]["seeds"]):
            raise ValueError("candidate %s has seeds %s, needs all of %s"
                             % (c, sorted(rows[c]["seeds"]), sorted(full)))
    points = {c: rows[c]["pooled"][MACRO]["point"] for c in candidates}
    best = max(points.values())
    chosen = next(c for c in candidates if points[c] == best)
    out = {"rule": "higher pooled non-edit macro on dev, all three seeds; tie -> first candidate",
           "split": table["split"], "content_digest": table.get("content_digest"),
           "candidates": list(candidates), "pooled_macro": points,
           "per_seed_macro": {c: {s: v[MACRO] for s, v in rows[c]["per_seed"].items()} for c in candidates},
           "chosen": chosen}
    others = [c for c in candidates if c != chosen]
    if table.get("reference") == chosen:
        out["gaps_vs_chosen"] = {c: rows[c].get("vs_ref", {}).get(MACRO) for c in others}
    return out
