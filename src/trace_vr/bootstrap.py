"""Confidence intervals by resampling root graphs, and a compact per-sample record.

Samples that share a root graph are not independent (same topology, twin images, several
questions), so the resampling unit is `graph_group_id`, never the single sample. When two
arms are compared, both are scored on the same resampled root graphs in every replicate
(paired bootstrap), and the interval is for the difference.

Per-sample correctness travels as a bit string plus a hash of the sample-id order, so a
run on another machine can be re-analysed here without trusting a hand-copied table.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import metrics, tasks
from .rng import Rng

MACRO = "non_edit_macro_accuracy"


def order_sha(samples: Sequence[Dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(s["sample_id"] for s in samples).encode("utf-8")).hexdigest()[:16]


def score_flags(samples: Sequence[Dict[str, Any]], responses: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Strict per-sample scores in sample order, as bit strings."""
    scored = [metrics.score_sample(s, responses.get(s["sample_id"])) for s in samples]
    return {
        "n": len(samples),
        "missing_responses": sum(1 for s in samples if s["sample_id"] not in responses),
        "order_sha": order_sha(samples),
        "correct": "".join("1" if r["correct"] else "0" for r in scored),
        "format_ok": "".join("1" if r["format_ok"] else "0" for r in scored),
    }


def decode_flags(samples: Sequence[Dict[str, Any]], payload: Dict[str, Any], key: str = "correct") -> Dict[str, bool]:
    """Bit string -> {sample_id: bool}. Refuses a payload made for a different sample list."""
    bits = payload[key]
    if payload.get("order_sha") != order_sha(samples):
        raise ValueError("flags were produced for a different sample order")
    if len(bits) != len(samples) or set(bits) - {"0", "1"}:
        raise ValueError("flag string does not match the sample list")
    return {s["sample_id"]: b == "1" for s, b in zip(samples, bits)}


def subset_flags(samples: Sequence[Dict[str, Any]], payload: Dict[str, Any],
                 subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Re-express a flags payload made for `samples` on `subset` (every subset id must be in
    `samples`), e.g. the first 100 of 200 evaluated root graphs. The original hash is checked."""
    out = dict(payload, n=len(subset), order_sha=order_sha(subset), subset_of=payload.get("order_sha"))
    for key in ("correct", "format_ok"):
        if key in payload:
            full = decode_flags(samples, payload, key)
            out[key] = "".join("1" if full[s["sample_id"]] else "0" for s in subset)
    out.pop("missing_responses", None)
    return out


def _main_rows(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in samples if s["metadata"]["image_variant"] in metrics.MAIN_VARIANTS]


def _counts(samples: Sequence[Dict[str, Any]], flags: Dict[str, bool]):
    """groups in first-seen order, tasks present, and per group/task [right, total]."""
    groups: List[str] = []
    table: Dict[str, Dict[str, List[int]]] = {}
    present: List[str] = []
    for s in _main_rows(samples):
        g, t = s["graph_group_id"], s["task"]
        if g not in table:
            table[g] = {}
            groups.append(g)
        if t not in present:
            present.append(t)
        cell = table[g].setdefault(t, [0, 0])
        cell[0] += 1 if flags[s["sample_id"]] else 0
        cell[1] += 1
    return groups, [t for t in tasks.ALL_TASKS if t in present], table


def _stats(draw: Sequence[str], table, task_list: Sequence[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for t in task_list:
        right = total = 0
        for g in draw:
            cell = table[g].get(t)
            if cell:
                right += cell[0]
                total += cell[1]
        out[t] = (right / total) if total else None
    main = [out.get(t) for t in tasks.MAIN_NON_EDIT_TASKS]
    out[MACRO] = (sum(main) / len(main)) if all(v is not None for v in main) else None
    return out


def _interval(values: List[float], level: float) -> Tuple[float, float]:
    v = sorted(values)
    alpha = (1.0 - level) / 2.0
    lo = v[int(alpha * len(v))]
    hi = v[min(len(v) - 1, int((1.0 - alpha) * len(v)))]
    return lo, hi


def cluster_bootstrap(
    samples: Sequence[Dict[str, Any]],
    flags: Dict[str, bool],
    other: Optional[Dict[str, bool]] = None,
    n_boot: int = 2000,
    seed: str = "bootstrap-v1",
    level: float = 0.95,
) -> Dict[str, Any]:
    """Percentile intervals over resampled root graphs.

    Returns, per task and for the non-edit macro, the point estimate and interval of `flags`.
    With `other`, also the same for `other` and for the paired difference flags - other,
    plus the share of replicates in which the difference was above zero."""
    groups, task_list, table = _counts(samples, flags)
    if not groups:
        raise ValueError("no main samples to resample")
    names = list(task_list) + [MACRO]
    table_b = _counts(samples, other)[2] if other is not None else None
    point_a = _stats(groups, table, task_list)
    point_b = _stats(groups, table_b, task_list) if table_b is not None else None

    rng = Rng(seed, len(groups), n_boot)
    reps_a: Dict[str, List[float]] = {k: [] for k in names}
    reps_b: Dict[str, List[float]] = {k: [] for k in names}
    reps_d: Dict[str, List[float]] = {k: [] for k in names}
    for _ in range(n_boot):
        draw = [groups[rng.below(len(groups))] for _ in groups]
        a = _stats(draw, table, task_list)
        b = _stats(draw, table_b, task_list) if table_b is not None else None
        for k in names:
            if a[k] is None:
                continue  # a replicate that happened to contain no sample of this task
            reps_a[k].append(a[k])
            if b is not None and b[k] is not None:
                reps_b[k].append(b[k])
                reps_d[k].append(a[k] - b[k])

    def entry(point, reps):
        if point is None or not reps:
            return None
        lo, hi = _interval(reps, level)
        return {"point": point, "lo": lo, "hi": hi}

    out: Dict[str, Any] = {
        "unit": "graph_group_id", "root_graphs": len(groups), "n_boot": n_boot, "seed": seed, "level": level,
        "method": "percentile", "a": {k: entry(point_a[k], reps_a[k]) for k in names},
    }
    if point_b is not None:
        out["b"] = {k: entry(point_b[k], reps_b[k]) for k in names}
        diff: Dict[str, Any] = {}
        for k in names:
            if point_a[k] is None or point_b[k] is None or not reps_d[k]:
                diff[k] = None
                continue
            e = entry(point_a[k] - point_b[k], reps_d[k])
            e["share_above_zero"] = sum(1 for d in reps_d[k] if d > 0) / len(reps_d[k])
            diff[k] = e
        out["a_minus_b"] = diff
    return out
