"""Choosing `max_new_tokens` from dev outputs (protocol open items 1 and 5).

Greedy decoding is deterministic, so the response a model gives under a limit L is the first
L tokens of the response it gives under any larger limit L_max (up to rare floating-point
divergence, which is checked when two limits were actually run). One dev run at L_max with
the generated token ids saved therefore gives, for every candidate L <= L_max:

- the truncation rate (responses that did not finish within L tokens), per task;
- the strict and lenient score of the prefix response, per task.

Rule (fixed before any arm is scored on dev with it, DECISIONS 2026-11-12):
  per arm, the chosen limit is the smallest candidate at which every task's truncation rate
  is <= TRUNC_MAX; if no candidate qualifies, the largest candidate is used and the residual
  truncation is reported. Responses are never re-scored with knowledge of the label.

This module is pure Python; the caller supplies `decode(ids) -> str` (the processor's
tokenizer with skip_special_tokens=True), so tests run without torch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from . import metrics

CANDIDATES = (96, 192, 384, 768)
TRUNC_MAX = 0.02


def finished_within(row: Dict[str, Any], limit: int) -> bool:
    """True if the response ended (EOS) before reaching `limit` new tokens.

    `new_tokens` counts generated tokens including the end token; a response that hit the
    run's own limit (`hit_limit`) never finished.
    """
    if row.get("hit_limit"):
        return False
    return int(row["new_tokens"]) < limit


def prefix_text(row: Dict[str, Any], limit: int, decode: Callable[[Sequence[int]], str]) -> str:
    ids = row["ids"]
    if finished_within(row, limit):
        return row["text"]
    return decode(ids[:limit])


def scan(samples: List[Dict[str, Any]], rows: List[Dict[str, Any]], run_limit: int,
         decode: Callable[[Sequence[int]], str], candidates: Sequence[int] = CANDIDATES) -> Dict[str, Any]:
    """Truncation and score at each candidate limit <= run_limit.

    `rows` are prediction lines with sample_id, task, text, new_tokens, hit_limit, ids.
    """
    by_id = {r["sample_id"]: r for r in rows}
    missing = [s["sample_id"] for s in samples if s["sample_id"] not in by_id]
    if missing:
        raise ValueError("%d samples have no prediction (first: %s)" % (len(missing), missing[0]))
    for r in rows:
        if "ids" not in r:
            raise ValueError("prediction %s has no token ids; rerun eval_arm.py" % r["sample_id"])
        if len(r["ids"]) != int(r["new_tokens"]):
            raise ValueError("prediction %s: len(ids) != new_tokens" % r["sample_id"])
    out: Dict[str, Any] = {"run_limit": run_limit, "trunc_max": TRUNC_MAX, "limits": {}}
    for L in candidates:
        if L > run_limit:
            continue
        responses = {s["sample_id"]: prefix_text(by_id[s["sample_id"]], L, decode) for s in samples}
        res = metrics.evaluate(samples, responses, lenient=True)
        trunc: Dict[str, float] = {}
        for task in sorted({s["task"] for s in samples}):
            ids = [s["sample_id"] for s in samples if s["task"] == task]
            trunc[task] = sum(not finished_within(by_id[i], L) for i in ids) / len(ids)
        out["limits"][str(L)] = {
            "truncation_rate": trunc,
            "max_truncation_rate": max(trunc.values()),
            "non_edit_macro": res["non_edit_macro_accuracy"],
            "non_edit_macro_lenient": res.get("non_edit_macro_lenient_accuracy"),
            "per_task_accuracy": {t: e["accuracy"] for t, e in res["per_task"].items()},
            "per_task_format_failure": {t: e.get("format_failure_rate") for t, e in res["per_task"].items()},
        }
    out["chosen_limit"] = choose(out)
    return out


def choose(scan_result: Dict[str, Any]) -> Optional[int]:
    limits = sorted(int(k) for k in scan_result["limits"])
    if not limits:
        return None
    for L in limits:
        if scan_result["limits"][str(L)]["max_truncation_rate"] <= scan_result["trunc_max"]:
            return L
    return limits[-1]


def check_prefix_consistency(short_rows: List[Dict[str, Any]], long_rows: List[Dict[str, Any]],
                             short_limit: int) -> Dict[str, int]:
    """Compare a real run at `short_limit` with the prefixes of a longer run (same inputs).

    Returns counts; `mismatch` > 0 means greedy outputs diverged (numerics), which is reported.
    """
    long_by = {r["sample_id"]: r for r in long_rows}
    same = mismatch = 0
    for r in short_rows:
        lr = long_by.get(r["sample_id"])
        if lr is None:
            continue
        if list(r["ids"]) == list(lr["ids"][:short_limit])[:len(r["ids"])] and (
                r.get("hit_limit") or len(r["ids"]) == len(lr["ids"])):
            same += 1
        else:
            mismatch += 1
    return {"same": same, "mismatch": mismatch}
