"""Scores computed from raw responses. Tables in reports must come from here, not by hand.

A response that fails the strict format check is wrong, and is also counted separately.
Pair scores require BOTH members to be right, so a model that always says the same thing
gets no credit for "consistency".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import tasks

MAIN_VARIANTS = ("root", "edit")


def score_sample(sample: Dict[str, Any], response: Optional[str]) -> Dict[str, Any]:
    t = sample["private_truth"]
    return tasks.score_response(sample["public_input"]["query"], t["nodes"], t["edges"], response)


def _mean(flags: List[bool]) -> Optional[float]:
    return (sum(1 for f in flags if f) / len(flags)) if flags else None


def evaluate(samples: List[Dict[str, Any]], responses: Dict[str, Optional[str]], lenient: bool = False) -> Dict[str, Any]:
    """`responses` maps sample_id -> raw model text. A missing entry counts as an empty response.

    With lenient=True each task also gets `lenient_accuracy`: the score when the last JSON
    object found anywhere in the response is used. It is a diagnostic, never the main score."""
    scored = {s["sample_id"]: score_sample(s, responses.get(s["sample_id"])) for s in samples}
    out: Dict[str, Any] = {"samples": len(samples), "missing_responses": sum(1 for s in samples if s["sample_id"] not in responses)}

    per_task: Dict[str, Any] = {}
    for task in tasks.ALL_TASKS:
        rows = [s for s in samples if s["task"] == task and s["metadata"]["image_variant"] in MAIN_VARIANTS]
        if not rows:
            continue
        res = [scored[s["sample_id"]] for s in rows]
        entry: Dict[str, Any] = {
            "n": len(rows),
            "accuracy": _mean([r["correct"] for r in res]),
            "format_failure_rate": _mean([not r["format_ok"] for r in res]),
        }
        if lenient:
            entry["lenient_accuracy"] = _mean(
                [score_sample(s, tasks.lenient_candidate(responses.get(s["sample_id"])))["correct"] for s in rows]
            )
        if task == tasks.REPAIR:
            ok = [r for r in res if r["format_ok"]]
            entry["status_accuracy"] = _mean([bool(r.get("status_correct")) for r in res])
            # Among samples whose true status is "edited": was the proposed edit legal and goal-satisfying?
            edited = [(s, r) for s, r in zip(rows, res) if s["private_truth"]["status"] == "edited"]
            entry["valid_edit_rate"] = _mean([bool(r.get("valid")) and bool(r.get("legal")) for _, r in edited])
            entry["minimal_repair_accuracy"] = _mean([bool(r.get("minimal")) for _, r in edited])
            entry["n_true_edited"] = len(edited)
            entry["n_parsed"] = len(ok)
        if task == tasks.EXTRACTION:
            for key in ("node_precision", "node_recall", "edge_precision", "edge_recall", "direction_accuracy"):
                vals = [r[key] for r in res if r.get(key) is not None]
                entry[key] = (sum(vals) / len(vals)) if vals else None
        per_task[task] = entry
    out["per_task"] = per_task

    if lenient:
        lm = [per_task[t]["lenient_accuracy"] for t in tasks.MAIN_NON_EDIT_TASKS if t in per_task]
        out["non_edit_macro_lenient_accuracy"] = (sum(lm) / len(lm)) if len(lm) == len(tasks.MAIN_NON_EDIT_TASKS) else None
    main = [per_task[t]["accuracy"] for t in tasks.MAIN_NON_EDIT_TASKS if t in per_task]
    out["non_edit_macro_accuracy"] = (sum(main) / len(main)) if len(main) == len(tasks.MAIN_NON_EDIT_TASKS) else None

    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    kind_of: Dict[str, str] = {}
    for s in samples:
        for p in s["pairs"]:
            by_pair.setdefault(p["pair_id"], []).append(s)
            kind_of[p["pair_id"]] = p["kind"] + ":" + s["task"]
    pairs: Dict[str, List[bool]] = {}
    for pid, members in by_pair.items():
        if len(members) != 2:
            continue
        pairs.setdefault(kind_of[pid], []).append(all(scored[m["sample_id"]]["correct"] for m in members))
    out["pair_accuracy"] = {k: {"n": len(v), "both_correct": _mean(v)} for k, v in sorted(pairs.items())}
    return out
