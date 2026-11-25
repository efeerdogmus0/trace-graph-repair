"""Model-free shortcut probes: how far can one get without looking at the image?

Three kinds of numbers per task and split:

* constant answers (always "true", always "impossible", ...);
* simple text heuristics for repair (e.g. "add the edge that the goal names");
* the text-only ceiling: the best accuracy of ANY predictor that sees only the
  instruction text, i.e. for each distinct instruction the single response that is right
  for the most samples. With answer-flipping twins this is 50% by construction, and a
  higher value means some instructions lack a twin or leak the answer.

These are computed with the real scorer, so they are directly comparable to model scores.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from . import tasks


def _correct(sample: Dict[str, Any], response: str) -> bool:
    t = sample["private_truth"]
    return bool(tasks.score_response(sample["public_input"]["query"], t["nodes"], t["edges"], response)["correct"])


def _rate(samples: List[Dict[str, Any]], responder) -> float:
    return sum(1 for s in samples if _correct(s, responder(s))) / len(samples)


def _edit(op: str, pair) -> str:
    return tasks.format_target({"status": "edited", "edits": [{"op": op, "src": pair[0], "dst": pair[1]}]})


def _direct_add(sample: Dict[str, Any]) -> str:
    q = sample["public_input"]["query"]
    for pair in q["must_reach"]:
        if pair in q["allowed_additions"]:
            return _edit("add_edge", pair)
    return '{"status":"impossible","edits":[]}'


def _candidates(sample: Dict[str, Any]) -> List[str]:
    """Every response that is fully correct for this sample."""
    if sample["task"] == tasks.REPAIR:
        t = sample["private_truth"]
        if t["status"] != "edited":
            return [tasks.format_target({"status": t["status"], "edits": []})]
        return [tasks.format_target({"status": "edited", "edits": sol}) for sol in t["all_minimal_solutions"]]
    return [tasks.format_target(sample["training_target"])]


def text_only_ceiling(samples: List[Dict[str, Any]]) -> float:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        groups.setdefault(s["public_input"]["instruction"], []).append(s)
    right = 0
    for members in groups.values():
        options = sorted({c for s in members for c in _candidates(s)})
        right += max(sum(1 for s in members if c in _candidates(s)) for c in options)
    return right / len(samples)


def report(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Probes on the main samples only (root and edited images).

    Relabelled and restyled copies repeat a question with the same answer on purpose, so
    they are left out here; they are scored through the invariance pairs instead.
    """
    samples = [s for s in samples if s["metadata"]["image_variant"] in ("root", "edit")]
    out: Dict[str, Any] = {}
    for task in (tasks.REPAIR, tasks.REACHABILITY, tasks.CRITICAL_EDGE, tasks.EDGE_EXISTS):
        rows = [s for s in samples if s["task"] == task]
        if not rows:
            continue
        entry: Dict[str, Any] = {"samples": len(rows), "text_only_ceiling": round(text_only_ceiling(rows), 4)}
        if task == tasks.REPAIR:
            entry["always_unchanged"] = round(_rate(rows, lambda s: '{"status":"unchanged","edits":[]}'), 4)
            entry["always_impossible"] = round(_rate(rows, lambda s: '{"status":"impossible","edits":[]}'), 4)
            entry["add_edge_named_by_goal_else_impossible"] = round(_rate(rows, _direct_add), 4)
            entry["add_first_listed_edge"] = round(
                _rate(rows, lambda s: _edit("add_edge", s["public_input"]["query"]["allowed_additions"][0])), 4
            )
            entry["remove_edge_named_by_first_forbidden_pair"] = round(
                _rate(rows, lambda s: _edit("remove_edge", s["public_input"]["query"]["must_not_reach"][0])), 4
            )
        else:
            entry["always_true"] = round(_rate(rows, lambda s: json.dumps({"answer": True})), 4)
            entry["always_false"] = round(_rate(rows, lambda s: json.dumps({"answer": False})), 4)
        out[task] = entry
    return out
