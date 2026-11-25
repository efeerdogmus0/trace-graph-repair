"""Perception/reasoning split of end-to-end arms by how readable each image is (CPU only).

An image counts as *readable* when arm E's extraction of it is strictly exact (same node set and
same directed edge set as the truth, strict JSON). Every end-to-end arm is then scored separately on
samples whose image is readable and on the rest, with root-graph bootstrap intervals, and each arm is
compared with a reference arm inside each subset (same resampled roots).

Readability is a property measured by one other model (E's parser), so it is a proxy for how hard a
drawing is to read, not ground truth about what any other arm perceived.

Reading rule (fixed 2026-11-12, before E's extraction was read back), for arm X vs reference R on the
non-edit macro:
  - too_few:            fewer than MIN_ROOTS root graphs contribute samples to either subset
  - reasoning_gap:      the readable-subset X-R interval lies entirely below 0
                        (X trails R even on drawings a parser reads exactly)
  - perception_gap:     readable-subset interval contains 0 and unreadable-subset interval lies below 0
  - no_gap:             neither interval lies below 0
  - unclear:            anything else
"""
import json
from typing import Any, Dict, Optional, Sequence

from . import bootstrap, parse_solve as ps, tasks

MACRO = bootstrap.MACRO
MIN_ROOTS = 30


def readable_images(rows: Sequence[Dict[str, Any]], extraction_by_image: Dict[str, Optional[str]]) -> Dict[str, bool]:
    """image_id -> strict exact extraction. Raises if an image of `rows` has no extraction."""
    truth: Dict[str, Dict[str, Any]] = {}
    for s in rows:
        truth.setdefault(s["image_id"], s["private_truth"])
    missing = [i for i in truth if i not in extraction_by_image]
    if missing:
        raise ValueError("%d images have no extraction, e.g. %s" % (len(missing), missing[0]))
    out = {}
    for img, t in truth.items():
        sc = tasks.score_response(ps.extraction_query(), t["nodes"], t["edges"], extraction_by_image[img])
        out[img] = bool(sc["correct"])
    return out


def load_extraction(path) -> Dict[str, Optional[str]]:
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["image_id"]] = r["text"]
    return out


def _subset(rows, readable, want):
    return [s for s in rows if readable[s["image_id"]] is want]


def reading(readable_diff: Optional[Dict[str, float]], unreadable_diff: Optional[Dict[str, float]],
            roots_readable: int, roots_unreadable: int, min_roots: int = MIN_ROOTS) -> str:
    if roots_readable < min_roots or roots_unreadable < min_roots or readable_diff is None or unreadable_diff is None:
        return "too_few"
    if readable_diff["hi"] < 0:
        return "reasoning_gap"
    below_u = unreadable_diff["hi"] < 0
    if readable_diff["lo"] <= 0 <= readable_diff["hi"] and below_u:
        return "perception_gap"
    if not below_u:
        return "no_gap"
    return "unclear"


def split_table(rows: Sequence[Dict[str, Any]], readable: Dict[str, bool], flags: Dict[str, Dict[str, bool]],
                ref: str, n_boot: int = 2000) -> Dict[str, Any]:
    main = bootstrap._main_rows(rows)
    imgs = {s["image_id"] for s in main}
    out: Dict[str, Any] = {
        "reference": ref, "n_boot": n_boot, "min_roots": MIN_ROOTS,
        "images": len(imgs), "readable_images": sum(1 for i in imgs if readable[i]),
        "subsets": {}, "arms": {}}
    subsets = {"readable": _subset(main, readable, True), "unreadable": _subset(main, readable, False)}
    for name, sub in subsets.items():
        out["subsets"][name] = {"samples": len(sub), "root_graphs": len({s["graph_group_id"] for s in sub})}
    for label, f in flags.items():
        entry: Dict[str, Any] = {}
        for name, sub in subsets.items():
            if not sub:
                entry[name] = None
                continue
            res = bootstrap.cluster_bootstrap(sub, f, None if label == ref else flags[ref], n_boot=n_boot)
            entry[name] = {"acc": res["a"][MACRO], "vs_ref": (res.get("a_minus_b") or {}).get(MACRO),
                           "per_task": {k: v for k, v in res["a"].items() if k != MACRO}}
        if label != ref:
            r, u = entry["readable"], entry["unreadable"]
            entry["reading"] = reading(r and r["vs_ref"], u and u["vs_ref"],
                                       out["subsets"]["readable"]["root_graphs"], out["subsets"]["unreadable"]["root_graphs"])
        out["arms"][label] = entry
    return out
