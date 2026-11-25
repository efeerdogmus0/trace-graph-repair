"""Error analysis from per-sample correctness flags (no model, no raw responses needed).

For yes/no tasks a strict-correct flag plus the true answer gives the model's answer when
the format was valid, so answer bias ("says true whatever the image shows") is recoverable.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from . import metrics, tasks


def _truth_label(s: Dict[str, Any]) -> str:
    t = s["training_target"]
    return str(t["status"]) if s["task"] == tasks.REPAIR else str(t.get("answer"))


def _size_bucket(s: Dict[str, Any]) -> str:
    n = s["metadata"]["node_count"]
    return "nodes<=8" if n <= 8 else ("nodes9-12" if n <= 12 else "nodes>=13")


FACETS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "truth": _truth_label,
    "image_variant": lambda s: s["metadata"]["image_variant"],
    "graph_family": lambda s: s["graph_family"],
    "size": _size_bucket,
    "layout": lambda s: s["metadata"]["layout"],
    "style": lambda s: s["metadata"]["style_id"],
}


def breakdown(samples: Sequence[Dict[str, Any]], flags: Dict[str, bool]) -> Dict[str, Any]:
    """accuracy and n per task, facet and facet value (main image variants only)."""
    out: Dict[str, Any] = {}
    for s in samples:
        if s["metadata"]["image_variant"] not in metrics.MAIN_VARIANTS:
            continue
        for facet, fn in FACETS.items():
            cell = out.setdefault(s["task"], {}).setdefault(facet, {}).setdefault(fn(s), {"n": 0, "right": 0})
            cell["n"] += 1
            cell["right"] += 1 if flags[s["sample_id"]] else 0
    for facets in out.values():
        for values in facets.values():
            for cell in values.values():
                cell["accuracy"] = cell["right"] / cell["n"]
    return out


def answer_bias(samples: Sequence[Dict[str, Any]], flags: Dict[str, bool],
                format_ok: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    """Per yes/no task: how often the model said true, next to how often true was right.

    Samples whose response failed the format check have no recoverable answer and are
    counted in `no_answer`. Without `format_ok` every wrong response is assumed to be the
    opposite boolean, which is only safe when the format failure rate is zero."""
    out: Dict[str, Any] = {}
    for task in tasks.BOOLEAN_TASKS:
        rows = [s for s in samples if s["task"] == task and s["metadata"]["image_variant"] in metrics.MAIN_VARIANTS]
        if not rows:
            continue
        said_true = truth_true = no_answer = 0
        for s in rows:
            truth = bool(s["training_target"]["answer"])
            truth_true += truth
            if format_ok is not None and not format_ok[s["sample_id"]]:
                no_answer += 1
                continue
            said_true += truth if flags[s["sample_id"]] else (not truth)
        answered = len(rows) - no_answer
        out[task] = {"n": len(rows), "no_answer": no_answer, "truth_true_rate": truth_true / len(rows),
                     "said_true_rate": (said_true / answered) if answered else None}
    return out


def pair_table(samples: Sequence[Dict[str, Any]], flags: Dict[str, bool], kind: str = "critical") -> Dict[str, Any]:
    """For twin pairs of one kind: both right / only one right / both wrong, per task."""
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        for p in s["pairs"]:
            if p["kind"] == kind:
                by_pair.setdefault(p["pair_id"], []).append(s)
    out: Dict[str, Any] = {}
    for members in by_pair.values():
        if len(members) != 2:
            continue
        cell = out.setdefault(members[0]["task"], {"pairs": 0, "both_right": 0, "one_right": 0, "both_wrong": 0})
        right = sum(1 for m in members if flags[m["sample_id"]])
        cell["pairs"] += 1
        cell[("both_wrong", "one_right", "both_right")[right]] += 1
    return out


# --- Density-matched slice (protocol item 7): size-OOD roots are sparser per node than
# large random graphs would be, because layout rejection removes dense ones. Scoring the OOD
# split only on roots whose edges-per-node lies inside the reference (ID) range separates a
# size effect from a density effect. Density is an input property (read from the root
# drawing's metadata), not a label.

def root_density(samples: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """edges / nodes of each root graph, keyed by graph_group_id (from the root image)."""
    out: Dict[str, float] = {}
    for s in samples:
        md = s["metadata"]
        if md["image_variant"] == "root" and s["graph_group_id"] not in out:
            out[s["graph_group_id"]] = md["edge_count"] / md["node_count"]
    return out


def quantile(values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile (deterministic, no interpolation)."""
    v = sorted(values)
    if not v:
        raise ValueError("no values")
    k = max(1, min(len(v), int(-(-q * len(v) // 1))))  # ceil(q*n), clipped to [1, n]
    return v[k - 1]


def density_window(ref: Dict[str, float], lo_q: float = 0.05, hi_q: float = 0.95) -> Dict[str, float]:
    vals = list(ref.values())
    return {"lo": quantile(vals, lo_q), "hi": quantile(vals, hi_q), "lo_q": lo_q, "hi_q": hi_q,
            "ref_roots": len(vals), "ref_mean": sum(vals) / len(vals)}


def density_matched(samples: Sequence[Dict[str, Any]], window: Dict[str, float],
                    flags: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    """Roots of `samples` inside [lo, hi] edges-per-node; with flags, per-task accuracy on all
    main samples and on the matched ones (non-edit macro = mean of reachability and critical edge)."""
    dens = root_density(samples)
    keep = {g for g, d in dens.items() if window["lo"] <= d <= window["hi"]}
    res: Dict[str, Any] = {
        "window": window, "roots": len(dens), "matched_roots": len(keep),
        "mean_density_all": sum(dens.values()) / len(dens) if dens else None,
        "mean_density_matched": (sum(dens[g] for g in keep) / len(keep)) if keep else None,
        "below": sum(1 for d in dens.values() if d < window["lo"]),
        "above": sum(1 for d in dens.values() if d > window["hi"])}
    if flags is None:
        return res
    acc: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        if s["metadata"]["image_variant"] not in metrics.MAIN_VARIANTS:
            continue
        for subset in ("all", "matched"):
            if subset == "matched" and s["graph_group_id"] not in keep:
                continue
            cell = acc.setdefault(subset, {}).setdefault(s["task"], {"n": 0, "right": 0})
            cell["n"] += 1
            cell["right"] += 1 if flags[s["sample_id"]] else 0
    for per_task in acc.values():
        for cell in per_task.values():
            cell["accuracy"] = cell["right"] / cell["n"]
        ne = [per_task[t]["accuracy"] for t in (tasks.REACHABILITY, tasks.CRITICAL_EDGE) if t in per_task]
        per_task["non_edit_macro"] = sum(ne) / len(ne) if len(ne) == 2 else None
    res["accuracy"] = acc
    return res


# --- Invariance pairs (protocol section 5): same graph and question, relabelled or restyled
# image, same answer. The root member is scored in the main run, the copy in the invariance run.

def invariance_table(main_rows: Sequence[Dict[str, Any]], main_flags: Dict[str, bool],
                     inv_rows: Sequence[Dict[str, Any]], inv_flags: Dict[str, bool]) -> Dict[str, Any]:
    """Per pair kind and task: pairs, both right, root right, copy right, and how often the two
    answers have the same correctness (for yes/no questions with valid format, the same answer)."""
    by_id = {s["sample_id"]: s for s in main_rows}
    out: Dict[str, Any] = {}
    for v in inv_rows:
        for p in v["pairs"]:
            if p["role"] != "variant" or not p["kind"].startswith("invariant_"):
                continue
            base = next((b for b in by_id.values()
                         if any(q["pair_id"] == p["pair_id"] and q["role"] == "base" for q in b["pairs"])), None)
            if base is None:
                raise ValueError("pair partner of %s is not in the main rows" % v["sample_id"])
            b_ok, v_ok = main_flags[base["sample_id"]], inv_flags[v["sample_id"]]
            cell = out.setdefault(p["kind"], {}).setdefault(v["task"], {"pairs": 0, "both_right": 0, "root_right": 0,
                                                                          "copy_right": 0, "same_correctness": 0})
            cell["pairs"] += 1
            cell["both_right"] += b_ok and v_ok
            cell["root_right"] += b_ok
            cell["copy_right"] += v_ok
            cell["same_correctness"] += b_ok == v_ok
    for tasks_ in out.values():
        for cell in tasks_.values():
            for k in ("both_right", "root_right", "copy_right", "same_correctness"):
                cell[k + "_rate"] = cell[k] / cell["pairs"]
    return out
