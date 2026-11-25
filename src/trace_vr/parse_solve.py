"""Arm E: parse the graph from the image once, then answer every question with the exact solver.

The model only sees the extraction prompt (task `graph_extraction`) for each image. Its
predicted graph is cleaned by fixed rules, the exact solver answers each query on that
predicted graph, and the answer is written as a canonical JSON response. That response is
then scored against the TRUE graph by the same `metrics.evaluate` used for every arm, so E
is compared on identical samples and identical scoring.

Cleaning rules (decided before any E result exists; see DECISIONS 2026-11-11):
- strict extraction parse; if it fails, the lenient candidate is NOT used for the main
  score (a separate lenient variant is reported, as for the other arms);
- node set = listed nodes + edge endpoints + labels named in the query (a node the model
  did not list is an isolated node, not an error);
- self-loops dropped, duplicate edges merged; antiparallel pairs are kept as predicted;
- a critical-edge question whose premise fails on the predicted graph (src cannot reach dst,
  or the edge is absent) has no defined answer: the response is None and scores as wrong.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import tasks

UNANSWERABLE = "premise_fails_on_predicted_graph"


def extraction_query() -> Dict[str, Any]:
    return {"type": tasks.EXTRACTION}


def extraction_prompt() -> str:
    return tasks.render_instruction(extraction_query())


def _query_labels(query: Dict[str, Any]) -> List[str]:
    kind = query["type"]
    if kind == tasks.REPAIR:
        pairs = list(query["must_reach"]) + list(query["must_not_reach"]) + list(query["allowed_additions"])
        return [x for p in pairs for x in p]
    out = [query["src"], query["dst"]]
    if kind == tasks.CRITICAL_EDGE:
        out += list(query["edge"])
    return out


def clean_graph(value: Dict[str, Any], query: Dict[str, Any]) -> Tuple[List[str], List[List[str]], Dict[str, int]]:
    """Apply the cleaning rules. Returns (nodes, edges, counts of each rule that fired)."""
    nodes: List[str] = []
    for x in list(value["nodes"]) + [y for e in value["edges"] for y in e] + _query_labels(query):
        if x not in nodes:
            nodes.append(x)
    edges: List[List[str]] = []
    counts = {"self_loops_dropped": 0, "duplicate_edges_merged": 0,
              "nodes_added_from_edges_or_query": len(nodes) - len(set(value["nodes"]))}
    for a, b in value["edges"]:
        if a == b:
            counts["self_loops_dropped"] += 1
        elif [a, b] in edges:
            counts["duplicate_edges_merged"] += 1
        else:
            edges.append([a, b])
    return nodes, edges, counts


def answer_from_extraction(query: Dict[str, Any], extraction_text: Optional[str],
                           lenient: bool = False) -> Tuple[Optional[str], Dict[str, Any]]:
    """Canonical response text for `query`, solved on the graph parsed from `extraction_text`.

    Returns (response or None, info). info["reason"] is None on success."""
    text = tasks.lenient_candidate(extraction_text) if lenient else extraction_text
    parsed = tasks.parse_response(tasks.EXTRACTION, text)
    if not parsed.ok:
        return None, {"reason": "extraction_format:" + str(parsed.error)}
    nodes, edges, counts = clean_graph(parsed.value, query)
    try:
        answer = tasks.solve_query(query, nodes, edges)
    except ValueError:
        return None, {"reason": UNANSWERABLE, **counts}
    return tasks.format_target(answer), {"reason": None, **counts}


def responses_for_samples(samples: Sequence[Dict[str, Any]], extraction_by_image: Dict[str, Optional[str]],
                          lenient: bool = False) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """sample_id -> derived response, plus a count of failure reasons over all samples."""
    out: Dict[str, Optional[str]] = {}
    reasons: Dict[str, int] = {}
    for s in samples:
        resp, info = answer_from_extraction(s["public_input"]["query"], extraction_by_image.get(s["image_id"]), lenient)
        out[s["sample_id"]] = resp
        key = info["reason"] or "answered"
        reasons[key] = reasons.get(key, 0) + 1
    return out, reasons
