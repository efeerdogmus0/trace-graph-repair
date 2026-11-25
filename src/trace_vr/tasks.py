"""Task definitions at the label level: instructions, output schemas, parsing and scoring.

A `query` is the structured form of exactly what the instruction text says. It is public:
it contains nothing that is not also written in the instruction. The true graph is never
part of a query.

Parsing policy (primary score): after stripping surrounding whitespace and at most one
Markdown code fence, the response must be a single JSON object that matches the declared
schema exactly. Anything else is a format failure and counts as wrong. Responses are never
repaired using the ground truth.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graphs import Graph
from . import semantics as sem

REPAIR = "minimal_graph_repair"
REACHABILITY = "reachability"
CRITICAL_EDGE = "critical_edge"
EDGE_EXISTS = "edge_exists"
SHORTEST_PATH = "shortest_path_length"
EXTRACTION = "graph_extraction"
EDIT_OUTCOME = "edit_outcome"  # D2 auxiliary only: does the repair goal hold after a given edit?
INFILL = "graph_infilling"  # arm C only: MCDGraph-style masked infilling (missing edges of a listed graph)
MATCH = "graph_matching"  # arm C only: one-image stand-in for MCDGraph's contrastive discrimination

ALL_TASKS = (REPAIR, REACHABILITY, CRITICAL_EDGE, EDGE_EXISTS, SHORTEST_PATH, EXTRACTION, EDIT_OUTCOME,
             INFILL, MATCH)
MAIN_NON_EDIT_TASKS = (REACHABILITY, CRITICAL_EDGE)
BOOLEAN_TASKS = (REACHABILITY, CRITICAL_EDGE, EDGE_EXISTS, EDIT_OUTCOME, MATCH)

# Single uppercase letters without the easily confused I, O and Q.
LABEL_ALPHABET = "ABCDEFGHJKLMNPRSTUVWXYZ"

STATUSES = ("edited", "unchanged", "impossible")
OPS = (sem.ADD, sem.REMOVE)


# ------------------------------------------------------------------ instructions
def _arrow(pair: Sequence[str]) -> str:
    return "%s->%s" % (pair[0], pair[1])


_INTRO = "Look at the directed graph in the image. Arrows show edge directions."
_JSON_ONLY = "Answer with a single JSON object and nothing else."


def render_instruction(query: Dict[str, Any]) -> str:
    """Deterministic instruction text for a query. One template per task (see protocol)."""
    kind = query["type"]
    if kind == REPAIR:
        goals = ["- %s must be able to reach %s." % (s, t) for s, t in query["must_reach"]]
        goals += ["- %s must NOT be able to reach %s." % (s, t) for s, t in query["must_not_reach"]]
        adds = ", ".join(_arrow(p) for p in query["allowed_additions"])
        return "\n".join(
            [
                _INTRO,
                "Goal (reaching means following arrows along one or more edges):",
                *goals,
                "You may change at most %d edge. Edges you are allowed to add: %s. "
                "You may remove any edge that exists in the image." % (query["max_edits"], adds),
                'If the goal already holds, answer {"status":"unchanged","edits":[]}.',
                'If no single allowed change achieves the goal, answer {"status":"impossible","edits":[]}.',
                'Otherwise answer {"status":"edited","edits":[{"op":"add_edge","src":"<node>","dst":"<node>"}]} '
                'using "op":"add_edge" or "op":"remove_edge".',
                _JSON_ONLY,
            ]
        )
    if kind == REACHABILITY:
        return (
            "%s\nFollowing the arrow directions along one or more edges, is there a path from node %s "
            'to node %s?\nAnswer {"answer": true} or {"answer": false}. %s'
            % (_INTRO, query["src"], query["dst"], _JSON_ONLY)
        )
    if kind == CRITICAL_EDGE:
        return (
            "%s\nNode %s can currently reach node %s by following arrows. If the edge %s were removed, "
            "would %s no longer be able to reach %s?\n"
            'Answer {"answer": true} if the connection would be lost, or {"answer": false} if %s could '
            "still reach %s. %s"
            % (
                _INTRO,
                query["src"],
                query["dst"],
                _arrow(query["edge"]),
                query["src"],
                query["dst"],
                query["src"],
                query["dst"],
                _JSON_ONLY,
            )
        )
    if kind == EDGE_EXISTS:
        return (
            "%s\nIs there an arrow that goes directly from node %s to node %s?\n"
            'Answer {"answer": true} or {"answer": false}. %s'
            % (_INTRO, query["src"], query["dst"], _JSON_ONLY)
        )
    if kind == SHORTEST_PATH:
        return (
            "%s\nWhat is the smallest number of arrows you must follow to get from node %s to node %s?\n"
            'Answer {"answer": <integer>}, or {"answer": null} if node %s cannot be reached from node %s. %s'
            % (_INTRO, query["src"], query["dst"], query["dst"], query["src"], _JSON_ONLY)
        )
    if kind == EDIT_OUTCOME:
        goals = ["- %s must be able to reach %s." % (a, b) for a, b in query["must_reach"]]
        goals += ["- %s must NOT be able to reach %s." % (a, b) for a, b in query["must_not_reach"]]
        e = query["edit"]
        change = ("add the edge %s->%s" if e["op"] == sem.ADD else "remove the edge %s->%s") % (e["src"], e["dst"])
        return "\n".join(
            [
                _INTRO,
                "Goal (reaching means following arrows along one or more edges):",
                *goals,
                "Suppose you %s and change nothing else. Would the goal hold afterwards?" % change,
                'Answer {"answer": true} or {"answer": false}. %s' % _JSON_ONLY,
            ]
        )
    if kind == EXTRACTION:
        return (
            "%s\nList every node label and every directed edge shown in the image.\n"
            'Answer {"nodes":["<node>", ...],"edges":[["<src>","<dst>"], ...]}. %s' % (_INTRO, _JSON_ONLY)
        )
    if kind == INFILL:
        shown = ", ".join(_arrow(p) for p in query["shown_edges"])
        return (
            "%s\nSome of its directed edges are listed here: %s. Exactly %d edge%s shown in the image "
            "%s missing from this list.\nList the missing edges.\n"
            'Answer {"edges":[["<src>","<dst>"], ...]}. %s'
            % (_INTRO, shown, query["n_missing"], "" if query["n_missing"] == 1 else "s",
               "is" if query["n_missing"] == 1 else "are", _JSON_ONLY)
        )
    if kind == MATCH:
        listed = ", ".join(_arrow(p) for p in query["edges"])
        return (
            "%s\nA graph is described in text with these directed edges: %s. Does this list contain "
            "exactly the edges drawn in the image, each with the same direction?\n"
            'Answer {"answer": true} or {"answer": false}. %s' % (_INTRO, listed, _JSON_ONLY)
        )
    raise KeyError("unknown task type: %r" % (kind,))


_KEY_ORDER = ("status", "edits", "op", "src", "dst", "answer", "nodes", "edges")


def _ordered(obj: Any) -> Any:
    if isinstance(obj, dict):
        keys = sorted(obj, key=lambda k: (_KEY_ORDER.index(k) if k in _KEY_ORDER else len(_KEY_ORDER), k))
        return {k: _ordered(obj[k]) for k in keys}
    if isinstance(obj, list):
        return [_ordered(x) for x in obj]
    return obj


def format_target(obj: Any) -> str:
    """Canonical compact JSON used as the training target string.

    Keys are written in the order the instruction shows them (status before edits, op/src/dst),
    whatever order the object was stored in.
    """
    return json.dumps(_ordered(obj), separators=(",", ":"), ensure_ascii=True)


# ------------------------------------------------------------------------ parsing
class Parsed:
    __slots__ = ("ok", "value", "error")

    def __init__(self, ok: bool, value: Any = None, error: Optional[str] = None) -> None:
        self.ok = ok
        self.value = value
        self.error = error

    def __repr__(self) -> str:
        return "Parsed(ok=%r, value=%r, error=%r)" % (self.ok, self.value, self.error)


_FENCE = re.compile(r"^```[A-Za-z]*[ \t]*\r?\n(.*?)\r?\n?```$", re.DOTALL)


def _load_object(text: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if text is None or not str(text).strip():
        return None, "empty"
    s = str(text).strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except (ValueError, RecursionError):
        return None, "invalid_json"
    if not isinstance(obj, dict):
        return None, "not_an_object"
    return obj, None


def _label(x: Any) -> Optional[str]:
    return x.strip() if isinstance(x, str) and x.strip() else None


def parse_response(task: str, text: Optional[str]) -> Parsed:
    """Strict schema parse. Returns a normalized JSON-able value or a format error code."""
    obj, err = _load_object(text)
    if obj is None:
        return Parsed(False, error=err)
    keys = set(obj.keys())
    if task == REPAIR:
        if keys != {"status", "edits"}:
            return Parsed(False, error="wrong_keys")
        if obj["status"] not in STATUSES:
            return Parsed(False, error="bad_status")
        if not isinstance(obj["edits"], list):
            return Parsed(False, error="edits_not_list")
        edits = []
        for e in obj["edits"]:
            if not isinstance(e, dict) or set(e.keys()) != {"op", "src", "dst"}:
                return Parsed(False, error="bad_edit")
            src, dst = _label(e["src"]), _label(e["dst"])
            if e["op"] not in OPS or src is None or dst is None:
                return Parsed(False, error="bad_edit")
            edits.append({"op": e["op"], "src": src, "dst": dst})
        return Parsed(True, {"status": obj["status"], "edits": edits})
    if task in BOOLEAN_TASKS:
        if keys != {"answer"}:
            return Parsed(False, error="wrong_keys")
        if type(obj["answer"]) is not bool:
            return Parsed(False, error="answer_not_boolean")
        return Parsed(True, {"answer": obj["answer"]})
    if task == SHORTEST_PATH:
        if keys != {"answer"}:
            return Parsed(False, error="wrong_keys")
        a = obj["answer"]
        if a is not None and (type(a) is not int or a < 0):
            return Parsed(False, error="answer_not_integer_or_null")
        return Parsed(True, {"answer": a})
    if task == EXTRACTION:
        if keys != {"nodes", "edges"}:
            return Parsed(False, error="wrong_keys")
        if not isinstance(obj["nodes"], list) or not isinstance(obj["edges"], list):
            return Parsed(False, error="bad_lists")
        nodes = [_label(x) for x in obj["nodes"]]
        if any(x is None for x in nodes):
            return Parsed(False, error="bad_node")
        edges = []
        for e in obj["edges"]:
            if not isinstance(e, list) or len(e) != 2:
                return Parsed(False, error="bad_edge")
            a, b = _label(e[0]), _label(e[1])
            if a is None or b is None:
                return Parsed(False, error="bad_edge")
            edges.append([a, b])
        return Parsed(True, {"nodes": nodes, "edges": edges})
    if task == INFILL:
        if keys != {"edges"} or not isinstance(obj["edges"], list):
            return Parsed(False, error="wrong_keys" if keys != {"edges"} else "bad_lists")
        edges = []
        for e in obj["edges"]:
            if not isinstance(e, list) or len(e) != 2:
                return Parsed(False, error="bad_edge")
            a, b = _label(e[0]), _label(e[1])
            if a is None or b is None:
                return Parsed(False, error="bad_edge")
            edges.append([a, b])
        return Parsed(True, {"edges": sorted(edges)})
    raise KeyError("unknown task type: %r" % (task,))


def lenient_candidate(text: Optional[str]) -> Optional[str]:
    """Last balanced {...} span in the text that is valid JSON, or None.

    Diagnostic only: lets us see what a model that wraps its answer in prose "meant".
    The primary score never uses this.
    """
    if not text:
        return None
    s = str(text)
    ends = [i for i, c in enumerate(s) if c == "}"]
    for end in reversed(ends):
        depth = 0
        for start in range(end, -1, -1):
            if s[start] == "}":
                depth += 1
            elif s[start] == "{":
                depth -= 1
                if depth == 0:
                    chunk = s[start:end + 1]
                    try:
                        if isinstance(json.loads(chunk), dict):
                            return chunk
                    except ValueError:
                        pass
                    break
    return None


# ------------------------------------------------------------------------ solving
def _index(nodes: Sequence[str]) -> Dict[str, int]:
    return {label: i for i, label in enumerate(nodes)}


def build_graph(nodes: Sequence[str], edges: Sequence[Sequence[str]]) -> Tuple[Graph, Dict[str, int]]:
    idx = _index(nodes)
    return Graph(len(nodes), [(idx[a], idx[b]) for a, b in edges]), idx


def _pairs(idx: Dict[str, int], pairs: Sequence[Sequence[str]]) -> List[Tuple[int, int]]:
    return [(idx[a], idx[b]) for a, b in pairs]


def solve_repair_labels(query: Dict[str, Any], nodes: Sequence[str], edges: Sequence[Sequence[str]]):
    g, idx = build_graph(nodes, edges)
    sol = sem.solve_repair(
        g,
        _pairs(idx, query["must_reach"]),
        _pairs(idx, query["must_not_reach"]),
        _pairs(idx, query["allowed_additions"]),
    )
    return g, idx, sol


def edits_to_labels(solution: Sequence[Tuple[str, int, int]], nodes: Sequence[str]) -> List[Dict[str, str]]:
    return [{"op": op, "src": nodes[u], "dst": nodes[v]} for op, u, v in solution]


def solve_query(query: Dict[str, Any], nodes: Sequence[str], edges: Sequence[Sequence[str]]) -> Any:
    """Exact answer for a query on a *given* graph (true graph, or one predicted from an image).

    For repair the first minimal solution in sorted order is returned. When the graph is a
    prediction, the answer is only as good as that prediction.
    """
    kind = query["type"]
    if kind == EXTRACTION:
        return {"nodes": sorted(nodes), "edges": sorted([a, b] for a, b in edges)}
    if kind == INFILL:
        shown = {tuple(e) for e in query["shown_edges"]}
        return {"edges": sorted([a, b] for a, b in edges if (a, b) not in shown)}
    if kind == MATCH:
        return {"answer": sorted(tuple(e) for e in query["edges"]) == sorted(tuple(e) for e in edges)}
    if kind == EDIT_OUTCOME:
        g, idx = build_graph(nodes, edges)
        e = query["edit"]
        g2 = sem.apply_action(g, (e["op"], idx[e["src"]], idx[e["dst"]]))
        return {"answer": sem.satisfied(g2, _pairs(idx, query["must_reach"]), _pairs(idx, query["must_not_reach"]))}
    if kind == REPAIR:
        _, _, sol = solve_repair_labels(query, nodes, edges)
        edits = edits_to_labels(sol.solutions[0], nodes) if sol.status == "edited" else []
        return {"status": sol.status, "edits": edits}
    g, idx = build_graph(nodes, edges)
    s, t = idx[query["src"]], idx[query["dst"]]
    if kind == REACHABILITY:
        return {"answer": g.reaches(s, t)}
    if kind == CRITICAL_EDGE:
        d, x = query["edge"]
        return {"answer": sem.critical_answer(g, s, t, (idx[d], idx[x]))}
    if kind == EDGE_EXISTS:
        return {"answer": g.has_edge(s, t)}
    if kind == SHORTEST_PATH:
        return {"answer": g.shortest_path_len(s, t)}
    raise KeyError("unknown task type: %r" % (kind,))


# ------------------------------------------------------------------------ scoring
def _score_repair(query, nodes, edges, value) -> Dict[str, Any]:
    g, idx, sol = solve_repair_labels(query, nodes, edges)
    res: Dict[str, Any] = {
        "correct": False,
        "status_correct": value["status"] == sol.status,
        "legal": None,
        "valid": None,
        "minimal": None,
        "error": None,
    }
    edits = value["edits"]
    if value["status"] != "edited":
        if edits:
            res["error"] = "edits_given_with_non_edited_status"
            return res
        res["correct"] = res["status_correct"]
        if value["status"] == "unchanged":
            res["legal"] = True
            res["valid"] = sol.status == "unchanged"
            res["minimal"] = res["valid"]
        return res
    if not 1 <= len(edits) <= int(query["max_edits"]):
        res["error"] = "edit_count_out_of_budget"
        return res
    allowed = set(_pairs(idx, query["allowed_additions"]))
    g2 = g
    res["legal"] = False
    for e in edits:
        if e["src"] not in idx or e["dst"] not in idx:
            res["error"] = "unknown_node"
            return res
        u, v = idx[e["src"]], idx[e["dst"]]
        if u == v:
            res["error"] = "self_loop"
            return res
        if e["op"] == sem.ADD:
            if (u, v) not in allowed:
                res["error"] = "addition_not_allowed"
                return res
            if g2.has_edge(u, v):
                res["error"] = "edge_already_present"
                return res
            g2 = g2.with_edge(u, v)
        else:
            if not g2.has_edge(u, v):
                res["error"] = "edge_not_present"
                return res
            g2 = g2.without_edge(u, v)
    res["legal"] = True
    res["valid"] = sem.satisfied(g2, _pairs(idx, query["must_reach"]), _pairs(idx, query["must_not_reach"]))
    res["minimal"] = bool(res["valid"] and sol.min_edits is not None and len(edits) == sol.min_edits)
    res["correct"] = bool(res["minimal"])
    if not res["valid"]:
        res["error"] = "constraints_not_satisfied"
    elif not res["minimal"]:
        res["error"] = "not_minimal"
    return res


def _score_extraction(nodes, edges, value) -> Dict[str, Any]:
    true_nodes, pred_nodes = set(nodes), set(value["nodes"])
    true_edges = {(a, b) for a, b in edges}
    pred_edges = {(a, b) for a, b in value["edges"]}
    und_true = {frozenset(e) for e in true_edges}
    matched_pairs = [e for e in pred_edges if frozenset(e) in und_true]

    def ratio(num: int, den: int) -> Optional[float]:
        return (num / den) if den else None

    return {
        "correct": pred_nodes == true_nodes and pred_edges == true_edges,
        "node_precision": ratio(len(pred_nodes & true_nodes), len(pred_nodes)),
        "node_recall": ratio(len(pred_nodes & true_nodes), len(true_nodes)),
        "edge_precision": ratio(len(pred_edges & true_edges), len(pred_edges)),
        "edge_recall": ratio(len(pred_edges & true_edges), len(true_edges)),
        "direction_accuracy": ratio(sum(1 for e in matched_pairs if e in true_edges), len(matched_pairs)),
        "error": None,
    }


def score_response(
    query: Dict[str, Any], nodes: Sequence[str], edges: Sequence[Sequence[str]], text: Optional[str]
) -> Dict[str, Any]:
    """Score a raw model response against the true graph. A format failure is simply wrong."""
    task = query["type"]
    parsed = parse_response(task, text)
    if not parsed.ok:
        return {"correct": False, "format_ok": False, "error": "format:" + str(parsed.error), "parsed": None}
    if task == REPAIR:
        res = _score_repair(query, nodes, edges, parsed.value)
    elif task == EXTRACTION:
        res = _score_extraction(nodes, edges, parsed.value)
    else:
        truth = solve_query(query, nodes, edges)
        res = {"correct": parsed.value == truth, "error": None}
    res["format_ok"] = True
    res["parsed"] = parsed.value
    return res
