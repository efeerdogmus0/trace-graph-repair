import json

from trace_vr import arms, metrics, parse_solve as ps, tasks
from test_dataset import built  # noqa: F401  (module-scoped fixture)


def _truth_text(s):
    t = s["private_truth"]
    return json.dumps({"nodes": t["nodes"], "edges": t["edges"]})


def test_perfect_extraction_scores_every_main_sample_right(built):
    rows = arms.eval_samples(built["samples"]["test_id"], 4)
    ext = {s["image_id"]: _truth_text(s) for s in rows}
    resp, reasons = ps.responses_for_samples(rows, ext)
    assert reasons == {"answered": len(rows)}
    res = metrics.evaluate(rows, resp)
    for t in (tasks.REPAIR, tasks.REACHABILITY, tasks.CRITICAL_EDGE):
        assert res["per_task"][t]["accuracy"] == 1.0


def test_failures_are_wrong_not_crashes(built):
    rows = arms.eval_samples(built["samples"]["test_id"], 4)
    # prose -> strict format failure, lenient finds the JSON
    s = rows[0]
    prose = "Here is the graph: " + _truth_text(s)
    r, info = ps.answer_from_extraction(s["public_input"]["query"], prose)
    assert r is None and info["reason"].startswith("extraction_format")
    r2, info2 = ps.answer_from_extraction(s["public_input"]["query"], prose, lenient=True)
    assert r2 is not None and info2["reason"] is None
    # empty graph: every query still gets an answer or a premise failure, never an exception
    ext = {x["image_id"]: '{"nodes":[],"edges":[]}' for x in rows}
    resp, reasons = ps.responses_for_samples(rows, ext)
    assert sum(reasons.values()) == len(rows)
    crit = [x for x in rows if x["task"] == tasks.CRITICAL_EDGE]
    assert all(resp[x["sample_id"]] is None for x in crit)  # src cannot reach dst on an empty graph
    metrics.evaluate(rows, resp)


def test_cleaning_rules():
    q = {"type": tasks.REACHABILITY, "src": "A", "dst": "Z"}
    nodes, edges, c = ps.clean_graph({"nodes": ["A", "B"], "edges": [["A", "B"], ["A", "B"], ["B", "B"], ["B", "C"]]}, q)
    assert nodes == ["A", "B", "C", "Z"] and edges == [["A", "B"], ["B", "C"]]
    assert c == {"self_loops_dropped": 1, "duplicate_edges_merged": 1, "nodes_added_from_edges_or_query": 2}
    assert ps.answer_from_extraction(q, '{"nodes":["A","B"],"edges":[["A","B"]]}')[0] == '{"answer":false}'
