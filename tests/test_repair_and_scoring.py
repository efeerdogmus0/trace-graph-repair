"""Risks 2 and 3: minimal-edit labels and the semantic parser/scorer."""
import json

from trace_vr import families, reference as ref, semantics as sem, tasks
from trace_vr.rng import Rng

NODES = ["A", "B", "C", "D"]


def q(must_reach, must_not, additions):
    return {"type": tasks.REPAIR, "must_reach": must_reach, "must_not_reach": must_not,
            "allowed_additions": additions, "max_edits": 1}


def canon(solutions):
    """Order-independent form of a list of solutions (each a list of edit dicts)."""
    return sorted(json.dumps(sol, sort_keys=True) for sol in solutions)


def solve(query, edges):
    _, _, sol = tasks.solve_repair_labels(query, NODES, edges)
    mine = (sol.status, canon(tasks.edits_to_labels(s, NODES) for s in sol.solutions), sol.min_edits)
    status, sols, k = ref.repair(NODES, [tuple(e) for e in edges], [tuple(p) for p in query["must_reach"]],
                                 [tuple(p) for p in query["must_not_reach"]], [tuple(p) for p in query["allowed_additions"]])
    theirs = (status, canon([{"op": o, "src": a, "dst": b} for o, a, b in s] for s in sols), k)
    assert mine == theirs
    return mine


def add(a, b):
    return {"op": "add_edge", "src": a, "dst": b}


def rem(a, b):
    return {"op": "remove_edge", "src": a, "dst": b}


# Hand-checked cases -------------------------------------------------------------
TWO_FIXES = (q([["A", "C"]], [["B", "D"]], [["A", "C"], ["B", "C"], ["B", "D"]]), [["A", "B"], ["D", "C"]])


def test_goal_already_holds_means_zero_edits():
    status, sols, k = solve(q([["A", "C"]], [["C", "A"]], [["C", "D"], ["D", "A"]]), [["A", "B"], ["B", "C"]])
    assert (status, sols, k) == ("unchanged", canon([[]]), 0)


def test_several_minimal_additions_and_one_forbidden_candidate():
    # A->C and B->C both work. B->D would also connect A to C (via D) but lets B reach D, which is forbidden.
    status, sols, k = solve(*TWO_FIXES)
    assert status == "edited" and k == 1
    assert sols == canon([[add("A", "C")], [add("B", "C")]])


def test_single_removal_that_keeps_the_required_path():
    status, sols, k = solve(q([["A", "B"]], [["A", "C"]], [["C", "D"], ["D", "B"]]), [["A", "B"], ["B", "C"], ["A", "D"]])
    assert (status, sols, k) == ("edited", canon([[rem("B", "C")]]), 1)


def test_two_violations_cannot_be_fixed_by_one_edit():
    status, sols, k = solve(q([["A", "D"]], [["A", "B"]], [["A", "C"], ["B", "C"]]), [["A", "B"], ["C", "D"]])
    assert (status, sols, k) == ("impossible", [], None)


def test_impossible_when_no_listed_addition_helps():
    status, sols, k = solve(q([["A", "C"]], [["C", "A"]], [["C", "D"], ["D", "B"]]), [["A", "B"]])
    assert (status, sols, k) == ("impossible", [], None)


# Scorer -------------------------------------------------------------------------
def score(response, case=TWO_FIXES):
    query, edges = case
    return tasks.score_response(query, NODES, edges, response)


def test_every_minimal_solution_is_accepted_regardless_of_key_order():
    assert score('{"status":"edited","edits":[{"op":"add_edge","src":"A","dst":"C"}]}')["correct"]
    assert score('{"edits":[{"dst":"C","src":"B","op":"add_edge"}],"status":"edited"}')["correct"]
    assert score('```json\n{"status":"edited","edits":[{"op":"add_edge","src":"B","dst":"C"}]}\n```')["correct"]


def test_wrong_repairs_are_classified():
    def err(edits, status="edited"):
        return score(json.dumps({"status": status, "edits": edits}))

    forbidden = err([add("B", "D")])
    assert not forbidden["correct"] and forbidden["legal"] and not forbidden["valid"]
    assert err([add("C", "A")])["error"] == "addition_not_allowed"
    assert err([add("A", "Z")])["error"] == "unknown_node"
    assert err([rem("B", "A")])["error"] == "edge_not_present"  # reversed direction
    assert err([add("A", "B")])["error"] == "addition_not_allowed"  # existing edge, and not listed
    useless = err([rem("A", "B")])
    assert useless["legal"] and not useless["valid"] and useless["error"] == "constraints_not_satisfied"
    assert err([add("A", "C"), add("B", "C")])["error"] == "edit_count_out_of_budget"
    assert err([])["error"] == "edit_count_out_of_budget"
    assert err([add("A", "C")], status="unchanged")["error"] == "edits_given_with_non_edited_status"
    for status in ("unchanged", "impossible"):
        res = err([], status=status)
        assert not res["correct"] and not res["status_correct"]


def test_valid_but_unnecessary_edit_is_not_minimal():
    case = (q([["A", "C"]], [["C", "A"]], [["C", "D"], ["D", "A"]]), [["A", "B"], ["B", "C"]])
    res = score('{"status":"edited","edits":[{"op":"add_edge","src":"C","dst":"D"}]}', case)
    assert res["legal"] and res["valid"] and not res["minimal"] and not res["correct"] and res["error"] == "not_minimal"
    assert score('{"status":"unchanged","edits":[]}', case)["correct"]


def test_format_failures_are_wrong_and_never_repaired():
    for bad in [None, "", "   ", "not json", 'The answer is {"status":"unchanged","edits":[]}', "[1]",
                '{"status":"maybe","edits":[]}', '{"status":"edited"}', '{"status":"edited","edits":[{"op":"flip","src":"A","dst":"C"}]}',
                '{"status":"edited","edits":[{"op":"add_edge","src":"A"}]}', '{"status":"edited","edits":"A->C"}',
                '{"status":"edited","edits":[{"op":"add_edge","src":"A","dst":"C"}],"note":"x"}']:
        res = score(bad)
        assert not res["correct"] and not res["format_ok"], bad


def test_boolean_and_integer_schemas_are_strict():
    assert tasks.parse_response(tasks.REACHABILITY, ' {"answer": true}\n').value == {"answer": True}
    for bad in ['{"answer": "true"}', '{"answer": 1}', '{"answer": null}', '{"answer": true, "why": "x"}', "true", '{"Answer": true}']:
        assert not tasks.parse_response(tasks.REACHABILITY, bad).ok, bad
    assert tasks.parse_response(tasks.SHORTEST_PATH, '{"answer": 3}').value == {"answer": 3}
    assert tasks.parse_response(tasks.SHORTEST_PATH, '{"answer": null}').value == {"answer": None}
    for bad in ['{"answer": true}', '{"answer": -1}', '{"answer": 2.0}', '{"answer": "3"}']:
        assert not tasks.parse_response(tasks.SHORTEST_PATH, bad).ok, bad


def test_non_edit_answers_and_extraction_scores():
    edges = [["A", "B"], ["B", "C"]]
    nodes = ["A", "B", "C"]
    reach = {"type": tasks.REACHABILITY, "src": "A", "dst": "C"}
    assert tasks.score_response(reach, nodes, edges, '{"answer": true}')["correct"]
    assert not tasks.score_response(reach, nodes, edges, '{"answer": false}')["correct"]
    crit = {"type": tasks.CRITICAL_EDGE, "src": "A", "dst": "C", "edge": ["A", "B"]}
    assert tasks.score_response(crit, nodes, edges, '{"answer": true}')["correct"]
    assert tasks.score_response(crit, nodes, edges + [["A", "C"]], '{"answer": false}')["correct"]
    dist = {"type": tasks.SHORTEST_PATH, "src": "A", "dst": "C"}
    assert tasks.score_response(dist, nodes, edges, '{"answer": 2}')["correct"]
    assert tasks.score_response({"type": tasks.SHORTEST_PATH, "src": "C", "dst": "A"}, nodes, edges, '{"answer": null}')["correct"]
    ext = tasks.score_response({"type": tasks.EXTRACTION}, nodes, edges, '{"nodes":["A","B","C"],"edges":[["A","B"],["C","B"]]}')
    assert not ext["correct"]
    assert ext["edge_precision"] == 0.5 and ext["edge_recall"] == 0.5 and ext["direction_accuracy"] == 0.5
    assert ext["node_precision"] == 1.0 and ext["node_recall"] == 1.0


# Property checks on generated tasks ----------------------------------------------
def test_generated_repairs_match_reference_and_scorer_matches_solution_set():
    checked = 0
    for i in range(60):
        rng = Rng("repair-prop", i)
        fam = rng.choice(["layered_dag", "tree_plus", "sparse_random", "ring_chords", "two_cluster"])
        g = families.generate(fam, rng.randint(5, 9), rng.child("g"))
        nodes = rng.child("labels").sample(tasks.LABEL_ALPHABET, g.n)
        edges = [[nodes[u], nodes[v]] for u, v in g.edges]
        for cat in sem.REPAIR_CATEGORIES:
            spec = None
            for k in range(30):
                spec = sem.propose_repair(g, rng.child(cat, k), cat)
                if spec:
                    break
            if not spec:
                continue
            query = q([[nodes[a], nodes[b]] for a, b in spec["must_reach"]], [[nodes[a], nodes[b]] for a, b in spec["must_not"]],
                      [[nodes[a], nodes[b]] for a, b in spec["additions"]])
            status, sols, _ = ref.repair(nodes, [tuple(e) for e in edges], [tuple(p) for p in query["must_reach"]],
                                         [tuple(p) for p in query["must_not_reach"]], [tuple(p) for p in query["allowed_additions"]])
            assert status == spec["solution"].status and sem.repair_category(spec["solution"]) == cat
            right = {json.dumps({"status": "edited", "edits": [{"op": o, "src": a, "dst": b}]}) for (o, a, b), in sols} if status == "edited" \
                else {json.dumps({"status": status, "edits": []})}
            responses = {json.dumps({"status": s, "edits": []}) for s in ("unchanged", "impossible")}
            responses |= {json.dumps({"status": "edited", "edits": [add(a, b)]}) for a, b in query["allowed_additions"]}
            responses |= {json.dumps({"status": "edited", "edits": [rem(a, b)]}) for a, b in edges}
            for resp in sorted(responses):
                assert tasks.score_response(query, nodes, edges, resp)["correct"] == (resp in right), (resp, right)
                checked += 1
    assert checked > 1500


def test_twins_keep_the_text_valid_and_change_every_correct_answer():
    pairs = 0
    for i in range(40):
        rng = Rng("twin-prop", i)
        g = families.generate("sparse_random", rng.randint(6, 9), rng.child("g"))
        for cat in sem.REPAIR_CATEGORIES:
            spec = sem.propose_repair(g, rng.child(cat), cat)
            if not spec:
                continue
            base = spec["solution"].answer_key()
            for mod, sol2 in sem.repair_twin_mods(g, spec):
                g2 = sem.apply_mod(g, mod)
                assert all(g2.can_add(u, v) for u, v in spec["additions"])  # the listed additions stay addable
                assert not (base & sol2.answer_key())
                pairs += 1
        for ans in (True, False):
            r = sem.propose_reachability(g, rng.child("reach", ans), ans)
            if r:
                assert g.reaches(r["src"], r["dst"]) is ans
                assert all(sem.apply_mod(g, m).reaches(r["src"], r["dst"]) is not ans for m in r["twin_mods"])
            c = sem.propose_critical_edge(g, rng.child("crit", ans), ans)
            if c:
                assert sem.critical_answer(g, c["src"], c["dst"], c["edge"]) is ans
                assert all(sem.critical_answer(sem.apply_mod(g, m), c["src"], c["dst"], c["edge"]) is not ans for m in c["twin_mods"])
    assert pairs > 200


def test_training_target_text_follows_the_instructed_key_order():
    stored = {"edits": [{"dst": "D", "op": "remove_edge", "src": "W"}], "status": "edited"}  # as read back from sorted JSON
    assert tasks.format_target(stored) == '{"status":"edited","edits":[{"op":"remove_edge","src":"W","dst":"D"}]}'
    assert tasks.format_target({"answer": True}) == '{"answer":true}'


def test_lenient_candidate_finds_the_last_json_object_only_as_a_diagnostic():
    text = 'Let me think. {"answer": false} is tempting, but paths exist.\n\nFinal: {"answer": true}'
    assert tasks.lenient_candidate(text) == '{"answer": true}'
    assert tasks.lenient_candidate('so the edit is {"status":"edited","edits":[{"op":"add_edge","src":"A","dst":"C"}]} done') == \
        '{"status":"edited","edits":[{"op":"add_edge","src":"A","dst":"C"}]}'
    assert tasks.lenient_candidate("no json here {broken") is None
    assert tasks.lenient_candidate(None) is None
    assert not tasks.parse_response(tasks.REACHABILITY, text).ok  # the strict score still fails this response
