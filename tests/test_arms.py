from trace_vr import arms, tasks
from test_dataset import built  # noqa: F401  (module-scoped fixture)


def test_arms_see_the_same_roots_and_the_same_number_of_samples_and_images(built):
    rows = built["samples"]["train"]
    d, b = arms.training_samples(rows, "D"), arms.training_samples(rows, "B")
    assert len(d) == len(b) == 2 * 8
    assert {s["graph_group_id"] for s in d} == {s["graph_group_id"] for s in b}
    assert len({s["image_id"] for s in d}) == len({s["image_id"] for s in b}) == 16
    assert {s["task"] for s in d} == {tasks.REPAIR}
    assert {s["task"] for s in b} == {tasks.REACHABILITY, tasks.CRITICAL_EDGE}
    assert not any(s["task"] in tasks.MAIN_NON_EDIT_TASKS for s in d)  # D never sees a non-edit answer
    ev = arms.eval_samples(built["samples"]["test_id"], 2)
    assert len({s["graph_group_id"] for s in ev}) == 2 and len(ev) == 12


def test_oracle_input_modes():
    import json as _json
    from pathlib import Path as _P
    import pytest
    from trace_vr import dataset as _ds
    from conftest import require_data
    require_data("pilot")
    rows = [_json.loads(l) for l in _P("data/pilot/dev.samples.jsonl").read_text().splitlines()[:30]]
    for s in rows:
        plain = _ds.to_model_example(s, _P("data/pilot"))
        both = _ds.to_model_example(s, _P("data/pilot"), input_mode="image+edges")
        text = _ds.to_model_example(s, _P("data/pilot"), input_mode="edges")
        assert plain["prompt"] == s["public_input"]["instruction"]
        assert both["image"] == plain["image"] and text["image"] is None
        assert both["prompt"].endswith(plain["prompt"]) and text["prompt"].endswith(plain["prompt"])
        line = both["prompt"].split("\n", 1)[0]
        for a, b in s["private_truth"]["edges"]:
            assert "%s->%s" % (a, b) in line
        assert line.count("->") == len(s["private_truth"]["edges"])
    with pytest.raises(ValueError):
        _ds.to_model_example(rows[0], _P("data/pilot"), with_target=True, input_mode="edges")
    with pytest.raises(KeyError):
        _ds.to_model_example(rows[0], _P("data/pilot"), input_mode="text")


def test_arm_c_desc_matches_d_images_and_labels():
    from pathlib import Path
    from trace_vr import arms, dataset, tasks
    from conftest import require_data
    require_data("smoke")
    man, dd, bs, _ = dataset.load_dataset(Path("manifests/smoke.json"))
    rows = bs["train"]
    d = arms.training_samples(rows, "D")
    c = arms.training_samples(rows, "C_desc")
    assert len(c) == len(d)
    assert [s["image_path"] for s in c] == [s["image_path"] for s in d]
    for cs, ds in zip(c, d):
        assert cs["task"] == tasks.EXTRACTION
        assert sorted(map(tuple, cs["training_target"]["edges"])) == sorted(map(tuple, ds["private_truth"]["edges"]))
        ex = dataset.to_model_example(cs, dd, with_target=True)
        assert set(ex) == {"image", "prompt", "target"}
        res = tasks.score_response(cs["public_input"]["query"], ds["private_truth"]["nodes"],
                                   ds["private_truth"]["edges"], ex["target"])
        assert res["correct"] and res["format_ok"]


def test_arm_d2_edit_outcome_same_images_balanced_and_exact():
    from pathlib import Path
    from trace_vr import arms, dataset, semantics as sem, tasks
    from conftest import require_data
    require_data("smoke")
    man, dd, bs, _ = dataset.load_dataset(Path("manifests/smoke.json"))
    rows = bs["train"]
    d = arms.training_samples(rows, "D")
    d2 = arms.training_samples(rows, "D2")
    assert len(d2) == len(d)
    assert [s["image_path"] for s in d2] == [s["image_path"] for s in d]
    out = [s for s in d2 if s["task"] == tasks.EDIT_OUTCOME]
    assert len(out) == len(d) // 2  # one per root group
    assert not any(s["task"] in tasks.MAIN_NON_EDIT_TASKS for s in d2)
    assert d2 == arms.training_samples(rows, "D2")  # deterministic
    for s, src in zip(d2, d):
        if s["task"] != tasks.EDIT_OUTCOME:
            assert s is src
            continue
        q = s["public_input"]["query"]
        t = src["private_truth"]
        g, idx = tasks.build_graph(t["nodes"], t["edges"])
        e = q["edit"]
        u, v = idx[e["src"]], idx[e["dst"]]
        if e["op"] == sem.REMOVE:
            assert g.has_edge(u, v)
        else:
            assert not g.has_edge(u, v) and [e["src"], e["dst"]] in [list(p) for p in src["public_input"]["query"]["allowed_additions"]]
        # independent check of the label
        g2 = g.without_edge(u, v) if e["op"] == sem.REMOVE else g.with_edge(u, v)
        ok = all(g2.reaches(idx[a], idx[b]) for a, b in q["must_reach"]) and \
            not any(g2.reaches(idx[a], idx[b]) for a, b in q["must_not_reach"])
        assert s["training_target"] == {"answer": ok}
        ex = dataset.to_model_example(s, dd, with_target=True)
        assert e["src"] + "->" + e["dst"] in ex["prompt"]
        assert tasks.score_response(q, t["nodes"], t["edges"], ex["target"])["correct"]
        assert not tasks.score_response(q, t["nodes"], t["edges"], '{"answer": %s}' % str(not ok).lower())["correct"]


def test_arm_c_three_structural_tasks_on_d_images():
    """Protocol item 4: C cycles description / infilling / matching over D's image slots."""
    from pathlib import Path
    from trace_vr import arms, dataset, tasks
    from conftest import require_data
    require_data("smoke")
    man, dd, bs, _ = dataset.load_dataset(Path("manifests/smoke.json"))
    rows = bs["train"]
    d = arms.training_samples(rows, "D")
    c = arms.training_samples(rows, "C")
    assert c == arms.training_samples(rows, "C")  # deterministic
    assert len(c) == len(d)
    assert [s["image_path"] for s in c] == [s["image_path"] for s in d]
    kinds = [s["task"] for s in c]
    assert kinds == [arms.C_CYCLE[k % 3] for k in range(len(c))]
    assert not any(k in tasks.MAIN_NON_EDIT_TASKS or k == tasks.REPAIR for k in kinds)
    match_answers = []
    for cs, ds in zip(c, d):
        truth = ds["private_truth"]
        q = cs["public_input"]["query"]
        assert cs["public_input"]["instruction"] == tasks.render_instruction(q)
        assert cs["training_target"] == tasks.solve_query(q, truth["nodes"], truth["edges"])
        ex = dataset.to_model_example(cs, dd, with_target=True)
        res = tasks.score_response(q, truth["nodes"], truth["edges"], ex["target"])
        assert res["correct"] and res["format_ok"], cs["sample_id"]
        true_edges = sorted(map(tuple, truth["edges"]))
        if cs["task"] == tasks.INFILL:
            shown = sorted(map(tuple, q["shown_edges"]))
            hidden = sorted(map(tuple, cs["training_target"]["edges"]))
            assert q["n_missing"] == len(hidden) == max(1, len(true_edges) // 3)
            assert sorted(shown + hidden) == true_edges
            for a, b in q["shown_edges"]:
                assert "%s->%s" % (a, b) in ex["prompt"]
        if cs["task"] == tasks.MATCH:
            ans = cs["training_target"]["answer"]
            match_answers.append(ans)
            listed = sorted(map(tuple, q["edges"]))
            if ans:
                assert listed == true_edges
            else:  # hard negative: the other image of the same root group, one edit away
                twin = [s for s in d if s["graph_group_id"] == ds["graph_group_id"] and s is not ds]
                assert len(twin) == 1
                assert listed == sorted(map(tuple, twin[0]["private_truth"]["edges"]))
                assert len(set(listed) ^ set(true_edges)) == 1
    assert match_answers and abs(sum(match_answers) - len(match_answers) / 2.0) <= 1


def test_infill_and_match_parsing_is_strict():
    from trace_vr import tasks
    q = {"type": tasks.INFILL, "shown_edges": [["A", "B"]], "n_missing": 2}
    nodes, edges = ["A", "B", "C"], [["A", "B"], ["B", "C"], ["C", "A"]]
    assert tasks.score_response(q, nodes, edges, '{"edges":[["C","A"],["B","C"]]}')["correct"]
    assert not tasks.score_response(q, nodes, edges, '{"edges":[["B","C"]]}')["correct"]
    assert not tasks.score_response(q, nodes, edges, '{"edges":[["B","C"],["C","A"],["C","A"]]}')["correct"]
    assert not tasks.score_response(q, nodes, edges, '{"answer":[]}')["format_ok"]
    m = {"type": tasks.MATCH, "edges": [["B", "C"], ["A", "B"], ["C", "A"]]}
    assert tasks.solve_query(m, nodes, edges) == {"answer": True}
    m2 = {"type": tasks.MATCH, "edges": [["A", "B"], ["C", "B"], ["C", "A"]]}
    assert tasks.solve_query(m2, nodes, edges) == {"answer": False}
    assert "Exactly 2 edges" in tasks.render_instruction(q)
