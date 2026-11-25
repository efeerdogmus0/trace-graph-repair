import pytest

from trace_vr import arms, bootstrap, metrics, tasks
from test_dataset import built  # noqa: F401  (module-scoped fixture)


def _rows(built):
    return arms.eval_samples(built["samples"]["test_id"], 2)


def _truth(sample):
    return tasks.format_target(sample["training_target"])


def test_flags_round_trip_and_agree_with_metrics(built):
    rows = _rows(built)
    responses = {s["sample_id"]: (_truth(s) if i % 3 else "not json") for i, s in enumerate(rows)}
    payload = bootstrap.score_flags(rows, responses)
    assert payload["n"] == len(rows) and payload["missing_responses"] == 0
    flags = bootstrap.decode_flags(rows, payload)
    assert [flags[s["sample_id"]] for s in rows] == [i % 3 != 0 for i in range(len(rows))]
    assert bootstrap.decode_flags(rows, payload, "format_ok") == flags
    # point estimates of the bootstrap are the numbers metrics.evaluate reports
    ref = metrics.evaluate(rows, responses)
    res = bootstrap.cluster_bootstrap(rows, flags, n_boot=50)
    for task, entry in ref["per_task"].items():
        assert res["a"][task]["point"] == pytest.approx(entry["accuracy"])
    assert res["a"][bootstrap.MACRO]["point"] == pytest.approx(ref["non_edit_macro_accuracy"])


def test_flags_for_another_sample_order_are_refused(built):
    rows = _rows(built)
    payload = bootstrap.score_flags(rows, {})
    assert payload["missing_responses"] == len(rows) and set(payload["correct"]) == {"0"}
    with pytest.raises(ValueError):
        bootstrap.decode_flags(list(reversed(rows)), payload)
    with pytest.raises(ValueError):
        bootstrap.decode_flags(rows, dict(payload, correct=payload["correct"][:-1]))


def test_intervals_contain_the_point_are_deterministic_and_collapse_without_variation(built):
    rows = _rows(built)
    right = {s["sample_id"]: True for s in rows}
    res = bootstrap.cluster_bootstrap(rows, right, n_boot=100)
    for e in res["a"].values():
        assert e == {"point": 1.0, "lo": 1.0, "hi": 1.0}
    first = rows[0]["graph_group_id"]
    mixed = {s["sample_id"]: s["graph_group_id"] == first for s in rows}  # one root all right, the other all wrong
    r1 = bootstrap.cluster_bootstrap(rows, mixed, n_boot=300)
    r2 = bootstrap.cluster_bootstrap(rows, mixed, n_boot=300)
    assert r1 == r2
    e = r1["a"][bootstrap.MACRO]
    assert e["point"] == pytest.approx(0.5) and e["lo"] == 0.0 and e["hi"] == 1.0  # whole roots move together


def test_paired_difference_uses_the_same_roots_for_both_arms(built):
    rows = _rows(built)
    first = rows[0]["graph_group_id"]
    a = {s["sample_id"]: True for s in rows}
    b = {s["sample_id"]: s["graph_group_id"] != first for s in rows}
    res = bootstrap.cluster_bootstrap(rows, a, b, n_boot=300)
    d = res["a_minus_b"][bootstrap.MACRO]
    assert d["point"] == pytest.approx(0.5) and 0.0 <= d["lo"] <= d["hi"] <= 1.0
    # identical arms: the difference is exactly zero in every replicate, whatever roots are drawn
    same = bootstrap.cluster_bootstrap(rows, b, b, n_boot=300)["a_minus_b"][bootstrap.MACRO]
    assert same == {"point": 0.0, "lo": 0.0, "hi": 0.0, "share_above_zero": 0.0}


def test_analysis_recovers_answers_and_counts_pairs(built):
    from trace_vr import analysis
    rows = _rows(built)
    # a model that says "true" to every yes/no question and gets every repair right
    says_true = {s["sample_id"]: (s["task"] == tasks.REPAIR or bool(s["training_target"].get("answer"))) for s in rows}
    bias = analysis.answer_bias(rows, says_true)
    for task in tasks.MAIN_NON_EDIT_TASKS:
        assert bias[task]["said_true_rate"] == 1.0 and bias[task]["truth_true_rate"] == 0.5
    pairs = analysis.pair_table(rows, says_true)
    for task in tasks.MAIN_NON_EDIT_TASKS:  # twins have opposite answers, so a constant answer gets exactly one right
        assert pairs[task]["one_right"] == pairs[task]["pairs"] == 2
    assert pairs[tasks.REPAIR]["both_right"] == 2
    # format failures have no recoverable answer
    none_ok = {s["sample_id"]: False for s in rows}
    b2 = analysis.answer_bias(rows, none_ok, none_ok)
    assert all(e["said_true_rate"] is None and e["no_answer"] == e["n"] for e in b2.values())
    bd = analysis.breakdown(rows, says_true)
    for task, facets in bd.items():
        for values in facets.values():
            assert sum(c["n"] for c in values.values()) == sum(1 for s in rows if s["task"] == task)
    assert bd[tasks.REACHABILITY]["truth"]["True"]["accuracy"] == 1.0
    assert bd[tasks.REACHABILITY]["truth"]["False"]["accuracy"] == 0.0


def test_subset_flags_keeps_each_sample_flag(built):
    big = arms.eval_samples(built["samples"]["test_id"], 2)
    small = arms.eval_samples(built["samples"]["test_id"], 1)
    assert 0 < len(small) < len(big)
    responses = {s["sample_id"]: (_truth(s) if i % 2 else "x") for i, s in enumerate(big)}
    payload = bootstrap.score_flags(big, responses)
    sub = bootstrap.subset_flags(big, payload, small)
    full = bootstrap.decode_flags(big, payload)
    part = bootstrap.decode_flags(small, sub)
    assert part == {s["sample_id"]: full[s["sample_id"]] for s in small}
    assert sub == dict(sub, **{"n": len(small), "subset_of": payload["order_sha"]})
    with pytest.raises(ValueError):
        bootstrap.subset_flags(small, payload, small)
