"""Invariance pairs: relabelled/restyled copies are scored in a separate run and paired with
root samples from the main run (protocol section 5)."""
import pytest

from trace_vr import arms, analysis, metrics, tasks
from test_dataset import built  # noqa: F401  (module-scoped fixture)


def _rows(built, n=4):
    split = built["samples"]["test_id"]
    return arms.eval_samples(split, n), arms.invariance_samples(split, n)


def test_main_set_unchanged_and_invariance_set_disjoint(built):
    main, inv = _rows(built)
    assert arms.select_samples(built["samples"]["test_id"], 4) == main
    assert arms.select_samples(built["samples"]["test_id"], 4, "invariance") == inv
    assert inv, "fixture build has no invariance copies"
    assert not {s["sample_id"] for s in main} & {s["sample_id"] for s in inv}
    assert {s["metadata"]["image_variant"] for s in inv} <= {"relabel", "restyle"}
    assert {s["task"] for s in inv} <= {tasks.REACHABILITY, tasks.CRITICAL_EDGE}
    assert {s["graph_group_id"] for s in inv} <= {s["graph_group_id"] for s in main}
    with pytest.raises(ValueError):
        arms.select_samples(built["samples"]["test_id"], 4, "all")


def test_every_copy_has_its_partner_in_the_main_run(built):
    main, inv = _rows(built)
    base_pairs = {p["pair_id"] for s in main for p in s["pairs"] if p["role"] == "base"}
    for s in inv:
        ids = [p["pair_id"] for p in s["pairs"] if p["kind"].startswith("invariant_")]
        assert ids and all(i in base_pairs for i in ids)
        partner = next(b for b in main if any(p["pair_id"] in ids and p["role"] == "base" for p in b["pairs"]))
        assert partner["metadata"]["image_variant"] == "root"
        assert partner["training_target"] == s["training_target"]  # same answer by construction


def test_invariance_table_counts(built):
    main, inv = _rows(built)
    all_true = analysis.invariance_table(main, {s["sample_id"]: True for s in main}, inv, {s["sample_id"]: True for s in inv})
    n = sum(c["pairs"] for t in all_true.values() for c in t.values())
    assert n == len(inv)
    for t in all_true.values():
        for c in t.values():
            assert c["both_right_rate"] == c["same_correctness_rate"] == 1.0
    copies_wrong = analysis.invariance_table(main, {s["sample_id"]: True for s in main}, inv, {s["sample_id"]: False for s in inv})
    for t in copies_wrong.values():
        for c in t.values():
            assert c["both_right"] == c["copy_right"] == c["same_correctness"] == 0 and c["root_right_rate"] == 1.0
    # a constant answerer is perfectly consistent across appearance changes, right or wrong
    resp = {s["sample_id"]: '{"answer": true}' for s in main + inv}
    m_flags = {s["sample_id"]: metrics.score_sample(s, resp[s["sample_id"]])["correct"] for s in main}
    i_flags = {s["sample_id"]: metrics.score_sample(s, resp[s["sample_id"]])["correct"] for s in inv}
    tab = analysis.invariance_table(main, m_flags, inv, i_flags)
    assert all(c["same_correctness_rate"] == 1.0 for t in tab.values() for c in t.values())


def test_missing_partner_is_an_error(built):
    main, inv = _rows(built)
    with pytest.raises(ValueError):
        analysis.invariance_table([], {}, inv, {s["sample_id"]: True for s in inv})


def test_run_suffix_keeps_existing_file_names():
    assert arms.run_suffix("dev") == "dev"
    assert arms.run_suffix("dev", "edges") == "dev_edges"
    assert arms.run_suffix("dev", "image+edges") == "dev_image_edges"
    assert arms.run_suffix("test_id", "image", "invariance") == "test_id_inv"


def test_final_eval_runs_the_invariance_set():
    from pathlib import Path
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "colab" / "final_eval.sh").read_text()
    assert 'image invariance' in sh and 'metrics_${SPLIT}_inv.json' in sh
