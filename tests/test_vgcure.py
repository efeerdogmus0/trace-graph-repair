import json
from pathlib import Path

import pytest

from trace_vr import tasks, vgcure


def _u(img, nb):
    return {"id": img, "graph_image": "er_directed_akg_images/%d.png" % img, "task": {"neighbor_query": nb}}


def _r(img, q, a):
    return {"id": img, "graph_image": "er_directed_akg_images/%d.png" % img,
            "task": {"connective_path_query": {"qa": {"question": q, "answer": a}}}}


def _nb(anchor, listed, pos, neg, neg_kind="out"):
    T = "\nThe answer should contain only \"True\" or \"False\", without reasoning process."
    return {
        "qa": {"question": "Which nodes are out-neighbors of %s in this graph?\nPlease ..." % anchor,
               "answer": "[%s]" % ", ".join(listed)},
        "fc_sup": {"claim": "%s is a out-neighbors of %s in this graph.%s" % (pos, anchor, T), "label": "True"},
        "fc_ref": {"claim": "%s is a %s-neighbors of %s in this graph.%s" % (neg, neg_kind, anchor, T), "label": "False"},
    }


def test_edge_direction_and_order():
    us = [_u(10, _nb("E1", ["E2"], "E2", "E3")), _u(2, _nb("E4", ["E5", "E1"], "E5", "E1", "in"))]
    rs = [_r(10, "Is there a path from E1 to E3?\nIf yes ...", "No"),
          _r(2, "Is there a path from E4 to E2?\nIf yes ...", "Yes. The paths are[[E4, E5, E2]]")]
    s = vgcure.select_source("er", us, rs, per_source=2)
    assert [x["root_id"] for x in s[::3]] == ["vgcure/er/2", "vgcure/er/10"]  # numeric order, not file order
    pos, neg, reach = s[0:3]
    assert (pos["query"]["src"], pos["query"]["dst"], pos["answer"]) == ("E4", "E5", True)
    # "E1 is a in-neighbor of E4" = edge E1->E4, false although E4->E1 exists
    assert (neg["query"]["src"], neg["query"]["dst"], neg["answer"]) == ("E1", "E4", False)
    assert neg["negative_kind"] == "reversed_edge"
    assert reach["task"] == tasks.REACHABILITY and reach["answer"] is True
    assert s[5]["answer"] is False and s[4]["negative_kind"] == "same_direction"
    assert pos["instruction"] == tasks.render_instruction(pos["query"])


def test_inconsistent_claim_rejected():
    us = [_u(1, _nb("E1", ["E2", "E3"], "E2", "E3"))]  # negative is actually listed
    rs = [_r(1, "Is there a path from E1 to E3?", "Yes")]
    with pytest.raises(ValueError):
        vgcure.select_source("er", us, rs, per_source=1)


def test_too_few_graphs():
    us = [_u(1, _nb("E1", ["E2"], "E2", "E3"))]
    rs = [_r(1, "Is there a path from E1 to E3?", "No")]
    with pytest.raises(ValueError):
        vgcure.select_source("er", us, rs, per_source=2)


def test_committed_selection_record():
    rec = json.loads(Path("reports/external/vgcure_subset.json").read_text())
    assert rec["revision"] == vgcure.HF_REVISION and rec["per_source"] == vgcure.PER_SOURCE
    assert rec["n_samples"] == 3 * vgcure.PER_SOURCE * len(vgcure.SOURCES) == len(rec["sample_ids"])
    assert set(rec["counts"]) == set(vgcure.SOURCES)
    assert "samples" not in rec  # no VGCure question text in git (no licence stated)
    src = Path("external/vgcure_meta")
    if not (src / "understanding").exists():
        pytest.skip("VGCure test files not downloaded")
    sub = vgcure.build_subset(src)
    assert sub["subset_sha256"] == rec["subset_sha256"]
    assert sub["source_files_sha256"] == rec["source_files_sha256"]


def test_dev_subset_is_disjoint_from_the_preregistered_one():
    """Exploratory dev subset (2026-11-14): the next DEV_PER_SOURCE usable graphs, no graph shared."""
    dev = json.loads(Path(vgcure.DEV_RECORD).read_text())
    rec = json.loads(Path(vgcure.TEST_RECORD).read_text())
    graphs = lambda r: {i.rsplit("/", 1)[0] for i in r["sample_ids"]}
    assert dev["skip"] == vgcure.PER_SOURCE and dev["per_source"] == vgcure.DEV_PER_SOURCE
    assert len(graphs(dev)) == vgcure.DEV_PER_SOURCE * len(vgcure.SOURCES) and not graphs(dev) & graphs(rec)
    assert dev["source_files_sha256"] == rec["source_files_sha256"] and "samples" not in dev
    src = Path("external/vgcure_meta")
    if not (src / "understanding").exists():
        pytest.skip("VGCure test files not downloaded")
    sub = vgcure.build_subset(src, vgcure.DEV_PER_SOURCE, skip=vgcure.PER_SOURCE)
    assert sub["subset_sha256"] == dev["subset_sha256"]
