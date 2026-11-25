import importlib.util, json, re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_figure", ROOT / "scripts" / "paper_figure.py")
pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)

from trace_vr import reference as ref  # noqa: E402

DATA = ROOT / "data" / "pilot" / "dev.samples.jsonl"


@pytest.mark.skipif(not DATA.exists(), reason="pilot data not built")
def test_figure_claims_hold_under_reference_solver():
    samples = [json.loads(l) for l in DATA.read_text().splitlines() if l]
    info = pf.describe(pf.pick_group(samples))
    b = next(s for s in samples if s["image_path"] == info["root_image"] and s["task"] == "critical_edge")
    t = next(s for s in samples if s["image_path"] == info["twin_image"])
    nodes = b["private_truth"]["nodes"]
    q = re.search(r"Node (\w+) can currently reach node (\w+).*edge (\w+)->(\w+)", info["question"])
    s_, t_, u, v = q.groups()
    root_e = [tuple(e) for e in b["private_truth"]["edges"]]
    twin_e = [tuple(e) for e in t["private_truth"]["edges"]]
    # twin differs by exactly one edge and flips the answer
    assert len(set(root_e) ^ set(twin_e)) == 1
    assert ref.critical(nodes, root_e, s_, t_, (u, v)) == info["root_answer"]
    assert ref.critical(nodes, twin_e, s_, t_, (u, v)) == info["twin_answer"]
    assert info["root_answer"] != info["twin_answer"]
    # the shown repair satisfies every goal on the root graph
    edges = set(root_e)
    for e in info["repair_minimal_solutions"][0]:
        (edges.discard if e["op"] == "remove_edge" else edges.add)((e["src"], e["dst"]))
    for g in info["repair_goals"]:
        m = re.match(r"(\w+) must (NOT )?be able to reach (\w+)", g)
        assert ref.reachable(nodes, sorted(edges), m.group(1), m.group(3)) == (m.group(2) is None)


def test_figure_refuses_locked_test_split():
    with pytest.raises(SystemExit):
        pf.main(["manifests/pilot_test.json", "test_id", "/nonexistent/fig"])
