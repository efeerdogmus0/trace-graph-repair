import pytest
from trace_vr import seeds

M = seeds.MACRO


def row(label, points, ref_gap=None):
    r = {"label": label, "seeds": sorted(points),
         "per_seed": {str(s): {M: v} for s, v in points.items()},
         "pooled": {M: {"point": sum(points.values()) / len(points), "lo": 0, "hi": 1}}}
    if ref_gap is not None:
        r["vs_ref"] = {M: {"point": ref_gap, "lo": ref_gap - 0.01, "hi": ref_gap + 0.01}}
    return r


def table(b, c, split="dev", ref="B"):
    return {"split": split, "reference": ref, "content_digest": "x",
            "arms": [row("B", b), row("C", c, ref_gap=0.0)]}


ALL = (3407, 2027, 9173)


def test_picks_higher_pooled():
    t = table({s: 0.8 for s in ALL}, {s: 0.5 for s in ALL})
    assert seeds.choose_reference(t)["chosen"] == "B"
    t = table({s: 0.5 for s in ALL}, {s: 0.6 for s in ALL})
    assert seeds.choose_reference(t)["chosen"] == "C"


def test_tie_keeps_b():
    t = table({s: 0.7 for s in ALL}, {s: 0.7 for s in ALL})
    assert seeds.choose_reference(t)["chosen"] == "B"


def test_needs_all_seeds_and_dev():
    with pytest.raises(ValueError):
        seeds.choose_reference(table({s: 0.8 for s in ALL}, {3407: 0.9, 2027: 0.9}))
    with pytest.raises(ValueError):
        seeds.choose_reference(table({s: 0.8 for s in ALL}, {s: 0.5 for s in ALL}, split="test_id"))
    with pytest.raises(ValueError):
        seeds.choose_reference({"split": "dev", "arms": [row("B", {s: 0.8 for s in ALL})]})
