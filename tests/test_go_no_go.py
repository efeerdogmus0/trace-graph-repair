import pytest

from trace_vr import go_no_go as g


def e(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi}


SMALL = e(0.01, -0.02, 0.04)


def test_large_machine_effect_forces_retrain_of_D():
    assert g.decide(e(0.05, 0.02, 0.08), e(-0.2, -0.25, -0.15), e(-0.1, -0.15, -0.05))["decision"] == "retrain_D_on_kaggle"
    # interval contains 0 but point beyond tolerance
    assert not g.machine_small(e(-0.035, -0.07, 0.001))


def test_supported_when_a_D_arm_is_above_B():
    assert g.decide(SMALL, e(0.05, 0.01, 0.09), e(-0.1, -0.2, -0.05))["decision"] == "H1_supported_on_dev"


def test_close_when_both_far_below():
    r = g.decide(SMALL, e(-0.175, -0.2, -0.149), e(-0.136, -0.163, -0.109))
    assert r["decision"] == "close_H1_not_supported"
    assert r["best_point_gap"] == pytest.approx(-0.136)


def test_revision_2_when_gap_small_or_uncertain():
    assert g.decide(SMALL, e(-0.05, -0.08, -0.02), e(-0.2, -0.25, -0.15))["decision"] == "revision_2"
    assert g.decide(SMALL, e(-0.2, -0.25, 0.01), e(-0.2, -0.25, -0.15))["decision"] == "revision_2"


def test_rejects_malformed_entries():
    with pytest.raises(ValueError):
        g.decide(SMALL, e(0.5, 0.0, 0.1), SMALL)
    with pytest.raises(ValueError):
        g.decide(SMALL, {"point": 0.0}, SMALL)


def test_thresholds_are_the_preregistered_ones():
    # changing these after job11 is read back would void the pre-registration (DECISIONS 2026-11-12)
    assert (g.MACHINE_TOL, g.GAP_CLOSE) == (0.03, 0.10)
