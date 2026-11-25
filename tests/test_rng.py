"""The generator must give the same numbers on every platform and Python version.

These values were recorded once (Python 3.9.6, macOS arm64). If this test fails on another
machine, datasets generated there are NOT the same datasets.
"""
from trace_vr.rng import Rng, stable_hash


def test_recorded_sequence_is_reproduced():
    r = Rng("trace", 1)
    assert [r.below(1000) for _ in range(6)] == [752, 715, 405, 700, 79, 299]
    assert r.shuffled(range(10)) == [6, 4, 8, 5, 3, 1, 7, 9, 0, 2]
    assert r.sample("ABCDEFGH", 3) == ['H', 'D', 'A']
    assert [r.random() for _ in range(2)] == [0.2030247331210414, 0.37818870845174535]
    assert Rng("trace", 1).child("x").randint(5, 10) == 6
    assert stable_hash("a", 1, "b") == '7ed6228c580e42c6'


def test_streams_are_independent_and_repeatable():
    a, b = Rng("s", 1), Rng("s", 1)
    assert [a.below(50) for _ in range(20)] == [b.below(50) for _ in range(20)]
    assert [Rng("s", 1).below(10**9) for _ in range(1)] != [Rng("s", 2).below(10**9) for _ in range(1)]
    parent = Rng("p")
    first = parent.child("c").below(10**9)
    parent.below(7)  # advancing the parent must not change what a child produces
    assert parent.child("c").below(10**9) == first


def test_below_is_in_range_and_covers_it():
    r = Rng("range")
    seen = {r.below(7) for _ in range(400)}
    assert seen == set(range(7))
    assert all(3 <= r.randint(3, 5) <= 5 for _ in range(100))
