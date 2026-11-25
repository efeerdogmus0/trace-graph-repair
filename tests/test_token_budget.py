"""Token-limit selection from one long dev run (protocol items 1 and 5)."""
import json
from pathlib import Path

import pytest

from conftest import require_data
from trace_vr import arms, dataset, tasks, token_budget as tb

ROOT = Path(__file__).resolve().parents[1]
EOS = 0


def decode(ids):
    return "".join(chr(i) for i in ids if i != EOS)


def fake_row(sample, preamble, run_limit):
    text = "x" * preamble + json.dumps(sample["training_target"], sort_keys=True)
    ids = [ord(c) for c in text] + [EOS]
    hit = len(ids) >= run_limit
    ids = ids[:run_limit]
    return {"sample_id": sample["sample_id"], "task": sample["task"], "text": decode(ids),
            "new_tokens": len(ids), "hit_limit": hit, "ids": ids}


@pytest.fixture(scope="module")
def samples():
    require_data("smoke")
    _, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / "smoke.json")
    return arms.eval_samples(by_split["dev"], 6)


def test_long_repair_preamble_sets_the_limit(samples):
    rows = [fake_row(s, 300 if s["task"] == tasks.REPAIR else 0, 768) for s in samples]
    res = tb.scan(samples, rows, 768, decode)
    assert set(res["limits"]) == {"96", "192", "384", "768"}
    assert res["limits"]["96"]["truncation_rate"][tasks.REPAIR] == 1.0
    assert res["limits"]["96"]["truncation_rate"][tasks.REACHABILITY] == 0.0
    assert res["chosen_limit"] == 384
    # non-edit tasks answer with clean JSON: strict macro is perfect at every limit
    assert all(v["non_edit_macro"] == 1.0 for v in res["limits"].values())
    # repair has a prose preamble: strict fails, the lenient diagnostic recovers it once finished
    assert res["limits"]["768"]["per_task_accuracy"][tasks.REPAIR] == 0.0


def test_prefix_matches_a_real_shorter_run(samples):
    long_rows = [fake_row(s, 150, 768) for s in samples]
    short_rows = [fake_row(s, 150, 96) for s in samples]
    assert tb.check_prefix_consistency(short_rows, long_rows, 96) == {"same": len(samples), "mismatch": 0}
    res = tb.scan(samples, long_rows, 768, decode)
    for r in short_rows:  # the scan's 96-token response equals what a 96-token run produced
        long_r = next(x for x in long_rows if x["sample_id"] == r["sample_id"])
        assert tb.prefix_text(long_r, 96, decode) == r["text"]
    assert res["limits"]["96"]["max_truncation_rate"] == 1.0


def test_nothing_qualifies_uses_largest_and_bad_rows_raise(samples):
    rows = [fake_row(s, 2000, 768) for s in samples]
    assert tb.scan(samples, rows, 768, decode)["chosen_limit"] == 768
    bad = [dict(r) for r in rows]; del bad[0]["ids"]
    with pytest.raises(ValueError):
        tb.scan(samples, bad, 768, decode)
    with pytest.raises(ValueError):
        tb.scan(samples, rows[1:], 768, decode)
