"""result_summary.py: announcement draft / Turkish summary from a final_primary.json. Synthetic files only."""
import importlib.util, json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
P = "primary_six_cell"


def load():
    spec = importlib.util.spec_from_file_location("result_summary", ROOT / "scripts/result_summary.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _e(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi}


def _primary(reading, task_drop=(), same=True):
    arms = [{"label": a, "seeds": [2027, 3407, 9173], "pooled": {P: _e(v, v - .02, v + .02)}}
            for a, v in (("B", .61), ("C", .52), ("D", .66), ("E", .58))]
    row = lambda a, b, v: {"a": a, "b": b, "diff": {P: _e(v, v - .03, v + .03)}, "reading": reading if (a, b) == ("D", "B") else "x",
                           "seed_matched": {"3407": v, "2027": v, "9173": v}, "task_drop": list(task_drop),
                           "splits_same_sign": same}
    return {"statistic": P, "reference": "B", "arms": arms,
            "differences": [row("C", "B", -.09), row("D", "B", .05), row("E", "B", -.03), row("E", "D", -.08)]}


@pytest.mark.parametrize("reading", ["positive_clear", "positive", "negative", "inconclusive", "incomplete_seeds"])
def test_each_reading_has_fixed_sentences(reading):
    m = load()
    t = _primary(reading)
    en, tr = m.render(t, "en"), m.render(t, "tr")
    assert m.EN[reading] in en and m.TR[reading] in tr
    assert "+0.050 [+0.020, +0.080]" in en and "+0.050 [+0.020, +0.080]" in tr
    assert "DRAFT" in en and "TASLAK" in tr
    assert m.SCALE_EN in en and m.SCALE_TR in tr
    assert "Reported E - D: -0.080" in en and "Reported C - B: -0.090" in en


def test_sentences_differ_and_scale_matches_paper():
    m = load()
    assert len(set(m.EN.values())) == 5 and len(set(m.TR.values())) == 5 and set(m.EN) == set(m.READINGS) == set(m.TR)
    lim = (ROOT / "paper/sections/limitations.tex").read_text()
    assert "$512$ root graphs" in lim and "Qwen3.5-2B" in lim and "LoRA" in lim


def test_secondary_facts_stated_not_changing_sentence():
    m = load()
    en = m.render(_primary("positive", task_drop=["critical_edge"], same=False), "en")
    assert m.EN["positive"] in en and "critical_edge" in en and "splits have the sign of the pooled difference: no" in en


def test_refuses_without_file_or_d_row(tmp_path):
    m = load()
    with pytest.raises(SystemExit, match="does not exist"):
        m.main(["--primary", str(tmp_path / "none.json"), "--out-en", str(tmp_path / "a"), "--out-tr", str(tmp_path / "b")])
    t = _primary("positive")
    t["differences"] = [d for d in t["differences"] if d["a"] != "D"]
    f = tmp_path / "p.json"; f.write_text(json.dumps(t))
    with pytest.raises(SystemExit, match="no D - B row"):
        m.main(["--primary", str(f), "--out-en", str(tmp_path / "a"), "--out-tr", str(tmp_path / "b")])
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    f.write_text(json.dumps({"statistic": "other"}))
    with pytest.raises(SystemExit, match="not a final_primary"):
        m.main(["--primary", str(f)])


def test_real_repository_refuses_today():
    m = load()
    if (ROOT / "reports/final/final_primary.json").exists():
        pytest.skip("locked test opened")
    with pytest.raises(SystemExit, match="does not exist"):
        m.load(ROOT / "reports/final/final_primary.json")


def test_writes_both_files(tmp_path):
    m = load()
    f = tmp_path / "p.json"; f.write_text(json.dumps(_primary("negative")))
    m.main(["--primary", str(f), "--out-en", str(tmp_path / "en.md"), "--out-tr", str(tmp_path / "tr.md")])
    assert m.EN["negative"] in (tmp_path / "en.md").read_text()
    assert m.TR["negative"] in (tmp_path / "tr.md").read_text()
