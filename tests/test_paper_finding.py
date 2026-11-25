"""paper_finding.py: abstract / intro finding paragraph from a final_primary.json. Synthetic files only."""
import importlib.util, json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def synthetic(reading):
    t = importlib.util.spec_from_file_location("trs", ROOT / "tests/test_result_summary.py")
    m = importlib.util.module_from_spec(t)
    t.loader.exec_module(m)
    return m._primary(reading)


@pytest.mark.parametrize("reading", ["positive_clear", "positive", "negative", "inconclusive", "incomplete_seeds"])
def test_each_reading_fixed_sentence_numbers_scale(reading):
    m = load("paper_finding")
    out = m.render(synthetic(reading))
    for text in out.values():
        assert m.FINDING[reading] in text and m.SCALE in text
        assert "D$-$B is $+0.050$ (95\\% interval [$+0.020$, $+0.080$], mean of 3 seeds)" in text
        assert "finding-v1" in text
        for other in set(m.READINGS) - {reading}:
            assert m.FINDING[other] not in text
    assert m.BENCH in out["abstract.tex"] and "\\ref{sec:test-results}" in out["intro_finding.tex"]


def test_sentences_cover_readings_and_match_summary_rule():
    m, rs = load("paper_finding"), load("result_summary")
    assert set(m.FINDING) == set(rs.READINGS) and len(set(m.FINDING.values())) == 5
    lim = (ROOT / "paper/sections/limitations.tex").read_text()
    assert "$512$ root graphs" in lim and "Qwen3.5-2B" in lim and "LoRA" in lim
    assert "$0.50$" in (ROOT / "paper/sections/intro.tex").read_text()  # same design bound as BENCH


def test_refuses_without_primary(tmp_path):
    m = load("paper_finding")
    with pytest.raises(SystemExit, match="does not exist"):
        m.main(["--primary", str(tmp_path / "none.json"), "--out-dir", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()


def test_writes_both_files(tmp_path):
    m = load("paper_finding")
    p = tmp_path / "final_primary.json"
    p.write_text(json.dumps(synthetic("negative")))
    m.main(["--primary", str(p), "--out-dir", str(tmp_path / "o")])
    assert sorted(x.name for x in (tmp_path / "o").iterdir()) == ["abstract.tex", "intro_finding.tex"]


def test_paper_wired_and_no_finding_before_reading():
    main = (ROOT / "paper/main.tex").read_text()
    intro = (ROOT / "paper/sections/intro.tex").read_text()
    assert "\\IfFileExists{../reports/final/abstract.tex}{\\input{../reports/final/abstract.tex}}" in main
    assert "\\IfFileExists{../reports/final/intro_finding.tex}{\\input{../reports/final/intro_finding.tex}}{}" in intro
    if not (ROOT / "reports/final/final_primary.json").exists():
        assert not (ROOT / "reports/final/abstract.tex").exists()
        assert not (ROOT / "reports/final/intro_finding.tex").exists()
