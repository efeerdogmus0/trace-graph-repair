"""paper_conclusion.py: conclusion body from a final_primary.json. Synthetic files only."""
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
def test_each_reading_uses_only_its_sentences(reading):
    m = load("paper_conclusion")
    text = m.render(synthetic(reading))
    pf = m.pf
    for part in (m.CONTRIB, pf.FINDING[reading], m.MEANING[reading], m.NEXT[reading], pf.SCALE, m.LIMITS):
        assert part in text
    assert "D$-$B is $+0.050$ (95\\% interval [$+0.020$, $+0.080$], mean of 3 seeds)" in text
    assert "conclusion-v1" in text and "\\ref{sec:test-results}" in text
    for other in set(m.READINGS) - {reading}:
        for table in (pf.FINDING, m.MEANING, m.NEXT):
            assert table[other] not in text


def test_sentence_tables_cover_readings_and_agree_with_abstract():
    m, pf = load("paper_conclusion"), load("paper_finding")
    for table in (m.MEANING, m.NEXT):
        assert set(table) == set(pf.READINGS) and len(set(table.values())) == 5
    assert m.pf.FINDING == pf.FINDING and m.pf.SCALE == pf.SCALE
    # every label the text points to exists in the paper
    labels = "".join(p.read_text() for p in (ROOT / "paper/sections").glob("*.tex"))
    for lab in ("sec:negative", "sec:limitations", "sec:test-results"):
        assert "\\label{%s}" % lab in labels
    assert "\\paragraph{Controls not run.}" in (ROOT / "paper/sections/limitations.tex").read_text()


def test_refuses_without_primary(tmp_path):
    m = load("paper_conclusion")
    with pytest.raises(SystemExit, match="does not exist"):
        m.main(["--primary", str(tmp_path / "none.json"), "--out", str(tmp_path / "o/conclusion.tex")])
    assert not (tmp_path / "o").exists()


def test_writes_file(tmp_path):
    m = load("paper_conclusion")
    p = tmp_path / "final_primary.json"
    p.write_text(json.dumps(synthetic("inconclusive")))
    m.main(["--primary", str(p), "--out", str(tmp_path / "o/conclusion.tex")])
    assert m.MEANING["inconclusive"] in (tmp_path / "o/conclusion.tex").read_text()


def test_paper_wired_and_no_conclusion_before_reading():
    main = (ROOT / "paper/main.tex").read_text()
    sec = (ROOT / "paper/sections/conclusion.tex").read_text()
    assert main.index("\\input{sections/limitations}") < main.index("\\input{sections/conclusion}") < main.index("\\appendix")
    assert "\\IfFileExists{../reports/final/conclusion.tex}{\\input{../reports/final/conclusion.tex}}" in sec
    assert "No result is claimed yet." in sec
    if not (ROOT / "reports/final/final_primary.json").exists():
        assert not (ROOT / "reports/final/conclusion.tex").exists()
