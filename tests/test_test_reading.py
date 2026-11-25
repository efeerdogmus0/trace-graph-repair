"""test_reading.py: descriptive locked-test reading paragraph from a final_primary.json."""
import importlib.util, json, re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tr():
    return load("test_reading", ROOT / "scripts/test_reading.py")


def synthetic(reading):
    return load("trs", ROOT / "tests/test_result_summary.py")._primary(reading)


@pytest.mark.parametrize("reading", ["positive_clear", "negative", "inconclusive"])
def test_every_row_same_frame_outcome_blind(reading):
    t = synthetic(reading)
    out = tr().render(t)
    assert "reading-v1" in out
    items = [l for l in out.splitlines() if l.startswith("\\item ")]
    assert len(items) == len(t["differences"])
    for d, line in zip(t["differences"], items):
        assert line.startswith("\\item \\textbf{%s$-$%s:} six-cell mean " % (d["a"], d["b"]))
        assert "\\texttt{%s}" % d["reading"].replace("_", "\\_") in line
        e = d["diff"]["primary_six_cell"]
        assert "$%+.3f$ [$%+.3f$, $%+.3f$]" % (e["point"], e["lo"], e["hi"]) in line
    # no evaluative words: the frame must not depend on the outcome
    for w in ("better", "worse", "improve", "hurt", "strong", "weak", "significant"):
        assert w not in out.lower()


def test_refuses_without_primary(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        tr().main(["--primary", str(tmp_path / "none.json"), "--out", str(tmp_path / "o.tex")])
    assert not (tmp_path / "o.tex").exists()


def test_tracked_output_matches_primary_and_is_wired():
    sec = (ROOT / "paper/sections/test_results.tex").read_text()
    assert "\\IfFileExists{../reports/final/test_reading.tex}{\\input{../reports/final/test_reading.tex}}{}" in sec
    assert sec.index("{tab:test-primary}") < sec.index("test_reading.tex")
    p = ROOT / "reports/final/final_primary.json"
    out = ROOT / "reports/final/test_reading.tex"
    if not p.exists():
        assert not out.exists()
        return
    m = tr()
    assert out.read_text() == m.render(m.rs.load(str(p))), "rerun scripts/test_reading.py"
    assert re.search(r"\\ref\{tab:test-primary\}", out.read_text())
