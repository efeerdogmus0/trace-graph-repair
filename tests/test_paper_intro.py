"""paper/sections/intro.tex is written before results: no measured number, no result claim."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTRO = (ROOT / "paper" / "sections" / "intro.tex").read_text()
MAIN = (ROOT / "paper" / "main.tex").read_text()
SECTIONS = "".join(p.read_text() for p in (ROOT / "paper" / "sections").glob("*.tex"))
BODY = "\n".join(l for l in INTRO.splitlines() if not l.lstrip().startswith("%"))


def test_wired_into_main_once():
    assert MAIN.count("\\input{sections/intro}") == 1
    assert "\\section{Introduction}" not in MAIN


def test_cites_and_refs_resolve():
    keys = set(re.findall(r"\\bibitem\{([^}]+)\}", MAIN))
    for group in re.findall(r"\\cite\{([^}]+)\}", BODY):
        for k in group.split(","):
            assert k.strip() in keys, k
    labels = set(re.findall(r"\\label\{([^}]+)\}", SECTIONS))
    for ref in re.findall(r"\\ref\{([^}]+)\}", BODY):
        assert ref in labels, ref


def test_no_measured_numbers():
    # The only number allowed is the design bound for an image-blind predictor on twins.
    nums = re.findall(r"\d+(?:[.,]\d+)?", re.sub(r"\\(ref|label|cite|url)\{[^}]*\}", "", BODY))
    assert nums == ["0.50"], nums


def test_no_result_claims():
    low = " ".join(BODY.lower().split())
    for phrase in ["we show", "we find", "we found", "outperform", "improves", "significant",
                   "state-of-the-art", "state of the art", "our results"]:
        assert phrase not in low, phrase
