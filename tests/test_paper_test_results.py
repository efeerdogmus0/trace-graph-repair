"""paper/sections/test_results.tex: wired to the final report scripts, no hand-typed numbers."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "paper" / "sections" / "test_results.tex").read_text()
BODY = "\n".join(l for l in RAW.splitlines() if not l.lstrip().startswith("%"))
MAIN = (ROOT / "paper" / "main.tex").read_text()
SECTIONS = "".join(p.read_text() for p in (ROOT / "paper" / "sections").glob("*.tex"))


def _order():
    import importlib.util
    spec = importlib.util.spec_from_file_location("final_reports", ROOT / "scripts" / "final_reports.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ORDER


def test_wired_into_main_once_after_dev_results():
    assert MAIN.count("\\input{sections/test_results}") == 1
    assert MAIN.index("\\input{sections/results}") < MAIN.index("\\input{sections/test_results}")


def test_every_final_report_has_a_table_and_nothing_else():
    names = re.findall(r"\\final(?:long)?table\{([^}]+)\}", BODY)
    assert sorted(names) == sorted(_order())
    assert len(names) == len(set(names))
    assert names[0] == "final_primary"  # the deciding statistic comes first


def test_no_hand_typed_numbers():
    text = re.sub(r"\\setlength\{\\tabcolsep\}\{\d+pt\}", "", BODY)   # layout, not a result
    text = re.sub(r"\\(label|ref|finaltable|finallongtable|finalfigure|input|IfFileExists|texttt|includegraphics(?:\[[^\]]*\])?)\{[^}]*\}", "", text)
    text = text.replace("\\newcommand{\\finaltable}[3]", "").replace("\\newcommand{\\finallongtable}[3]", "").replace("\\newcommand{\\finalfigure}[3]", "").replace("95\\%", "").replace("#1", "").replace("#2", "").replace("#3", "")
    assert not re.search(r"\d", text), re.findall(r".{20}\d.{20}", text)


def test_refs_resolve():
    labels = set(re.findall(r"\\label\{([^}]+)\}", SECTIONS))
    for ref in re.findall(r"\\ref\{([^}]+)\}", BODY):
        assert ref in labels, ref


def test_longtable_reports_are_not_floated():
    """final_slices.tex is a longtable (rehearse_final_reports.py found it broke the build inside a float):
    it goes through \\finallongtable, and main.tex loads longtable."""
    assert re.findall(r"\\finallongtable\{([^}]+)\}", BODY) == ["final_slices"]
    assert re.search(r"\\usepackage\{[^}]*\blongtable\b[^}]*\}", MAIN)
    src = (ROOT / "scripts" / "final_slices.py").read_text()
    assert "\\begin{longtable}" in src
    for name in _order():
        if name != "final_slices":
            assert "longtable" not in (ROOT / "scripts" / (name + ".py")).read_text(), name


def test_ood_figure_is_wired_to_final_figures_only():
    """The locked-test figure is the one scripts/final_figures.py draws; it sits right after the per-split table,
    and no figure of this section exists before the locked test is reported."""
    assert re.findall(r"\\finalfigure\{([^}]+)\}", BODY) == ["ood_test"]
    assert BODY.index("\\finaltable{final_table}") < BODY.index("\\finalfigure{ood_test}") < BODY.index("\\finaltable{final_pairs}")
    src = (ROOT / "scripts" / "final_figures.py").read_text()
    assert '"ood_test"' in src and "final_table.json" in src
    assert (ROOT / "reports/final/final_table.json").exists() or not (ROOT / "paper/figures/ood_test.pdf").exists()
