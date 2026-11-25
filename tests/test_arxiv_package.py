"""scripts/arxiv_package.py builds a flat, self-contained arXiv source package (no compile here)."""
import importlib.util, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("arxiv_package", ROOT / "scripts/arxiv_package.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_package_is_flat_and_complete(tmp_path):
    mod = _mod()
    pkg = mod.build(tmp_path)
    texts = {p: p.read_text() for p in pkg.rglob("*.tex")}
    assert texts[pkg / "main.tex"].startswith("\\pdfoutput=1\n")
    for p, t in texts.items():
        assert "../" not in t, p
        assert not any(l.lstrip().startswith("%") for l in t.splitlines()), p
    assert (pkg / "final/abstract.tex").exists() and (pkg / "figures/fig1.png").exists()
    assert not list(pkg.rglob("*.json"))
    found = [b for b in mod.blockers(pkg) if "not in package" in b or "needs" in b or "climbs" in b]
    assert found == []


def test_missing_file_is_a_blocker(tmp_path):
    mod = _mod()
    pkg = mod.build(tmp_path)
    (pkg / "figures/arms_dev.pdf").unlink()
    (pkg / "final/final_primary.tex").unlink()
    b = mod.blockers(pkg)
    assert any("arms_dev.pdf" in x for x in b)
    assert any("final_primary" in x for x in b)


def test_tarball_is_reproducible(tmp_path):
    mod = _mod()
    digests = []
    for d in ("a", "b"):
        assert mod.main(["--out", str(tmp_path / d)]) in (0, 2)
        digests.append((tmp_path / d / "trace-arxiv.tar.gz").read_bytes())
    assert digests[0] == digests[1]


def test_refuses_to_write_into_the_repo():
    mod = _mod()
    try:
        mod.main(["--out", str(ROOT / "paper")])
    except SystemExit as e:
        assert "outside the repo" in str(e)
    else:
        raise AssertionError("wrote into paper/")
