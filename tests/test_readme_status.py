"""README status numbers must equal the locked-test primary statistic (reports/final/final_primary.json)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "reports" / "final" / "final_primary.json"


def _status():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("**Status (")
    return text[start:text.index("Per-adapter cards:", start)]


def test_readme_status_matches_final_primary():
    if not PRIMARY.exists():
        pytest.skip("locked test not opened")
    d = json.loads(PRIMARY.read_text(encoding="utf-8"))
    status = _status()
    assert "no locked-test result" not in status
    assert "pilot scale" in status
    for arm in d["arms"]:
        point = arm["pooled"]["primary_six_cell"]["point"]
        assert f"{arm['label']} {point:.3f}" in status, arm["label"]
    for diff in d["differences"]:
        if diff["b"] != "B":
            continue
        c = diff["diff"]["primary_six_cell"]
        name = f"{diff['a']} - B = {c['point']:+.3f}"
        assert name in status, name
        if diff["a"] in ("D", "E"):
            assert f"[{c['lo']:+.3f}, {c['hi']:+.3f}]" in status, name
            assert f"`{diff['reading']}`" in status, name


def _section(title):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index(title)
    end = text.find("\n#", start + len(title))
    return text[start:end if end != -1 else len(text)]


def test_readme_locked_test_reports_exist_and_style_cards_are_complete():
    """Every report the README's locked-test section names exists, and the style-card page references only
    images that are in the repository (2026-11-22: the images were committed without the page)."""
    import re
    if not PRIMARY.exists():
        pytest.skip("locked test not opened")
    sec = _section("### Locked-test reports")
    names = re.findall(r"`(reports/final/[^`]+)`", sec)
    names += ["reports/final/" + n for n in re.findall(r"`(final_[a-z]+\.md)`", sec)]
    assert len(names) >= 10
    for n in names:
        assert (ROOT / n).is_file(), n
    cards = (ROOT / "reports/final/style_cards.md").read_text(encoding="utf-8")
    imgs = set(re.findall(r"\((style_cards/[^)]+\.png)\)", cards))
    assert imgs
    for img in imgs:
        assert (ROOT / "reports/final" / img).is_file(), img
    assert (ROOT / "reports/final/style_cards.json").is_file()


def test_status_external_vgcure_numbers_match_record():
    rec = json.loads((ROOT / "reports/external/vgcure_final_seeds_vs_B.json").read_text(encoding="utf-8"))
    assert rec["record"] == "preregistered" and rec["reference"] == "B"
    d = rec["arms"]["D"]["diff_pooled"]["external_macro"]
    status = " ".join(_status().split())
    assert "D is ahead of B, %+.3f [%+.3f, %+.3f] at every seed" % (d["point"], d["lo"], d["hi"]) in status
    assert all(s["external_macro"]["lo"] > 0 for s in rec["arms"]["D"]["diff_per_seed"])
    assert rec["arms"]["E"]["diff_pooled"]["external_macro"]["hi"] < 0 and "E falls behind B" in status
