"""protocol_freeze.py: the pre-registered rules must be the ones in force when the locked test was opened."""
import json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import protocol_freeze as pf  # noqa: E402

PROTOCOL = (ROOT / "docs" / "protocol.md").read_text()
RECORD = json.loads((ROOT / "docs" / "protocol_freeze.json").read_text())


def test_rules_unchanged_since_the_test_was_opened():
    ok, lines = pf.check()
    assert ok, lines
    assert lines[-1].endswith("-> ok") and len(lines) == 2 + len(pf.RULE_FILES)


def test_record_is_taken_at_the_eval_freeze_commit():
    ev = json.loads((ROOT / "scripts" / "kaggle" / "eval_freeze.json").read_text())
    assert RECORD["commit"] == ev["commit"]
    assert sorted(RECORD["files"]) == sorted(pf.RULE_FILES)
    assert RECORD["commit"][:7] in PROTOCOL  # the header names the commit the record was taken at


def test_blob_id_is_gits():
    assert pf.blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"  # git's well-known empty blob


def _copy(tmp_path):
    for f in [pf.PROTOCOL] + pf.RULE_FILES:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, tmp_path / f)
    return tmp_path


def test_edit_in_sections_1_to_6_is_caught_but_not_in_7_or_8(tmp_path):
    r = _copy(tmp_path)
    p = r / pf.PROTOCOL
    p.write_text(PROTOCOL + "\nA later note in section 8.\n")
    assert pf.check(root=r)[0]
    p.write_text(PROTOCOL.replace("about 5 percentage points", "about 3 percentage points"))
    ok, lines = pf.check(root=r)
    assert not ok and "sections 1-6: CHANGED" in lines[0] and lines[-1].endswith("-> stop")


def test_changed_or_missing_rule_file_is_caught(tmp_path):
    r = _copy(tmp_path)
    q = r / "scripts/kaggle/final_queue.json"
    q.write_text(q.read_text() + " ")
    ok, lines = pf.check(root=r)
    assert not ok and any(l == "scripts/kaggle/final_queue.json: CHANGED" for l in lines)
    q.unlink()
    assert any(l.endswith("MISSING") for l in pf.check(root=r)[1])


def test_version_record_states_the_deviations():
    head = PROTOCOL[:PROTOCOL.index("## 1. ")]
    assert "Version 1.0" in head and "0.1 draft" in head  # says plainly that the number was raised late
    sec8 = PROTOCOL[PROTOCOL.index("## 8. "):]
    for needle in ("512 training roots", "200 / 200 / 200 / 100", "final_primary.py", "Not run on any model"):
        assert needle in sec8
    # numbers quoted in section 8 / items 1 and 5 come from tracked records
    bp = json.loads((ROOT / "reports" / "pilot" / "base_prompt.json").read_text())
    assert bp["chosen"] == "system" and bp["chosen_limit"] == 96 and "`system` at 96 tokens" in sec8
    v = bp["variants"]
    assert (v["common"]["non_edit_macro"], v["common"]["non_edit_macro_lenient"], v["common"]["max_truncation_rate"]) == (0.1375, 0.6125, 0.9)
    assert (v["direct"]["non_edit_macro"], v["system"]["non_edit_macro"]) == (0.6, 0.625)
    assert "0.1375 vs lenient 0.6125" in PROTOCOL
    for run, limit in [("pilot3_B_kaggle", 96), ("pilot3_C", 96), ("pilot3_D_kaggle", 96), ("pilot3_D2", 96), ("pilot3_E", 512)]:
        ev = json.loads((ROOT / "runs" / run / "summary.json").read_text())["eval_dev"]
        assert ev["max_new_tokens"] == limit and ev["truncated_at_limit"] == 0
