import importlib.util, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("datasheet", ROOT / "scripts" / "datasheet.py")
ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)


def _load(name):
    return (json.loads((ROOT / "manifests" / ("%s.json" % name)).read_text()),
            json.loads((ROOT / "reports" / name / "validation.json").read_text()))


def test_datasheet_totals_match_manifest():
    for name in ("smoke", "pilot"):
        man, val = _load(name)
        text = ds.render(man, val)
        splits = list(man.get("split_order") or man["splits"])
        # samples-by-task table total equals the manifest sample count, per split
        block = text.split("### Samples by task")[1].split("###")[0]
        totals = re.search(r"\| \*\*total\*\* \| (.+) \|", block).group(1).split(" | ")
        assert [int(t) for t in totals] == [man["splits"][s]["samples"] for s in splits]
        block = text.split("### Root graphs by family")[1].split("###")[0]
        totals = re.search(r"\| \*\*total\*\* \| (.+) \|", block).group(1).split(" | ")
        assert [int(t) for t in totals] == [man["splits"][s]["roots"] for s in splits]
        assert man["content_digest"] in text


def test_committed_pilot_datasheet_is_current():
    man, val = _load("pilot")
    assert (ROOT / "docs" / "datasheet_pilot_stats.md").read_text() == ds.render(man, val)
