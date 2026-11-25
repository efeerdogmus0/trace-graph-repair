import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overlap_search", ROOT / "scripts" / "overlap_search.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FEED = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry><id>http://arxiv.org/abs/2609.22327v1</id><published>2026-09-16T00:00:00Z</published>
  <updated>2026-09-16T00:00:00Z</updated><title>Visual Graph
   Reasoning</title><summary> a  b </summary></entry>
 <entry><id>http://arxiv.org/abs/2405.04950v2</id><published>2024-05-08T00:00:00Z</published>
  <updated>2024-06-01T00:00:00Z</updated><title>VisionGraph</title><summary>x</summary></entry>
</feed>"""


def test_parse_strips_whitespace_and_dates():
    e = mod.parse(FEED)
    assert [x["id"] for x in e] == ["2609.22327v1", "2405.04950v2"]
    assert e[0]["title"] == "Visual Graph Reasoning"
    assert e[0]["summary"] == "a b"
    assert e[1]["updated"] == "2024-06-01"


def test_merge_dedups_versions_and_ranks_by_query_count():
    e = mod.parse(FEED)
    hits = mod.merge([(1, e), (2, e[:1])])
    assert [h["id"] for h in hits] == ["2609.22327", "2405.04950"]
    assert hits[0]["queries"] == [1, 2]
    assert hits[1]["queries"] == [1]


def test_saved_search_matches_script_queries():
    import json

    saved = json.loads((ROOT / "docs/sources/overlap_2026-11-15/hits.json").read_text())
    assert saved["queries"] == mod.QUERIES
    assert len(list((ROOT / "docs/sources/overlap_2026-11-15/raw").glob("q*.xml"))) == len(mod.QUERIES)
