"""Repeatable arXiv overlap search for the related-work section.

Runs a fixed list of arXiv API queries (no web search engine, no sign-in), keeps the raw
Atom XML under an output directory, and writes a deduplicated list of hits (id, version
date, title, matching queries) as JSON and Markdown. Titles and abstracts are data for a
human to read; nothing here judges relevance.

    python scripts/overlap_search.py --out docs/sources/overlap_2026-11-15
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"

QUERIES = [
    'abs:"graph" AND abs:"vision-language" AND abs:"edit" AND abs:"reachability"',
    'abs:"graph repair" AND abs:"multimodal"',
    'abs:"visual graph" AND abs:"edit"',
    'abs:"vision-language" AND abs:"graph reasoning" AND abs:"fine-tuning"',
    'abs:"graph" AND abs:"image" AND abs:"counterfactual" AND abs:"vision-language"',
    'abs:"VGCure"',
    'abs:"graph" AND abs:"structural" AND abs:"fine-tuning" AND abs:"multimodal"',
    'abs:"rendered" AND abs:"graph" AND abs:"vision-language"',
    'abs:"visual graph reasoning"',
    'abs:"graph" AND abs:"image" AND abs:"shortest path" AND abs:"vision-language"',
    'abs:"diagram" AND abs:"edit" AND abs:"transfer" AND abs:"vision-language"',
    'abs:"minimal edit" AND abs:"vision-language"',
    'abs:"graph extraction" AND abs:"vision-language" AND abs:"solver"',
    'abs:"answer-flipping" OR abs:"contrast set" AND abs:"graph"',
]


def fetch(query: str, max_results: int) -> bytes:
    params = urllib.parse.urlencode(
        {"search_query": query, "max_results": max_results, "sortBy": "relevance"}
    )
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": "trace-overlap-search"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def parse(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        aid = e.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1]
        out.append(
            {
                "id": aid,
                "published": e.findtext(f"{ATOM}published", "")[:10],
                "updated": e.findtext(f"{ATOM}updated", "")[:10],
                "title": " ".join(e.findtext(f"{ATOM}title", "").split()),
                "summary": " ".join(e.findtext(f"{ATOM}summary", "").split()),
            }
        )
    return out


def merge(per_query: list[tuple[int, list[dict]]]) -> list[dict]:
    hits: dict[str, dict] = {}
    for qi, entries in per_query:
        for ent in entries:
            base = ent["id"].split("v")[0] if "v" in ent["id"] else ent["id"]
            h = hits.setdefault(base, {**ent, "id": base, "queries": []})
            h["queries"].append(qi)
    return sorted(hits.values(), key=lambda h: (-len(h["queries"]), h["id"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-results", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=3.0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    per_query = []
    for qi, q in enumerate(QUERIES, 1):
        xml = fetch(q, a.max_results)
        (out / "raw" / f"q{qi:02d}.xml").write_bytes(xml)
        entries = parse(xml)
        per_query.append((qi, entries))
        print(f"q{qi:02d} {len(entries):3d}  {q}")
        time.sleep(a.sleep)
    hits = merge(per_query)
    (out / "hits.json").write_text(
        json.dumps({"queries": QUERIES, "hits": hits}, indent=1, ensure_ascii=False) + "\n"
    )
    lines = ["| id | updated | queries | title |", "|---|---|---|---|"]
    for h in hits:
        lines.append(f"| {h['id']} | {h['updated']} | {','.join(map(str, h['queries']))} | {h['title']} |")
    (out / "hits.md").write_text("\n".join(lines) + "\n")
    print(f"{len(hits)} unique hits")


if __name__ == "__main__":
    main()
