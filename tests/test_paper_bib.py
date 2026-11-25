"""Every \\cite has a \\bibitem, and every arXiv bibitem matches the saved arXiv API record."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = {"a": "http://www.w3.org/2005/Atom"}


def _bibitems():
    s = (ROOT / "paper/main.tex").read_text()
    body = s[s.index("\\begin{thebibliography}"):s.index("\\end{thebibliography}")]
    items = {}
    for chunk in body.split("\\bibitem{")[1:]:
        key, text = chunk.split("}", 1)
        items[key] = " ".join(text.split())
    return items


def _arxiv():
    out = {}
    entries = [e for f in sorted((ROOT / "docs/sources").glob("arxiv_bib_*.xml"))
               for e in ET.parse(f).getroot().findall("a:entry", NS)]
    for e in entries:
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1].split("v")[0]
        out[aid] = (" ".join(e.find("a:title", NS).text.split()),
                    [a.find("a:name", NS).text for a in e.findall("a:author", NS)])
    return out


def test_every_cite_has_bibitem():
    items = _bibitems()
    tex = "".join(p.read_text() for p in (ROOT / "paper").rglob("*.tex"))
    keys = {k.strip() for g in re.findall(r"\\cite\{([^}]*)\}", tex) for k in g.split(",")}
    assert keys and keys <= set(items), keys - set(items)


def test_bibitems_match_arxiv_record():
    records = _arxiv()
    for key, text in _bibitems().items():
        m = re.search(r"arXiv:(\d{4}\.\d{5})", text)
        assert m, key
        title, authors = records[m.group(1)]
        assert title in text, (key, title)
        for name in authors:
            assert name in text, (key, name)
