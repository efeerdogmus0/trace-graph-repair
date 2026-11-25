"""Contact sheets and a browsable HTML page for visual inspection of a dataset."""
from __future__ import annotations

import html
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from . import dataset, render, tasks

THUMB = 320
PAD = 10
CAPTION_LINES = 4
LINE_H = 16


def _font():
    return render.load_font("DejaVuSans.ttf", 12)


def _short(sample: Dict[str, Any]) -> str:
    q = sample["public_input"]["query"]
    t = sample["training_target"]
    kind = q["type"]
    if kind == tasks.REPAIR:
        goal = "; ".join(["%s=>%s" % tuple(p) for p in q["must_reach"]] + ["%s=/>%s" % tuple(p) for p in q["must_not_reach"]])
        adds = ",".join("%s->%s" % tuple(p) for p in q["allowed_additions"])
        if t["status"] == "edited":
            ans = " ".join("%s %s->%s" % (e["op"].split("_")[0], e["src"], e["dst"]) for e in t["edits"])
        else:
            ans = t["status"]
        return "repair [%s] add:{%s}  ->  %s" % (goal, adds, ans)
    if kind == tasks.REACHABILITY:
        return "reach %s=>%s ?  ->  %s" % (q["src"], q["dst"], t["answer"])
    if kind == tasks.CRITICAL_EDGE:
        return "critical %s->%s for %s=>%s ?  ->  %s" % (q["edge"][0], q["edge"][1], q["src"], q["dst"], t["answer"])
    if kind == tasks.EDGE_EXISTS:
        return "edge %s->%s ?  ->  %s" % (q["src"], q["dst"], t["answer"])
    if kind == tasks.SHORTEST_PATH:
        return "shortest %s=>%s  ->  %s" % (q["src"], q["dst"], t["answer"])
    return "extract graph"


def _cell(data_dir: Path, image_path: str, lines: List[str]) -> Image.Image:
    cell = Image.new("RGB", (THUMB + 2 * PAD, THUMB + 2 * PAD + CAPTION_LINES * LINE_H), (255, 255, 255))
    pic = Image.open(data_dir / image_path).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
    cell.paste(pic, (PAD, PAD))
    d = ImageDraw.Draw(cell)
    d.rectangle([PAD - 1, PAD - 1, PAD + THUMB, PAD + THUMB], outline=(160, 160, 160))
    wrapped: List[str] = []
    for line in lines:
        wrapped += textwrap.wrap(line, 52) or [""]
    for i, line in enumerate(wrapped[:CAPTION_LINES]):
        d.text((PAD, PAD + THUMB + 4 + i * LINE_H), line, font=_font(), fill=(0, 0, 0))
    return cell


def _sheet(cells: List[Image.Image], cols: int) -> Image.Image:
    w, h = cells[0].size
    rows = -(-len(cells) // cols)
    sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * w, (i // cols) * h))
    return sheet


def render_gallery(manifest_path: Path, out_dir: Path, per_split: int = 12, pairs_per_task: int = 6) -> List[Path]:
    manifest, data_dir, samples_by_split, images_by_split = dataset.load_dataset(Path(manifest_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for split, images in images_by_split.items():
        roots = [r for r in images if r["variant"] == "root"][:per_split]
        if not roots:
            continue
        cells = [
            _cell(data_dir, r["image_path"], [
                "%s  n=%d m=%d" % (r["graph_family"], len(r["nodes"]), len(r["edges"])),
                "style %s  layout %s  crossings %d" % (r["style_id"], r["layout"], r["crossings"]),
            ])
            for r in roots
        ]
        path = out_dir / ("roots_%s.png" % split)
        _sheet(cells, 4).save(path)
        written.append(path)

    for split, samples in samples_by_split.items():
        by_id = {s["sample_id"]: s for s in samples}
        by_pair: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples:
            for p in s["pairs"]:
                by_pair.setdefault(p["pair_id"], []).append(s)
        for kind in ("critical", "invariant_relabel", "invariant_style"):
            for task in (tasks.REPAIR, tasks.REACHABILITY, tasks.CRITICAL_EDGE):
                cells: List[Image.Image] = []
                for s in samples:
                    if s["task"] != task:
                        continue
                    for p in s["pairs"]:
                        if p["kind"] != kind or p["role"] != "base":
                            continue
                        other = [o for o in by_pair[p["pair_id"]] if o["sample_id"] != s["sample_id"]][0]
                        cells.append(_cell(data_dir, s["image_path"], ["BASE  " + _short(s)]))
                        cells.append(_cell(data_dir, other["image_path"], ["VARIANT  " + _short(other)]))
                    if len(cells) >= 2 * pairs_per_task:
                        break
                if cells:
                    path = out_dir / ("pairs_%s_%s_%s.png" % (split, kind, task))
                    _sheet(cells[: 2 * pairs_per_task], 4).save(path)
                    written.append(path)

    written.append(_write_html(manifest, data_dir, samples_by_split, images_by_split, out_dir))
    return written


def _write_html(manifest, data_dir: Path, samples_by_split, images_by_split, out_dir: Path) -> Path:
    rel = Path(os.path.relpath(data_dir, out_dir)).as_posix()
    parts = [
        "<!doctype html><meta charset='utf-8'><title>TRACE dataset %s</title>" % html.escape(manifest["name"]),
        "<style>body{font:14px/1.4 sans-serif;margin:16px;max-width:1200px}"
        ".card{display:flex;gap:16px;border-top:1px solid #ccc;padding:12px 0}"
        "img{width:320px;height:320px;border:1px solid #ccc}code{background:#f2f2f2;padding:1px 4px}"
        "li{margin-bottom:4px}</style>",
        "<h1>TRACE dataset: %s</h1>" % html.escape(manifest["name"]),
        "<p>Ground truth is shown here for inspection only; a model never receives it.</p>",
    ]
    for split, images in images_by_split.items():
        by_image: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples_by_split[split]:
            by_image.setdefault(s["image_id"], []).append(s)
        parts.append("<h2>%s (%d images)</h2>" % (html.escape(split), len(images)))
        for r in images:
            items = "".join("<li><code>%s</code></li>" % html.escape(_short(s)) for s in by_image.get(r["image_id"], []))
            edit = "" if not r["edit"] else " edit: %s %s-&gt;%s" % (r["edit"]["op"], r["edit"]["src"], r["edit"]["dst"])
            parts.append(
                "<div class='card'><img loading='lazy' src='%s/%s'><div><b>%s</b> group %s<br>%s, n=%d, m=%d, style %s, "
                "layout %s, crossings %d%s<ul>%s</ul></div></div>"
                % (rel, r["image_path"], r["variant"], r["graph_group_id"], r["graph_family"], len(r["nodes"]),
                   len(r["edges"]), r["style_id"], r["layout"], r["crossings"], edit, items)
            )
    path = out_dir / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
