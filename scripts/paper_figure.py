"""Figure 1 for the paper: one root drawing, its answer-flipping twin, and its repair.

Usage: python scripts/paper_figure.py manifests/pilot.json dev paper/figures/fig1 [group_id]

Writes <out>.png (three panels) and <out>.json (the exact questions, answers and the edge
that differs, taken from the dataset, so the caption is not typed by hand). Without a
group id the first group of the split (file order) that has a critical-edge twin and a
repair sample on its root image is used. Test splits are refused (labels are read).
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_vr.dataset import check_split_allowed  # noqa: E402

FONT = ROOT / "assets" / "fonts" / "DejaVuSans.ttf"
BOLD = ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf"


def _question(sample: Dict[str, Any]) -> str:
    lines = [l for l in sample["public_input"]["instruction"].split("\n") if "?" in l]
    return lines[-1].strip() if lines else ""


def _goals(sample: Dict[str, Any]) -> List[str]:
    return [l[2:].strip() for l in sample["public_input"]["instruction"].split("\n") if l.startswith("- ")]


def _edges(sample: Dict[str, Any]) -> set:
    return {tuple(e) for e in sample["private_truth"]["edges"]}


def pick_group(samples: List[Dict[str, Any]], group_id: str = "") -> Dict[str, Any]:
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for s in samples:
        g = s["graph_group_id"]
        if g not in by_group:
            by_group[g] = []
            order.append(g)
        by_group[g].append(s)
    for g in ([group_id] if group_id else order):
        rows = by_group[g]
        crit = [s for s in rows if s["task"] == "critical_edge" and any(p["kind"] == "critical" for p in s["pairs"])]
        base = [s for s in crit if s["metadata"]["image_variant"] == "root"]
        repair = [s for s in rows if s["task"] == "minimal_graph_repair"
                  and s["metadata"]["image_variant"] == "root"
                  and s["private_truth"]["status"] == "edited"]
        if not base or not repair:
            continue
        pid = next(p["pair_id"] for p in base[0]["pairs"] if p["kind"] == "critical")
        twin = [s for s in crit if s is not base[0] and any(p["pair_id"] == pid for p in s["pairs"])]
        if not twin:
            continue
        return {"group": g, "base": base[0], "twin": twin[0], "repair": repair[0]}
    raise SystemExit("no group with a critical twin and an edited repair on the root image")


def describe(pick: Dict[str, Any]) -> Dict[str, Any]:
    b, t, r = pick["base"], pick["twin"], pick["repair"]
    removed = sorted(_edges(b) - _edges(t))
    added = sorted(_edges(t) - _edges(b))
    return {
        "group_id": pick["group"],
        "root_image": b["image_path"],
        "twin_image": t["image_path"],
        "question": _question(b),
        "root_answer": b["private_truth"]["answer"]["answer"],
        "twin_answer": t["private_truth"]["answer"]["answer"],
        "twin_removes": [f"{u}->{v}" for u, v in removed],
        "twin_adds": [f"{u}->{v}" for u, v in added],
        "repair_goals": _goals(r),
        "repair_minimal_solutions": r["private_truth"]["all_minimal_solutions"],
    }


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> List[str]:
    out, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def render(data_dir: Path, info: Dict[str, Any], out_png: Path) -> None:
    size, pad, text_h = 512, 16, 140
    font = ImageFont.truetype(str(FONT), 17)
    bold = ImageFont.truetype(str(BOLD), 19)
    sol = info["repair_minimal_solutions"][0]
    sol_txt = ", ".join(f"{e['op']} {e['src']}->{e['dst']}" for e in sol)
    panels = [
        (info["root_image"], "(a) root",
         [info["question"], f"answer: {json.dumps(info['root_answer'])}"]),
        (info["twin_image"], "(b) twin: one edge differs",
         [f"same question; edge {', '.join(info['twin_removes'] + info['twin_adds'])} "
          f"{'removed' if info['twin_removes'] else 'added'}",
          f"answer: {json.dumps(info['twin_answer'])}"]),
        (info["root_image"], "(c) repair on the root",
         ["goals: " + "; ".join(g.rstrip(".") for g in info["repair_goals"]), "minimal edit: " + sol_txt]),
    ]
    canvas = Image.new("RGB", (3 * size + 4 * pad, size + text_h + 2 * pad), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (path, title, lines) in enumerate(panels):
        x = pad + i * (size + pad)
        canvas.paste(Image.open(data_dir / path).convert("RGB").resize((size, size)), (x, pad))
        draw.rectangle([x, pad, x + size - 1, pad + size - 1], outline="#888888")
        y = pad + size + 8
        draw.text((x, y), title, font=bold, fill="black")
        y += 28
        for line in lines:
            for w in _wrap(draw, line, font, size):
                draw.text((x, y), w, font=font, fill="black")
                y += 22
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, optimize=True)


def main(argv: List[str]) -> None:
    manifest, split, out = Path(argv[0]), argv[1], Path(argv[2])
    group = argv[3] if len(argv) > 3 else ""
    check_split_allowed(split, "eval")
    m = json.loads(manifest.read_text())
    data_dir = (manifest.parent / m["data_dir"]).resolve()
    samples = [json.loads(l) for l in (data_dir / f"{split}.samples.jsonl").read_text().splitlines() if l]
    info = describe(pick_group(samples, group))
    info.update({"manifest": str(manifest), "split": split, "content_digest": m["content_digest"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    render(data_dir, info, out.with_suffix(".png"))
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
