"""Fixed dev gallery: recorded responses of the evaluated arms on a few dev samples, scored by the verifier.

Every response shown here was produced by a model in a recorded evaluation (the predictions file of that
run); nothing is generated now and nothing is written by hand. Verdicts come from metrics.score_sample,
which checks a response against the generator's true graph.

The selection rule (RULE below) was fixed in this file before any per-sample outcome was looked at:

  pool    main samples (root / one-edit images) of the split for which EVERY listed arm has a recorded
          response; with the frozen base included that is the base's smaller evaluation subset.
  order   ascending sha256("gallery-v1:" + sample_id); no score enters the order.
  slots   success            first reachability sample the focus arm answers correctly
          failure            first reachability sample the focus arm answers wrongly
          critical_pair      first critical-edge base sample whose twin is in the pool (both images shown)
          repair_edited      first repair sample whose true status is "edited"
          repair_impossible  first repair sample whose true status is "impossible"
          repair_failure     first repair sample the focus arm gets wrong
  Only `success`, `failure` and `repair_failure` look at an outcome, and only at the focus arm's. A slot
  with no candidate is reported as empty, never filled by another rule. A sample is used by one slot only.

Not in the gallery: style-change pairs. Their variant images were evaluated only on the locked test, whose
outputs stay closed until the final table; no dev response exists for them.

    PYTHONPATH=src python scripts/dev_gallery.py manifests/pilot.json dev --focus D \
        A=<pred_dir>[:<flags.json>] B=... C=... D=... E=<pred_dir>:<flags.json>:<extraction.jsonl> \
        --out reports/pilot/gallery_dev.json --out-md reports/pilot/gallery_dev.md [--copy-images DIR]

`<pred_dir>` holds predictions_<split>.jsonl. With a flags file the predictions file must have the sha256
recorded there (the gallery then shows the very responses the reported numbers were computed from).
A third field names arm E's extraction file; the gallery then says how far the read graph is from the true one.

Edit overlays (OVERLAY below, with --copy-images): on a repair card every edit named by the reference answer or by a
response that parses under the strict repair schema is drawn onto a copy of the dataset image, one image per
distinct edit. A removal repaints the edge's own recorded route and puts a cross at its middle; an addition is a
straight dashed arrow between the two nodes (a marker, not a route laid out by the generator: it may pass over
other edges or nodes). The colour says the operation only, never whether the edit is right; the verdict stays in
the table. An edit that cannot be shown (a node or edge that is not in the image, an edge already there, a self
loop) is listed with that reason and not drawn. The overlay does not change which samples are shown.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

from trace_vr import dataset, geometry as geo, metrics, semantics as sem, styles, tasks

ROOT = Path(__file__).resolve().parents[1]
RULE = "gallery-v1"
SLOTS = ("success", "failure", "critical_pair", "repair_edited", "repair_impossible", "repair_failure")
SLOT_TEXT = {
    "success": "Reachability, first sample the focus arm answers correctly",
    "failure": "Reachability, first sample the focus arm answers wrongly",
    "critical_pair": "Critical edge: one question on two images that differ by one edge (chosen without looking at any response)",
    "repair_edited": "Minimal repair where one edit is needed (chosen without looking at any response)",
    "repair_impossible": "Minimal repair where no allowed single edit works (chosen without looking at any response)",
    "repair_failure": "Minimal repair, first sample the focus arm gets wrong",
}
MAIN_VARIANTS = metrics.MAIN_VARIANTS
OVERLAY = "overlay-v1"
REFERENCE = "reference"
OP_COLOUR = {sem.ADD: (0, 114, 178), sem.REMOVE: (213, 94, 0)}   # blue / vermillion (colour-blind safe pair)
OP_WORD = {sem.ADD: "add", sem.REMOVE: "remove"}


def _walkthrough():
    spec = importlib.util.spec_from_file_location("walkthrough", ROOT / "demo" / "walkthrough.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wt = _walkthrough()


def order_key(sample_id: str) -> str:
    return hashlib.sha256((RULE + ":" + sample_id).encode()).hexdigest()


def parse_arm(text: str) -> Dict[str, Any]:
    if "=" not in text:
        raise SystemExit("arm must be LABEL=pred_dir[:flags.json[:extraction.jsonl]], got %r" % text)
    label, rest = text.split("=", 1)
    parts = rest.split(":")
    if len(parts) > 3 or not label:
        raise SystemExit("arm must be LABEL=pred_dir[:flags.json[:extraction.jsonl]], got %r" % text)
    return {"label": label, "pred_dir": Path(parts[0]), "flags": Path(parts[1]) if len(parts) > 1 and parts[1] else None,
            "extraction": Path(parts[2]) if len(parts) > 2 and parts[2] else None}


def load_arm(arm: Dict[str, Any], split: str) -> Dict[str, Any]:
    path = arm["pred_dir"] / ("predictions_%s.jsonl" % split)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    checked = False
    if arm["flags"] is not None:
        want = json.loads(arm["flags"].read_text())["predictions_sha256"]
        if want != sha:
            raise SystemExit("%s: %s is not the file recorded in %s" % (arm["label"], path, arm["flags"]))
        checked = True
    responses = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["sample_id"] in responses:
                raise SystemExit("%s: sample %s appears twice in %s" % (arm["label"], r["sample_id"], path))
            responses[r["sample_id"]] = r["text"]
    read = None
    if arm["extraction"] is not None:
        read = {}
        for line in arm["extraction"].read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                read[r["image_id"]] = r["text"]
    return {"label": arm["label"], "responses": responses, "predictions_sha256": sha, "sha_checked_against_flags": checked,
            "predictions": str(path), "read_graphs": read}


def read_graph_distance(sample: Dict[str, Any], text: Optional[str]) -> Optional[Dict[str, Any]]:
    """How far the graph a parser read from the image is from the true one (edges missed / invented)."""
    if text is None:
        return None
    parsed = tasks.parse_response(tasks.EXTRACTION, text)
    if not parsed.ok:
        return {"parsed": False}
    true = {tuple(e) for e in sample["private_truth"]["edges"]}
    got = {tuple(e) for e in parsed.value["edges"]}
    return {"parsed": True, "missed_edges": len(true - got), "invented_edges": len(got - true)}


def pool_of(samples: List[Dict[str, Any]], arms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [s for s in samples if s["metadata"]["image_variant"] in MAIN_VARIANTS
            and all(s["sample_id"] in a["responses"] for a in arms)]
    return sorted(rows, key=lambda s: order_key(s["sample_id"]))


def select(pool: List[Dict[str, Any]], focus: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Apply the slot rules in SLOTS order; a sample (and a twin shown with it) is used once."""
    used = set()

    def focus_correct(s):
        return bool(metrics.score_sample(s, focus["responses"][s["sample_id"]])["correct"])

    tests = {
        "success": lambda s: s["task"] == tasks.REACHABILITY and focus_correct(s),
        "failure": lambda s: s["task"] == tasks.REACHABILITY and not focus_correct(s),
        "critical_pair": lambda s: s["task"] == tasks.CRITICAL_EDGE and wt._pair_id(s, "base") is not None
        and wt.twin_of(pool, s) is not None,
        "repair_edited": lambda s: s["task"] == tasks.REPAIR and s["private_truth"]["status"] == "edited",
        "repair_impossible": lambda s: s["task"] == tasks.REPAIR and s["private_truth"]["status"] == "impossible",
        "repair_failure": lambda s: s["task"] == tasks.REPAIR and not focus_correct(s),
    }
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for slot in SLOTS:
        hit = None
        for s in pool:
            if s["sample_id"] in used or not tests[slot](s):
                continue
            if slot == "critical_pair" and wt.twin_of(pool, s)["sample_id"] in used:
                continue
            hit = s
            break
        out[slot] = hit
        if hit is not None:
            used.add(hit["sample_id"])
            if slot == "critical_pair":
                used.add(wt.twin_of(pool, hit)["sample_id"])
    return out


def why_not_drawable(image: Dict[str, Any], edit: Dict[str, str]) -> Optional[str]:
    """None if the edit can be marked on this image, else the reason it cannot."""
    if edit["src"] not in image["positions"] or edit["dst"] not in image["positions"]:
        return "names a node that is not in the image"
    if edit["src"] == edit["dst"]:
        return "a self loop"
    present = any(r["src"] == edit["src"] and r["dst"] == edit["dst"] for r in image["routes"])
    if edit["op"] == sem.REMOVE and not present:
        return "that edge is not in the image"
    if edit["op"] == sem.ADD and present:
        return "that edge is already in the image"
    return None


def named_edits(sample: Dict[str, Any], arms: List[Dict[str, Any]], image: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Distinct edits of the reference answer and of every strictly parsing response, in that order."""
    sources = [(REFERENCE, sample["training_target"]["edits"])]
    for a in arms:
        parsed = tasks.parse_response(tasks.REPAIR, a["responses"][sample["sample_id"]])
        sources.append((a["label"], parsed.value["edits"] if parsed.ok else []))
    out: List[Dict[str, Any]] = []
    for label, edits in sources:
        for e in edits:
            key = {"op": e["op"], "src": e["src"], "dst": e["dst"]}
            hit = next((o for o in out if all(o[k] == key[k] for k in key)), None)
            if hit is None:
                why = why_not_drawable(image, key)
                hit = {**key, "named_by": [], "not_drawn": why, "file": None}
                if why is None:
                    hit["file"] = "%s_%s_%s_%s.png" % (image["image_id"], OP_WORD[e["op"]], e["src"], e["dst"])
                out.append(hit)
            if label not in hit["named_by"]:
                hit["named_by"].append(label)
    return out


def draw_edit(img: Image.Image, image: Dict[str, Any], edit: Dict[str, str], supersample: int = 3) -> Image.Image:
    """A copy of the dataset image with one edit marked (see OVERLAY in the module docstring)."""
    if why_not_drawable(image, edit) is not None:
        raise ValueError("edit cannot be drawn: %s" % why_not_drawable(image, edit))
    style = styles.get_style(image["style_id"])
    s = supersample
    base_img = img.convert("RGB")
    big = Image.new("RGBA", (img.width * s, img.height * s), (0, 0, 0, 0))   # marker layer only: the image itself is not resampled
    d = ImageDraw.Draw(big)
    colour = OP_COLOUR[edit["op"]] + (255,)
    width = int(round((style.edge_width + 1.5) * s))

    def sc(points):
        return [(x * s, y * s) for x, y in points]

    if edit["op"] == sem.REMOVE:
        route = next([tuple(p) for p in r["points"]] for r in image["routes"]
                     if r["src"] == edit["src"] and r["dst"] == edit["dst"])
        d.line(sc(route), fill=colour, width=width, joint="curve")
        mx, my = geo.point_along(route, 0.5)
        arm = style.arrow_len * 0.8
        for dx, dy in ((1, 1), (1, -1)):
            d.line(sc([(mx - dx * arm, my - dy * arm), (mx + dx * arm, my + dy * arm)]), fill=colour, width=width)
    else:
        a, b = (tuple(float(v) for v in image["positions"][edit[k]]) for k in ("src", "dst"))
        ux, uy = geo.unit(b[0] - a[0], b[1] - a[1])
        gap = style.extent + 3.0
        start, tip = (a[0] + ux * gap, a[1] + uy * gap), (b[0] - ux * gap, b[1] - uy * gap)
        head, half = style.arrow_len * 1.2, style.arrow_half_width * 1.2
        base = (tip[0] - ux * head, tip[1] - uy * head)
        length = max(0.0, geo.dist(start, base))
        on, off, at = 10.0, 6.0, 0.0
        while at < length:
            end = min(at + on, length)
            d.line(sc([(start[0] + ux * at, start[1] + uy * at), (start[0] + ux * end, start[1] + uy * end)]),
                   fill=colour, width=width)
            at += on + off
        d.polygon(sc([tip, (base[0] - uy * half, base[1] + ux * half), (base[0] + uy * half, base[1] - ux * half)]),
                  fill=colour)
    out = base_img.convert("RGBA")
    out.alpha_composite(big.resize(img.size, Image.LANCZOS))
    return out.convert("RGB")


def card(sample: Dict[str, Any], arms: List[Dict[str, Any]], image: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rows = []
    for a in arms:
        text = a["responses"][sample["sample_id"]]
        row = {"arm": a["label"], "text": text, **wt.verdict(sample, text)}
        if a["read_graphs"] is not None:
            row["read_graph"] = read_graph_distance(sample, a["read_graphs"].get(sample["image_id"]))
        rows.append(row)
    out = {"sample_id": sample["sample_id"], "task": sample["task"], "image": sample["image_path"],
           "image_sha256": sample["image_sha256"], "instruction": sample["public_input"]["instruction"],
           "true_answer": tasks.format_target(sample["training_target"]), "rows": rows}
    if image is not None and sample["task"] == tasks.REPAIR:
        out["edits"] = named_edits(sample, arms, image)
    return out


def build(samples: List[Dict[str, Any]], arms: List[Dict[str, Any]], focus_label: str,
          images: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """`images` maps image_id -> image record (positions, routes); with it repair cards list the named edits."""
    labels = [a["label"] for a in arms]
    if len(set(labels)) != len(labels):
        raise SystemExit("arm labels must be distinct")
    if focus_label not in labels:
        raise SystemExit("--focus %s is not one of the arms (%s)" % (focus_label, ", ".join(labels)))
    pool = pool_of(samples, arms)
    if not pool:
        raise SystemExit("no sample has a recorded response from every arm")
    chosen = select(pool, arms[labels.index(focus_label)])
    slots = []
    for slot in SLOTS:
        s = chosen[slot]
        entry: Dict[str, Any] = {"slot": slot, "what": SLOT_TEXT[slot], "cards": []}
        if s is not None:
            entry["cards"].append(card(s, arms, None if images is None else images[s["image_id"]]))
            if slot == "critical_pair":
                entry["cards"].append(card(wt.twin_of(pool, s), arms))
        slots.append(entry)
    return {"rule": RULE, "overlay": None if images is None else OVERLAY, "focus": focus_label, "pool_samples": len(pool),
            "pool_root_graphs": len({s["graph_group_id"] for s in pool}),
            "arms": [{"label": a["label"], "predictions": a["predictions"], "predictions_sha256": a["predictions_sha256"],
                      "sha_checked_against_flags": a["sha_checked_against_flags"]} for a in arms],
            "slots": slots}


def _mark(row: Dict[str, Any]) -> str:
    if row["correct"]:
        mark = "correct"
    elif not row["format_ok"]:
        mark = "wrong (format failure: %s)" % str(row["error"]).replace("format:", "")
    else:
        mark = "wrong (%s)" % (row["error"] or ("status differs from the true one" if "status_correct" in row
                                                else "answer differs from the true graph's"))
    checks = [(name, row.get(key)) for name, key in (("legal", "legal"), ("goal met", "valid"), ("minimal", "minimal"))]
    if any(v is not None for _, v in checks):   # only an edit is checked; "unchanged" / "impossible" carry no edit
        mark += "; " + ", ".join("%s=%s" % (n, v) for n, v in checks if v is not None)
    rg = row.get("read_graph")
    if rg is not None:
        mark += ("; computed from the graph its parser read: %d true edges missed, %d invented"
                 % (rg["missed_edges"], rg["invented_edges"])) if rg["parsed"] else "; its parser's graph did not parse"
    return mark


def _cell(text: str) -> str:
    return "`" + " ".join(str(text).split()).replace("|", "\\|").replace("`", "'") + "`"


def _edit_lines(c: Dict[str, Any], image_prefix: Optional[str]) -> List[str]:
    if not c["edits"]:
        return ["No edit is named by the reference answer or by any response.", ""]
    out = ["Edits named above, marked on the image (blue dashed arrow = edge to add, orange with a cross = edge to remove;",
           "the colour says the operation, not whether the edit is right; an added arrow is a straight marker, not a",
           "laid-out edge). `%s` is the generator's stored answer; when several minimal edits exist it is one of them." % REFERENCE, ""]
    for e in c["edits"]:
        what = "%s `%s->%s`, named by %s" % (OP_WORD[e["op"]], e["src"], e["dst"], ", ".join(e["named_by"]))
        if e["not_drawn"] is not None:
            out += ["- %s: not drawn (%s)." % (what, e["not_drawn"]), ""]
        elif image_prefix is None:
            out += ["- %s." % what, ""]
        else:
            out += ["- %s:" % what, "", "  ![%s](%s/%s)" % (e["file"][:-4], image_prefix, e["file"]), ""]
    return out


def markdown(g: Dict[str, Any], image_prefix: Optional[str] = None) -> str:
    out = ["# Dev gallery: recorded responses, scored by the verifier", "",
           "Generated by `scripts/dev_gallery.py`; do not edit. Every response below is a recorded model output from the",
           "predictions file named in the table of sources; the verdict is `metrics.score_sample` against the generator's",
           "true graph. Rule `%s` (see the script's docstring): samples in a hash order that uses no score; only the slots" % g["rule"],
           "named \"first sample the focus arm answers ...\" look at an outcome, and only at arm %s's. Pool: %d samples on %d dev"
           % (g["focus"], g["pool_samples"], g["pool_root_graphs"]),
           "root graphs (those with a recorded response from every arm). This is an illustration, not a measurement: the",
           "accuracies are in `results_dev.md`. Style-change pairs are absent because no dev response was recorded for them.",
           "Arm E's answer is computed by the exact solver from the graph its parser read from the image, so it is only as",
           "right as that reading; the last column says how far the reading is from the true graph.", "",
           "| Arm | Predictions file | sha256 (first 16) | Same file as the reported flags |", "|---|---|---|---|"]
    for a in g["arms"]:
        out.append("| %s | `%s` | `%s` | %s |" % (a["label"], a["predictions"], a["predictions_sha256"][:16],
                                                 "yes" if a["sha_checked_against_flags"] else "not checked"))
    for entry in g["slots"]:
        out += ["", "## %s" % entry["what"], ""]
        if not entry["cards"]:
            out.append("No sample in the pool meets this rule.")
            continue
        for i, c in enumerate(entry["cards"]):
            if entry["slot"] == "critical_pair":
                out += ["**%s**" % ("Base image" if i == 0 else "Twin image (one edge different, same question)"), ""]
            if image_prefix is not None:
                out += ["![%s](%s/%s)" % (c["sample_id"], image_prefix, Path(c["image"]).name), ""]
            out += ["Sample `%s`, image `%s`." % (c["sample_id"], c["image"]), "", "```", c["instruction"], "```", "",
                    "True answer (from the generator's graph): %s" % _cell(c["true_answer"]), "",
                    "| Arm | Recorded response | Verifier |", "|---|---|---|"]
            out += ["| %s | %s | %s |" % (r["arm"], _cell(r["text"]), _mark(r)) for r in c["rows"]]
            out.append("")
            if "edits" in c:
                out += _edit_lines(c, image_prefix)
    return "\n".join(out).rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest"); ap.add_argument("split"); ap.add_argument("arms", nargs="+")
    ap.add_argument("--focus", required=True)
    ap.add_argument("--out"); ap.add_argument("--out-md"); ap.add_argument("--copy-images")
    args = ap.parse_args(argv)
    dataset.check_split_allowed(args.split, "eval")
    man, data_dir, by_split, _images = dataset.load_dataset(Path(args.manifest))
    arms = [load_arm(parse_arm(a), args.split) for a in args.arms]
    records = {r["image_id"]: r for r in _images[args.split]} if args.copy_images else None
    g = build(by_split[args.split], arms, args.focus, records)
    g["manifest"] = args.manifest
    g["split"] = args.split
    g["content_digest"] = man["content_digest"]
    prefix = None
    if args.copy_images:
        dest = Path(args.copy_images)
        dest.mkdir(parents=True, exist_ok=True)
        for entry in g["slots"]:
            for c in entry["cards"]:
                src = Path(data_dir) / c["image"]
                if hashlib.sha256(src.read_bytes()).hexdigest() != c["image_sha256"]:
                    raise SystemExit("image %s does not match the manifest" % src)
                shutil.copyfile(src, dest / src.name)
                for e in c.get("edits", []):
                    if e["file"] is not None:
                        with Image.open(src) as im:
                            draw_edit(im, records[Path(c["image"]).stem], e).save(dest / e["file"], format="PNG", optimize=True)
                        e["sha256"] = hashlib.sha256((dest / e["file"]).read_bytes()).hexdigest()
        prefix = dest.name
    if args.out:
        Path(args.out).write_text(json.dumps(g, indent=1, sort_keys=True) + "\n")
    text = markdown(g, prefix)
    if args.out_md:
        Path(args.out_md).write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
