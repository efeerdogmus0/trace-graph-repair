"""scripts/dev_gallery.py: the fixed gallery of recorded responses.

The selection must follow the written rule (hash order, slot conditions, one use per sample), every
verdict must be the scorer's, a predictions file that is not the one the flags were computed from
must be refused, and locked test splits must be refused."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from trace_vr import bootstrap, metrics, tasks
from test_dataset import built  # noqa: F401  (module-scoped fixture)

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "train"  # the tiny build has no dev split; "train" is not locked


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gallery = _load("dev_gallery", ROOT / "scripts" / "dev_gallery.py")


def _main_rows(built):
    return [s for s in built["samples"][SPLIT] if s["metadata"]["image_variant"] in metrics.MAIN_VARIANTS]


def _exact(s):
    return tasks.format_target(s["training_target"])


def _always_true(s):
    return '{"answer":true}' if s["task"] != tasks.REPAIR else '{"status":"unchanged","edits":[]}'


def _write_arm(tmp_path, built, label, fn, rows=None, with_flags=True):
    rows = _main_rows(built) if rows is None else rows
    d = tmp_path / label
    d.mkdir()
    text = "".join(json.dumps({"sample_id": s["sample_id"], "task": s["task"], "text": fn(s)}) + "\n" for s in rows)
    (d / ("predictions_%s.jsonl" % SPLIT)).write_text(text)
    spec = "%s=%s" % (label, d)
    if with_flags:
        flags = bootstrap.score_flags(rows, {s["sample_id"]: fn(s) for s in rows})
        flags["predictions_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        (d / "flags.json").write_text(json.dumps(flags))
        spec += ":%s" % (d / "flags.json")
    return spec


def _build(tmp_path, built, specs, focus):
    arms = [gallery.load_arm(gallery.parse_arm(a), SPLIT) for a in specs]
    return gallery.build(built["samples"][SPLIT], arms, focus), arms


def test_selection_follows_the_written_rule(built, tmp_path):
    specs = [_write_arm(tmp_path, built, "ORACLE", _exact), _write_arm(tmp_path, built, "TRUE", _always_true)]
    g, arms = _build(tmp_path, built, specs, "TRUE")
    by_id = {s["sample_id"]: s for s in built["samples"][SPLIT]}
    pool = sorted(_main_rows(built), key=lambda s: hashlib.sha256(("gallery-v1:" + s["sample_id"]).encode()).hexdigest())
    assert g["pool_samples"] == len(pool)
    slots = {e["slot"]: e for e in g["slots"]}
    assert list(slots) == list(gallery.SLOTS)

    def first(cond, skip=()):
        return next(s["sample_id"] for s in pool if s["sample_id"] not in skip and cond(s))

    truth = lambda s: s["training_target"].get("answer")  # noqa: E731
    success = first(lambda s: s["task"] == tasks.REACHABILITY and truth(s) is True)
    failure = first(lambda s: s["task"] == tasks.REACHABILITY and truth(s) is False)
    assert slots["success"]["cards"][0]["sample_id"] == success
    assert slots["failure"]["cards"][0]["sample_id"] == failure
    pair = slots["critical_pair"]["cards"]
    assert len(pair) == 2 and all(c["task"] == tasks.CRITICAL_EDGE for c in pair)
    a, b = (by_id[c["sample_id"]] for c in pair)
    assert {p["pair_id"] for p in a["pairs"] if p["kind"] == "critical" and p["role"] == "base"} \
        & {p["pair_id"] for p in b["pairs"] if p["kind"] == "critical" and p["role"] == "variant"}
    assert a["training_target"] != b["training_target"] and a["public_input"]["instruction"] == b["public_input"]["instruction"]
    used = {success, failure, a["sample_id"], b["sample_id"]}
    edited = first(lambda s: s["task"] == tasks.REPAIR and s["private_truth"]["status"] == "edited", used)
    assert slots["repair_edited"]["cards"][0]["sample_id"] == edited
    impossible = slots["repair_impossible"]["cards"][0]["sample_id"]
    assert by_id[impossible]["private_truth"]["status"] == "impossible"
    wrong = slots["repair_failure"]["cards"]
    assert wrong and by_id[wrong[0]["sample_id"]]["private_truth"]["status"] != "unchanged"
    ids = [c["sample_id"] for e in g["slots"] for c in e["cards"]]
    assert len(ids) == len(set(ids))  # a sample is shown once


def test_verdicts_and_texts_are_the_recorded_ones(built, tmp_path):
    specs = [_write_arm(tmp_path, built, "ORACLE", _exact), _write_arm(tmp_path, built, "TRUE", _always_true)]
    g, arms = _build(tmp_path, built, specs, "TRUE")
    by_id = {s["sample_id"]: s for s in built["samples"][SPLIT]}
    fns = {"ORACLE": _exact, "TRUE": _always_true}
    for e in g["slots"]:
        for c in e["cards"]:
            s = by_id[c["sample_id"]]
            assert c["true_answer"] == _exact(s) and c["instruction"] == s["public_input"]["instruction"]
            assert [r["arm"] for r in c["rows"]] == ["ORACLE", "TRUE"]
            for r in c["rows"]:
                assert r["text"] == fns[r["arm"]](s)
                assert r["correct"] == bool(metrics.score_sample(s, r["text"])["correct"])
            assert c["rows"][0]["correct"] is True
    assert all(a["sha_checked_against_flags"] for a in g["arms"])
    md = gallery.markdown(g, "pics")
    assert "private_truth" not in md and "recorded model output" in md and "](pics/" in md
    assert md.count("| ORACLE |") == 1 + sum(len(e["cards"]) for e in g["slots"])


def test_empty_slot_is_reported_not_refilled(built, tmp_path):
    g, _ = _build(tmp_path, built, [_write_arm(tmp_path, built, "ORACLE", _exact)], "ORACLE")
    slots = {e["slot"]: e for e in g["slots"]}
    assert slots["failure"]["cards"] == [] and slots["repair_failure"]["cards"] == []
    assert slots["success"]["cards"] and slots["critical_pair"]["cards"]
    assert gallery.markdown(g).count("No sample in the pool meets this rule.") == 2


def test_pool_is_the_samples_every_arm_answered(built, tmp_path):
    rows = _main_rows(built)
    groups = sorted({s["graph_group_id"] for s in rows})[:3]
    small = [s for s in rows if s["graph_group_id"] in groups]
    specs = [_write_arm(tmp_path, built, "ORACLE", _exact), _write_arm(tmp_path, built, "BASE", _always_true, rows=small)]
    g, _ = _build(tmp_path, built, specs, "ORACLE")
    assert g["pool_samples"] == len(small) and g["pool_root_graphs"] == 3
    shown = {c["sample_id"] for e in g["slots"] for c in e["cards"]}
    assert shown <= {s["sample_id"] for s in small}


def test_a_different_predictions_file_is_refused(built, tmp_path):
    spec = _write_arm(tmp_path, built, "ORACLE", _exact)
    pred = tmp_path / "ORACLE" / ("predictions_%s.jsonl" % SPLIT)
    pred.write_text(pred.read_text().replace("true", "false", 1))
    with pytest.raises(SystemExit, match="not the file recorded"):
        gallery.load_arm(gallery.parse_arm(spec), SPLIT)


def test_read_graph_distance_counts_missed_and_invented_edges(built):
    s = next(r for r in _main_rows(built) if len(r["private_truth"]["edges"]) >= 2)
    nodes, edges = s["private_truth"]["nodes"], [list(e) for e in s["private_truth"]["edges"]]
    absent = next([a, b] for a in nodes for b in nodes if a != b and [a, b] not in edges)
    text = json.dumps({"nodes": nodes, "edges": edges[1:] + [absent]})
    assert gallery.read_graph_distance(s, text) == {"parsed": True, "missed_edges": 1, "invented_edges": 1}
    assert gallery.read_graph_distance(s, "the graph has some edges") == {"parsed": False}
    assert gallery.read_graph_distance(s, None) is None


def test_cli_writes_files_and_refuses_locked_splits(built, tmp_path):
    spec = _write_arm(tmp_path, built, "ORACLE", _exact)
    out, md, pics = tmp_path / "g.json", tmp_path / "g.md", tmp_path / "pics"
    gallery.main([str(built["manifest_path"]), SPLIT, spec, "--focus", "ORACLE", "--out", str(out), "--out-md", str(md),
                  "--copy-images", str(pics)])
    g = json.loads(out.read_text())
    names = {Path(c["image"]).name for e in g["slots"] for c in e["cards"]}
    drawn = [x for e in g["slots"] for c in e["cards"] for x in c.get("edits", []) if x["file"] is not None]
    assert names and drawn and g["overlay"] == gallery.OVERLAY
    assert {p.name for p in pics.iterdir()} == names | {x["file"] for x in drawn}
    assert all(hashlib.sha256((pics / x["file"]).read_bytes()).hexdigest() == x["sha256"] for x in drawn)
    assert g["content_digest"] == built["manifest"]["content_digest"]
    with pytest.raises(SystemExit):
        gallery.main([str(built["manifest_path"]), "test_id", spec, "--focus", "ORACLE"])


def test_tracked_gallery_page_is_the_generated_one():
    record = ROOT / "reports" / "pilot" / "gallery_dev.json"
    g = json.loads(record.read_text())
    assert (ROOT / "reports" / "pilot" / "gallery_dev.md").read_text() == gallery.markdown(g, "gallery_dev")
    assert g["rule"] == gallery.RULE and [e["slot"] for e in g["slots"]] == list(gallery.SLOTS)
    for e in g["slots"]:
        for c in e["cards"]:
            png = ROOT / "reports" / "pilot" / "gallery_dev" / Path(c["image"]).name
            assert hashlib.sha256(png.read_bytes()).hexdigest() == c["image_sha256"]
    assert all(a["sha_checked_against_flags"] for a in g["arms"])
    files = {Path(c["image"]).name for e in g["slots"] for c in e["cards"]}
    for e in g["slots"]:
        for c in e["cards"]:
            assert ("edits" in c) == (c["task"] == tasks.REPAIR)
            for x in c.get("edits", []):
                assert (x["file"] is None) == (x["not_drawn"] is not None)
                if x["file"] is not None:
                    png = ROOT / "reports" / "pilot" / "gallery_dev" / x["file"]
                    assert hashlib.sha256(png.read_bytes()).hexdigest() == x["sha256"]
                    files.add(x["file"])
    assert g["overlay"] == gallery.OVERLAY
    assert {p.name for p in (ROOT / "reports" / "pilot" / "gallery_dev").iterdir()} == files  # no stale card or overlay


def _repair_with_image(built, status="edited"):
    images = {r["image_id"]: r for r in built["images"][SPLIT]}
    s = next(r for r in _main_rows(built) if r["task"] == tasks.REPAIR and r["private_truth"]["status"] == status)
    return s, images[s["image_id"]]


def _absent(image):
    present = {(r["src"], r["dst"]) for r in image["routes"]}
    return next((a, b) for a in image["nodes"] for b in image["nodes"] if a != b and (a, b) not in present)


def test_named_edits_are_the_reference_and_the_parsed_responses(built):
    s, image = _repair_with_image(built)
    ref = s["training_target"]["edits"][0]
    a, b = _absent(image)
    r = image["routes"][0]
    responses = {
        "SAME": tasks.format_target(s["training_target"]),
        "ABSENT": json.dumps({"status": "edited", "edits": [{"op": "remove_edge", "src": a, "dst": b}]}),
        "DOUBLE": json.dumps({"status": "edited", "edits": [{"op": "add_edge", "src": r["src"], "dst": r["dst"]}]}),
        "GHOST": json.dumps({"status": "edited", "edits": [{"op": "add_edge", "src": "Q", "dst": a}]}),
        "PROSE": "remove the edge %s->%s" % (a, b),
        "NONE": '{"status":"impossible","edits":[]}',
    }
    arms = [{"label": k, "responses": {s["sample_id"]: v}} for k, v in responses.items()]
    edits = gallery.named_edits(s, arms, image)
    assert [(e["op"], e["src"], e["dst"]) for e in edits] == [
        (ref["op"], ref["src"], ref["dst"]), ("remove_edge", a, b), ("add_edge", r["src"], r["dst"]), ("add_edge", "Q", a)]
    assert edits[0]["named_by"] == [gallery.REFERENCE, "SAME"] and edits[0]["not_drawn"] is None
    assert edits[0]["file"].startswith(image["image_id"] + "_") and edits[0]["file"].endswith(".png")
    assert [e["not_drawn"] for e in edits[1:]] == ["that edge is not in the image", "that edge is already in the image",
                                                    "names a node that is not in the image"]
    assert all(e["file"] is None for e in edits[1:])
    assert not any(lab in e["named_by"] for e in edits for lab in ("PROSE", "NONE"))


def _far_pixels_unchanged(before, after, line, margin):
    from trace_vr import geometry as geo
    a, b = before.load(), after.load()
    changed = [(x, y) for x in range(before.width) for y in range(before.height) if a[x, y] != b[x, y]]
    assert changed
    return all(geo.poly_point_dist2(line, p) <= margin ** 2 for p in changed)


def _close(c, want, tol=40):
    return sum(abs(c[i] - want[i]) for i in range(3)) <= tol


def test_overlay_marks_the_named_edge_and_leaves_the_rest_of_the_image_alone(built):
    from PIL import Image
    from trace_vr import geometry as geo, semantics as sem, styles
    s, image = _repair_with_image(built)
    before = Image.open(Path(built["data_dir"]) / s["image_path"]).convert("RGB")

    r = image["routes"][0]
    route = [tuple(p) for p in r["points"]]
    after = gallery.draw_edit(before, image, {"op": sem.REMOVE, "src": r["src"], "dst": r["dst"]})
    assert after.size == before.size
    for frac in (0.25, 0.5, 0.75):   # the edge's own route is repainted, not a straight line between the nodes
        x, y = geo.point_along(route, frac)
        assert _close(after.getpixel((int(round(x)), int(round(y)))), gallery.OP_COLOUR[sem.REMOVE])
    st = styles.get_style(image["style_id"])
    cross_reach = st.arrow_len * 0.8 * 2 ** 0.5 + st.edge_width + 1.5 + 2   # cross arm along the diagonal + stroke + resampling
    assert _far_pixels_unchanged(before, after, route, cross_reach)

    a, b = _absent(image)
    pa, pb = (tuple(float(v) for v in image["positions"][k]) for k in (a, b))
    after = gallery.draw_edit(before, image, {"op": sem.ADD, "src": a, "dst": b})
    ux, uy = geo.unit(pb[0] - pa[0], pb[1] - pa[1])
    head = (pb[0] - ux * (st.extent + 3 + st.arrow_len * 0.6), pb[1] - uy * (st.extent + 3 + st.arrow_len * 0.6))
    tail = (pa[0] + ux * (st.extent + 3 + st.arrow_len * 0.6), pa[1] + uy * (st.extent + 3 + st.arrow_len * 0.6))
    assert _close(after.getpixel((int(round(head[0])), int(round(head[1])))), gallery.OP_COLOUR[sem.ADD])  # arrowhead at dst
    assert _far_pixels_unchanged(before, after, [pa, pb], 12)
    # direction: the arrowhead is wide near the target only
    def blue_across(p):
        return sum(_close(after.getpixel((int(round(p[0] - uy * k)), int(round(p[1] + ux * k)))), gallery.OP_COLOUR[sem.ADD], 90)
                   for k in range(-8, 9))
    assert blue_across(head) > blue_across(tail)
    assert before.tobytes() == Image.open(Path(built["data_dir"]) / s["image_path"]).convert("RGB").tobytes()

    with pytest.raises(ValueError):
        gallery.draw_edit(before, image, {"op": sem.REMOVE, "src": a, "dst": b})
    with pytest.raises(ValueError):
        gallery.draw_edit(before, image, {"op": sem.ADD, "src": r["src"], "dst": r["dst"]})


def test_cards_without_image_records_carry_no_edit_list(built, tmp_path):
    g, _ = _build(tmp_path, built, [_write_arm(tmp_path, built, "ORACLE", _exact)], "ORACLE")
    assert g["overlay"] is None and not any("edits" in c for e in g["slots"] for c in e["cards"])
    arms = [gallery.load_arm(gallery.parse_arm(_write_arm(tmp_path, built, "ORACLE2", _exact)), SPLIT)]
    images = {r["image_id"]: r for r in built["images"][SPLIT]}
    g2 = gallery.build(built["samples"][SPLIT], arms, "ORACLE2", images)
    assert [[c["sample_id"] for c in e["cards"]] for e in g2["slots"]] == [[c["sample_id"] for c in e["cards"]] for e in g["slots"]]
    md = gallery.markdown(g2, "pics")
    assert "named by reference, ORACLE2" in md and "not whether the edit is right" in md
