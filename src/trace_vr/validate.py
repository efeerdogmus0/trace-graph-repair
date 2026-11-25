"""Dataset validator: re-derives everything from the files on disk.

Truth is recomputed with the independent implementations in `reference.py`, images are
re-opened and read back pixel by pixel, and leakage is checked across all graphs. A
dataset is only usable if this reports zero errors.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from . import dataset, layout as lay, reference as ref, render, shortcuts, tasks
from .graphs import Graph
from .iso import TopologyRegistry
from .styles import get_style

SAMPLE_KEYS = {
    "sample_id", "graph_group_id", "graph_family", "split", "task", "image_id", "image_path", "image_sha256",
    "public_input", "training_target", "private_truth", "pairs", "metadata",
}
PUBLIC_KEYS = {"instruction", "query", "max_edits"}


class Report:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.error_count = 0
        self.checks: Dict[str, int] = {}

    def ran(self, name: str, n: int = 1) -> None:
        self.checks[name] = self.checks.get(name, 0) + n

    def error(self, code: str, message: str) -> None:
        self.error_count += 1
        if len(self.errors) < 200:
            self.errors.append("%s: %s" % (code, message))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def expect(self, name: str, condition: bool, message: str) -> bool:
        self.ran(name)
        if not condition:
            self.error(name, message)
        return bool(condition)


def _pairs(xs) -> List[Tuple[str, str]]:
    return [(a, b) for a, b in xs]


def _graph_of(record: Dict[str, Any]) -> Graph:
    idx = {label: i for i, label in enumerate(record["nodes"])}
    return Graph(len(idx), [(idx[a], idx[b]) for a, b in record["edges"]])


def _weakly_connected(nodes: List[str], edges: List[List[str]]) -> bool:
    both = _pairs(edges) + [(b, a) for a, b in edges]
    reach = ref.closure_fw(nodes, both)
    return all(reach[(nodes[0], n)] for n in nodes[1:])


def _check_image(rep: Report, rec: Dict[str, Any], data_dir: Path, size: int, check_pixels: bool) -> None:
    name = rec["image_id"]
    nodes, edges = rec["nodes"], rec["edges"]
    rep.expect("graph.labels_unique", len(set(nodes)) == len(nodes), name)
    rep.expect("graph.labels_in_alphabet", all(n in tasks.LABEL_ALPHABET for n in nodes), name)
    eset = {tuple(e) for e in edges}
    rep.expect("graph.no_duplicate_edges", len(eset) == len(edges), name)
    rep.expect("graph.no_self_loops", all(a != b for a, b in eset), name)
    rep.expect("graph.no_antiparallel", all((b, a) not in eset for a, b in eset), name)
    rep.expect("graph.endpoints_known", all(a in nodes and b in nodes for a, b in eset), name)
    if rec["variant"] != "edit":
        rep.expect("graph.root_weakly_connected", _weakly_connected(nodes, edges), name)

    path = data_dir / rec["image_path"]
    if not rep.expect("image.exists", path.exists(), str(path)):
        return
    rep.expect("image.sha256", dataset.sha256_bytes(path.read_bytes()) == rec["image_sha256"], name)
    img = Image.open(path)
    rep.expect("image.size", img.size == (size, size), "%s is %s" % (name, img.size))

    # Geometry, re-checked from the stored coordinates.
    style = get_style(rec["style_id"])
    idx = {label: i for i, label in enumerate(nodes)}
    pos = [tuple(rec["positions"][label]) for label in nodes]
    routes = {(idx[r["src"]], idx[r["dst"]]): [tuple(p) for p in r["points"]] for r in rec["routes"]}
    rep.expect("geometry.routes_match_edges", set(routes) == {(idx[a], idx[b]) for a, b in eset}, name)
    rep.expect("geometry.node_spacing", lay.check_positions(pos, style, size), name)
    ok, crossings, reason = lay.check_routes(routes, style)
    rep.expect("geometry.readable", ok, "%s: %s" % (name, reason))
    rep.expect("geometry.crossings_recorded", (not ok) or crossings == rec["crossings"], name)
    clear2 = (style.extent + lay.EDGE_NODE_CLEAR - 0.05) ** 2
    from . import geometry as geo

    for (u, v), route in routes.items():
        hit = [nodes[k] for k in range(len(nodes)) if k not in (u, v) and geo.poly_point_dist2(route, pos[k]) < clear2]
        rep.expect("geometry.edge_clear_of_nodes", not hit, "%s: %s->%s passes %s" % (name, nodes[u], nodes[v], hit))
    if check_pixels:
        problems = render.image_checks(img, style, rec["positions"], rec["routes"])
        rep.expect("pixels.match_drawing", not problems, "%s: %s" % (name, "; ".join(problems[:3])))


def _check_variant(rep: Report, rec: Dict[str, Any], root: Optional[Dict[str, Any]]) -> None:
    name = rec["image_id"]
    if rec["variant"] == "root":
        return
    if not rep.expect("variant.has_root", root is not None, name):
        return
    r_edges, edges = {tuple(e) for e in root["edges"]}, {tuple(e) for e in rec["edges"]}
    if rec["variant"] == "edit":
        e = (rec["edit"]["src"], rec["edit"]["dst"])
        want = r_edges | {e} if rec["edit"]["op"] == "add" else r_edges - {e}
        rep.expect("variant.edit_is_one_edge", edges == want and edges != r_edges, name)
        rep.expect("variant.edit_same_drawing", rec["positions"] == root["positions"] and rec["style_id"] == root["style_id"], name)
        kept = {(r["src"], r["dst"]): r["points"] for r in root["routes"]}
        same = all(r["points"] == kept[(r["src"], r["dst"])] for r in rec["routes"] if (r["src"], r["dst"]) in kept)
        rep.expect("variant.edit_keeps_other_edges_in_place", same, name)
    elif rec["variant"] == "restyle":
        rep.expect("variant.restyle_same_graph", edges == r_edges and rec["nodes"] == root["nodes"], name)
        rep.expect("variant.restyle_new_style", rec["style_id"] != root["style_id"], name)
    elif rec["variant"] == "relabel":
        by_pos = {tuple(p): label for label, p in root["positions"].items()}
        mapping = {label: by_pos.get(tuple(p)) for label, p in rec["positions"].items()}
        mapped = {(mapping[a], mapping[b]) for a, b in edges}
        rep.expect("variant.relabel_same_topology", None not in mapping.values() and mapped == r_edges, name)
        rep.expect("variant.relabel_changes_labels", rec["nodes"] != root["nodes"], name)


def _answer_set(sample: Dict[str, Any]) -> set:
    return set(shortcuts._candidates(sample))


def _check_sample(rep: Report, s: Dict[str, Any], split: str, images: Dict[str, Dict[str, Any]], data_dir: Path) -> None:
    sid = s.get("sample_id", "?")
    if not rep.expect("sample.keys", set(s.keys()) == SAMPLE_KEYS, "%s has keys %s" % (sid, sorted(s.keys()))):
        return
    rep.expect("sample.split", s["split"] == split, sid)
    rep.expect("sample.task_known", s["task"] in tasks.ALL_TASKS, sid)
    rec = images.get(s["image_id"])
    if not rep.expect("sample.image_known", rec is not None, sid):
        return
    rep.expect("sample.image_same_group", rec["graph_group_id"] == s["graph_group_id"], sid)
    rep.expect("sample.image_hash", rec["image_sha256"] == s["image_sha256"] and rec["image_path"] == s["image_path"], sid)
    truth, pub = s["private_truth"], s["public_input"]
    nodes, edges = truth["nodes"], truth["edges"]
    rep.expect("sample.truth_is_image_graph", nodes == rec["nodes"] and edges == rec["edges"], sid)
    rep.expect("sample.public_keys", set(pub.keys()) <= PUBLIC_KEYS, sid)
    query = pub["query"]
    rep.expect("sample.query_type", query.get("type") == s["task"], sid)
    rep.expect("sample.instruction_matches_query", pub["instruction"] == tasks.render_instruction(query), sid)
    example = dataset.to_model_example(s, data_dir)
    rep.expect("sample.model_view_is_minimal", set(example.keys()) == {"image", "prompt"}, sid)

    task = s["task"]
    target = s["training_target"]
    e_pairs = _pairs(edges)
    if task == tasks.REPAIR:
        mr, mn, adds = _pairs(query["must_reach"]), _pairs(query["must_not_reach"]), _pairs(query["allowed_additions"])
        mentioned = {x for p in mr + mn + adds for x in p}
        rep.expect("repair.labels_known", mentioned <= set(nodes), sid)
        rep.expect("repair.max_edits", query["max_edits"] == 1 and pub.get("max_edits") == 1, sid)
        eset = set(e_pairs)
        rep.expect("repair.additions_addable", all(a != b and (a, b) not in eset and (b, a) not in eset for a, b in adds), sid)
        rep.expect("repair.goal_not_contradictory", not (set(mr) & set(mn)) and mr and mn, sid)
        status, sols, min_edits = ref.repair(nodes, e_pairs, mr, mn, adds)
        stored = sorted(sorted((e["op"], e["src"], e["dst"]) for e in sol) for sol in truth["all_minimal_solutions"])
        rep.expect("repair.status", status == truth["status"] == target["status"], "%s: reference says %s" % (sid, status))
        rep.expect("repair.all_minimal_solutions", sorted(sorted(x) for x in sols) == stored, sid)
        rep.expect("repair.minimum_edit_count", min_edits == truth["minimum_edit_count"], sid)
        if status == "edited":
            rep.expect("repair.target_is_minimal_solution", target["edits"] in truth["all_minimal_solutions"], sid)
        else:
            rep.expect("repair.target_has_no_edits", target["edits"] == [], sid)
        for sol in truth["all_minimal_solutions"]:
            resp = tasks.format_target({"status": status, "edits": sol})
            rep.expect("scorer.accepts_every_minimal_solution", tasks.score_response(query, nodes, edges, resp)["correct"], sid)
        wrong = '{"status":"unchanged","edits":[]}' if status == "impossible" else '{"status":"impossible","edits":[]}'
        rep.expect("scorer.rejects_wrong_status", not tasks.score_response(query, nodes, edges, wrong)["correct"], sid)
    else:
        if task != tasks.EXTRACTION:
            rep.expect("qa.labels_known", query["src"] in nodes and query["dst"] in nodes and query["src"] != query["dst"], sid)
        if task == tasks.REACHABILITY:
            expected: Any = {"answer": ref.reachable(nodes, e_pairs, query["src"], query["dst"])}
        elif task == tasks.CRITICAL_EDGE:
            edge = tuple(query["edge"])
            posed = edge in set(e_pairs) and ref.reachable(nodes, e_pairs, query["src"], query["dst"])
            rep.expect("critical.well_posed", posed, sid)
            expected = {"answer": ref.critical(nodes, e_pairs, query["src"], query["dst"], edge)} if posed else None
        elif task == tasks.EDGE_EXISTS:
            expected = {"answer": (query["src"], query["dst"]) in set(e_pairs)}
        elif task == tasks.SHORTEST_PATH:
            expected = {"answer": ref.shortest_len(nodes, e_pairs, query["src"], query["dst"])}
        else:
            expected = {"nodes": sorted(nodes), "edges": sorted([a, b] for a, b in e_pairs)}
        rep.expect("qa.answer", expected == target == truth.get("answer"), "%s: reference says %s" % (sid, expected))
        if task in tasks.BOOLEAN_TASKS and expected is not None:
            flipped = tasks.format_target({"answer": not expected["answer"]})
            rep.expect("scorer.rejects_flipped_answer", not tasks.score_response(query, nodes, edges, flipped)["correct"], sid)
    rep.expect(
        "scorer.accepts_training_target",
        tasks.score_response(query, nodes, edges, tasks.format_target(target))["correct"],
        sid,
    )
    oracle = tasks.format_target(tasks.solve_query(query, nodes, edges))
    rep.expect("solver.true_graph_gives_correct_answer", tasks.score_response(query, nodes, edges, oracle)["correct"], sid)


def _check_pairs(rep: Report, samples: List[Dict[str, Any]], images: Dict[str, Dict[str, Any]]) -> None:
    groups: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for s in samples:
        for p in s["pairs"]:
            groups.setdefault(p["pair_id"], []).append((s, p))
    for pid, members in groups.items():
        if not rep.expect("pair.two_members", len(members) == 2, pid):
            continue
        (a, pa), (b, pb) = members
        if pa["role"] != "base":
            (a, pa), (b, pb) = (b, pb), (a, pa)
        rep.expect("pair.roles", pa["role"] == "base" and pb["role"] == "variant" and pa["kind"] == pb["kind"], pid)
        rep.expect("pair.same_group_split_task",
                   a["graph_group_id"] == b["graph_group_id"] and a["split"] == b["split"] and a["task"] == b["task"], pid)
        rep.expect("pair.different_images", a["image_id"] != b["image_id"], pid)
        kind = pa["kind"]
        if kind == "critical":
            rep.expect("pair.critical_same_instruction", a["public_input"]["instruction"] == b["public_input"]["instruction"], pid)
            rep.expect("pair.critical_answers_disjoint", not (_answer_set(a) & _answer_set(b)), pid)
            rep.expect("pair.critical_variant_is_edit", images[b["image_id"]]["variant"] == "edit", pid)
        else:
            want = "restyle" if kind == "invariant_style" else "relabel"
            rep.expect("pair.invariant_variant_kind", images[b["image_id"]]["variant"] == want, pid)
            rep.expect("pair.invariant_same_answer", a["training_target"] == b["training_target"], pid)
            if kind == "invariant_style":
                rep.expect("pair.style_same_instruction", a["public_input"]["instruction"] == b["public_input"]["instruction"], pid)


def validate(manifest_path: Path, check_pixels: bool = True) -> Dict[str, Any]:
    rep = Report()
    manifest, data_dir, samples_by_split, images_by_split = dataset.load_dataset(Path(manifest_path))
    size = int(manifest["config"]["image_size"])

    content_digests: Dict[str, str] = {}
    image_digests: Dict[str, str] = {}
    all_images: Dict[str, Dict[str, Any]] = {}
    group_split: Dict[str, str] = {}
    sample_ids: Dict[str, str] = {}
    sha_owner: Dict[str, str] = {}
    registry = TopologyRegistry()
    shortcut_report: Dict[str, Any] = {}
    recorded = manifest.get("excluded_manifests", {})
    rep.expect("leakage.exclusions_recorded",
               sorted(recorded) == sorted(manifest["config"].get("exclude_isomorphic_to", [])), str(sorted(recorded)))
    found_digests, excluded_graphs = dataset.load_exclusions(manifest["config"], Path(manifest_path).parent)
    for rel, digest in found_digests.items():
        rep.expect("leakage.excluded_manifest_unchanged", recorded.get(rel) == digest, rel)
    for g, gid in excluded_graphs:
        registry.add(g, gid)

    for split, info in manifest["splits"].items():
        content_hash, image_hash = hashlib.sha256(), hashlib.sha256()
        samples, images = samples_by_split[split], images_by_split[split]
        rep.expect("files.samples_hash", dataset.sha256_bytes((data_dir / info["samples_file"]).read_bytes()) == info["samples_sha256"], split)
        rep.expect("files.images_index_hash", dataset.sha256_bytes((data_dir / info["images_file"]).read_bytes()) == info["images_file_sha256"], split)
        rep.expect("files.counts", len(samples) == info["samples"] and len(images) == info["images"], split)
        roots = {r["graph_group_id"]: r for r in images if r["variant"] == "root"}
        rep.expect("files.root_count", len(roots) == info["roots"], "%s has %d roots" % (split, len(roots)))

        for rec in images:
            rep.expect("image.id_unique", rec["image_id"] not in all_images, rec["image_id"])
            all_images[rec["image_id"]] = rec
            rep.expect("image.split", rec["split"] == split, rec["image_id"])
            gid = rec["graph_group_id"]
            rep.expect("leakage.group_in_one_split", group_split.setdefault(gid, split) == split, gid)
            other = sha_owner.setdefault(rec["image_sha256"], gid)
            rep.expect("leakage.no_identical_image_across_groups", other == gid, "%s and %s" % (gid, other))
            _check_image(rep, rec, data_dir, size, check_pixels)
            _check_variant(rep, rec, roots.get(gid))
            g = _graph_of(rec)
            clash = registry.find_other_group(g, gid)
            rep.expect("leakage.no_isomorphic_graph_in_other_group", clash is None, "%s vs %s" % (gid, clash))
            registry.add(g, gid)

        used = set()
        for s in samples:
            rep.expect("sample.id_unique", s.get("sample_id") not in sample_ids, str(s.get("sample_id")))
            sample_ids[s.get("sample_id")] = split
            _check_sample(rep, s, split, all_images, data_dir)
            used.add(s.get("image_id"))
            row = dict(s)
            row["image_sha256"] = None
            content_hash.update(dataset.canonical_json(row).encode("ascii"))
            image_hash.update(str(s.get("image_sha256")).encode("ascii"))
        rep.expect("files.every_image_used", used == {r["image_id"] for r in images}, split)
        _check_pairs(rep, samples, all_images)
        content_digests[split], image_digests[split] = content_hash.hexdigest(), image_hash.hexdigest()
        shortcut_report[split] = shortcuts.report(samples)
        _distribution_warnings(rep, split, samples)

    rep.expect("manifest.content_digest", dataset.combine_digests(content_digests) == manifest["content_digest"], "content digest differs")
    rep.expect("manifest.images_digest", dataset.combine_digests(image_digests) == manifest["images_digest"], "images digest differs")
    return {
        "manifest": str(manifest_path),
        "ok": rep.error_count == 0,
        "error_count": rep.error_count,
        "errors": rep.errors,
        "warnings": rep.warnings,
        "checks_run": dict(sorted(rep.checks.items())),
        "total_checks": sum(rep.checks.values()),
        "text_only_shortcuts": shortcut_report,
        "stats": {split: dataset.summarize(samples_by_split[split], images_by_split[split]) for split in samples_by_split},
    }


def _distribution_warnings(rep: Report, split: str, samples: List[Dict[str, Any]]) -> None:
    for task in tasks.MAIN_NON_EDIT_TASKS:
        rows = [s for s in samples if s["task"] == task]
        if len(rows) >= 40:
            share = sum(1 for s in rows if s["training_target"]["answer"]) / len(rows)
            if not 0.4 <= share <= 0.6:
                rep.warn("%s/%s: %.0f%% of answers are true" % (split, task, share * 100))
        unpaired = [s for s in rows if not any(p["kind"] == "critical" for p in s["pairs"])
                    and s["metadata"]["image_variant"] in ("root", "edit")]
        if rows and len(unpaired) > 0.1 * len(rows):
            rep.warn("%s/%s: %d of %d samples have no answer-flipping twin" % (split, task, len(unpaired), len(rows)))
    repair = [s for s in samples if s["task"] == tasks.REPAIR]
    if len(repair) >= 40:
        for cat in ("add", "remove", "unchanged", "impossible"):
            share = sum(1 for s in repair if s["metadata"]["task_info"]["category"] == cat) / len(repair)
            if share < 0.15:
                rep.warn("%s/repair: category '%s' is only %.0f%%" % (split, cat, share * 100))
