"""Risks 4, 5 and 8 end to end: a small dataset is built, re-validated, rebuilt and deliberately broken."""
import copy
import json
import shutil
from pathlib import Path

import networkx as nx
import pytest
from PIL import Image

from trace_vr import dataset, metrics, shortcuts, tasks, validate

ID = {"families": {"layered_dag": 1, "tree_plus": 1, "sparse_random": 1}, "nodes": [5, 8], "styles": ["a", "b", "c"]}


def tiny_config():
    cfg = copy.deepcopy(dataset.DEFAULTS)
    cfg.update({
        "name": "tiny", "seed": 7, "invariant_rate": 0.5,
        "splits": {
            "train": dict(ID, roots=8),
            "test_id": dict(ID, roots=4),
            "test_style_ood": dict(ID, roots=3, styles=["x", "y"]),
            "test_struct_ood": {"roots": 3, "families": {"ring_chords": 1, "two_cluster": 1}, "nodes": [11, 13], "styles": ["a", "b", "c"]},
        },
    })
    return cfg


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    manifest_path = root / "manifests" / "tiny.json"
    manifest = dataset.build_dataset(tiny_config(), root / "data", manifest_path, log=lambda *_: None)
    _, data_dir, samples, images = dataset.load_dataset(manifest_path)
    return {"root": root, "manifest_path": manifest_path, "manifest": manifest, "data_dir": data_dir,
            "samples": samples, "images": images}


def all_samples(built):
    return [s for rows in built["samples"].values() for s in rows]


def test_validator_passes_on_a_fresh_build(built):
    report = validate.validate(built["manifest_path"])
    assert report["ok"], report["errors"][:5]
    assert report["total_checks"] > 3000


def test_rebuilding_gives_identical_content_and_images(built, tmp_path):
    again = dataset.build_dataset(tiny_config(), tmp_path / "data", tmp_path / "manifests" / "tiny.json", log=lambda *_: None)
    first = built["manifest"]
    assert again["content_digest"] == first["content_digest"]
    assert again["images_digest"] == first["images_digest"]
    for split in first["splits"]:
        assert again["splits"][split]["samples_sha256"] == first["splits"][split]["samples_sha256"]


def test_no_topology_is_shared_between_groups_checked_with_networkx(built):
    graphs = []
    split_of = {}
    for split, images in built["images"].items():
        for rec in images:
            assert split_of.setdefault(rec["graph_group_id"], split) == split
            g = nx.DiGraph()
            g.add_nodes_from(rec["nodes"])
            g.add_edges_from(tuple(e) for e in rec["edges"])
            graphs.append((rec["graph_group_id"], g))
    for i, (ga, a) in enumerate(graphs):
        for gb, b in graphs[i + 1:]:
            if ga != gb and a.number_of_nodes() == b.number_of_nodes() and a.number_of_edges() == b.number_of_edges():
                assert not nx.is_isomorphic(a, b), (ga, gb)


def test_model_view_contains_nothing_private(built):
    sample = copy.deepcopy(next(s for s in all_samples(built) if s["task"] == tasks.REPAIR))
    secret = "SECRET-TRUTH-MARKER"
    sample["private_truth"]["nodes"] = [secret]
    sample["private_truth"]["all_minimal_solutions"] = [[{"op": secret, "src": secret, "dst": secret}]]
    sample["metadata"]["task_info"]["category"] = secret
    sample["graph_group_id"] = secret
    sample["sample_id"] = secret
    view = dataset.to_model_example(sample, built["data_dir"])
    assert set(view) == {"image", "prompt"}
    assert secret not in json.dumps(view)
    with_target = dataset.to_model_example(sample, built["data_dir"], with_target=True)
    assert set(with_target) == {"image", "prompt", "target"}
    assert json.loads(with_target["target"]) == sample["training_target"]


def test_twins_share_the_instruction_and_never_the_answer(built):
    by_pair = {}
    for s in all_samples(built):
        for p in s["pairs"]:
            if p["kind"] == "critical":
                by_pair.setdefault(p["pair_id"], []).append(s)
    assert len(by_pair) >= 3 * 18  # every root graph has a pair for each of the three main tasks
    for a, b in by_pair.values():
        assert a["public_input"]["instruction"] == b["public_input"]["instruction"]
        assert a["image_id"] != b["image_id"]
        assert not (set(shortcuts._candidates(a)) & set(shortcuts._candidates(b)))


def test_no_text_only_predictor_beats_one_half_on_main_tasks(built):
    for split, rows in built["samples"].items():
        rep = shortcuts.report(rows)
        for task in (tasks.REPAIR, tasks.REACHABILITY, tasks.CRITICAL_EDGE):
            assert rep[task]["text_only_ceiling"] == 0.5, (split, task)


def test_metrics_with_oracle_constant_and_garbage_responders(built):
    rows = built["samples"]["train"]

    def answers(fn):
        return {s["sample_id"]: fn(s) for s in rows}

    oracle = metrics.evaluate(rows, answers(lambda s: tasks.format_target(
        tasks.solve_query(s["public_input"]["query"], s["private_truth"]["nodes"], s["private_truth"]["edges"]))))
    assert all(v["accuracy"] == 1.0 and v["format_failure_rate"] == 0.0 for v in oracle["per_task"].values())
    assert oracle["non_edit_macro_accuracy"] == 1.0
    assert all(v["both_correct"] == 1.0 for v in oracle["pair_accuracy"].values())

    always_true = metrics.evaluate(rows, answers(lambda s: '{"answer": true}'))
    assert always_true["per_task"][tasks.REACHABILITY]["accuracy"] == 0.5
    assert always_true["pair_accuracy"]["critical:reachability"]["both_correct"] == 0.0  # no credit for consistency
    assert always_true["per_task"][tasks.REPAIR]["format_failure_rate"] == 1.0

    garbage = metrics.evaluate(rows, {})
    assert garbage["missing_responses"] == len(rows)
    assert all(v["accuracy"] == 0.0 and v["format_failure_rate"] == 1.0 for v in garbage["per_task"].values())


def test_metrics_on_a_hand_checked_prediction_file():
    def sample(sid, edges, pair_role):
        return {"sample_id": sid, "task": tasks.REACHABILITY, "metadata": {"image_variant": "root" if pair_role == "base" else "edit"},
                "public_input": {"query": {"type": tasks.REACHABILITY, "src": "A", "dst": "C"}},
                "private_truth": {"nodes": ["A", "B", "C"], "edges": edges},
                "pairs": [{"pair_id": "p1", "kind": "critical", "role": pair_role}]}

    rows = [sample("s1", [["A", "B"], ["B", "C"]], "base"), sample("s2", [["A", "B"]], "variant")]
    rows.append({"sample_id": "s3", "task": tasks.CRITICAL_EDGE, "metadata": {"image_variant": "root"}, "pairs": [],
                 "public_input": {"query": {"type": tasks.CRITICAL_EDGE, "src": "A", "dst": "C", "edge": ["A", "B"]}},
                 "private_truth": {"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "C"]]}})
    # s1 right, s2 wrong (says true, truth is false), s3 unparseable.
    out = metrics.evaluate(rows, {"s1": '{"answer": true}', "s2": '{"answer": true}', "s3": "yes, it is critical"})
    assert out["per_task"][tasks.REACHABILITY] == {"n": 2, "accuracy": 0.5, "format_failure_rate": 0.0}
    assert out["per_task"][tasks.CRITICAL_EDGE] == {"n": 1, "accuracy": 0.0, "format_failure_rate": 1.0}
    assert out["non_edit_macro_accuracy"] == 0.25
    assert out["pair_accuracy"] == {"critical:reachability": {"n": 1, "both_correct": 0.0}}


# -- the validator must notice deliberate damage -----------------------------------
def damaged_copy(built, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(built["root"], root)
    return root, root / "manifests" / "tiny.json", root / "data"


def rewrite(path, fn):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = fn(rows)
    Path(path).write_text("".join(dataset.canonical_json(r) + "\n" for r in rows))


def codes(report):
    return {e.split(":")[0] for e in report["errors"]}


def test_validator_catches_a_wrong_answer(built, tmp_path):
    _, manifest, data = damaged_copy(built, tmp_path)

    def flip(rows):
        row = next(r for r in rows if r["task"] == tasks.REACHABILITY)
        row["training_target"]["answer"] = not row["training_target"]["answer"]
        row["private_truth"]["answer"] = dict(row["training_target"])
        return rows

    rewrite(data / "train.samples.jsonl", flip)
    found = codes(validate.validate(manifest))
    assert {"qa.answer", "scorer.accepts_training_target", "files.samples_hash"} <= found


def test_validator_catches_a_non_minimal_repair_target(built, tmp_path):
    _, manifest, data = damaged_copy(built, tmp_path)

    def worsen(rows):
        row = next(r for r in rows if r["task"] == tasks.REPAIR and r["private_truth"]["status"] == "unchanged")
        add = row["public_input"]["query"]["allowed_additions"][0]
        row["training_target"] = {"status": "edited", "edits": [{"op": "add_edge", "src": add[0], "dst": add[1]}]}
        return rows

    rewrite(data / "train.samples.jsonl", worsen)
    assert "repair.status" in codes(validate.validate(manifest))


def test_validator_catches_a_damaged_image(built, tmp_path):
    _, manifest, data = damaged_copy(built, tmp_path)
    rec = built["images"]["train"][0]
    Image.new("RGB", (512, 512), (255, 255, 255)).save(data / rec["image_path"])
    found = codes(validate.validate(manifest))
    assert {"image.sha256", "pixels.match_drawing"} <= found


def test_validator_catches_a_topology_copied_into_another_split(built, tmp_path):
    _, manifest, data = damaged_copy(built, tmp_path)
    src = next(r for r in built["images"]["train"] if r["variant"] == "root")
    clone = copy.deepcopy(src)
    clone.update({"image_id": "imgCLONE", "graph_group_id": "gCLONE", "split": "test_id", "image_path": "images/test_id/imgCLONE.png"})
    shutil.copyfile(data / src["image_path"], data / clone["image_path"])
    rewrite(data / "test_id.images.jsonl", lambda rows: rows + [clone])
    found = codes(validate.validate(manifest))
    assert {"leakage.no_isomorphic_graph_in_other_group", "leakage.no_identical_image_across_groups"} <= found


def test_validator_catches_a_group_split_across_two_splits(built, tmp_path):
    _, manifest, data = damaged_copy(built, tmp_path)
    moved = {}

    def take(rows):
        gid = rows[0]["graph_group_id"]
        moved["rows"] = [dict(r, split="test_id") for r in rows if r["graph_group_id"] == gid and r["variant"] == "edit"][:1]
        return [r for r in rows if r["image_id"] != moved["rows"][0]["image_id"]]

    rewrite(data / "train.images.jsonl", take)
    rewrite(data / "test_id.images.jsonl", lambda rows: rows + moved["rows"])
    assert "leakage.group_in_one_split" in codes(validate.validate(manifest))


def _locked_build(built, tmp_path):
    """Second dataset next to tiny.json that must avoid all of tiny's topologies.

    Same name, seed and split name as tiny's train split, so without the exclusion its first
    candidates would be exactly tiny's train graphs.
    """
    root, manifest, _ = damaged_copy(built, tmp_path)
    cfg = tiny_config()
    cfg["splits"] = {"train": dict(ID, roots=6)}
    cfg["exclude_isomorphic_to"] = ["tiny.json"]
    path = root / "manifests" / "tiny_locked.json"
    man = dataset.build_dataset(cfg, root / "data_locked", path, log=lambda *_: None)
    return root, path, man


def _nx(rec):
    g = nx.DiGraph()
    g.add_nodes_from(rec["nodes"])
    g.add_edges_from(tuple(e) for e in rec["edges"])
    return g


def test_exclusion_avoids_every_topology_of_the_earlier_dataset(built, tmp_path):
    root, path, man = _locked_build(built, tmp_path)
    assert man["excluded_manifests"] == {"tiny.json": built["manifest"]["content_digest"]}
    assert man["rejections"]["train"].get("duplicate_topology", 0) > 0  # the exclusion did bite
    report = validate.validate(path)
    assert report["ok"], report["errors"][:5]
    _, _, _, new_images = dataset.load_dataset(path)
    old = [_nx(r) for rows in built["images"].values() for r in rows]
    for rec in new_images["train"]:
        a = _nx(rec)
        for b in old:
            if a.number_of_nodes() == b.number_of_nodes() and a.number_of_edges() == b.number_of_edges():
                assert not nx.is_isomorphic(a, b), rec["image_id"]


def test_validator_catches_a_changed_excluded_dataset(built, tmp_path):
    root, path, _ = _locked_build(built, tmp_path)
    tiny = root / "manifests" / "tiny.json"
    m = json.loads(tiny.read_text())
    m["content_digest"] = "0" * 64
    tiny.write_text(json.dumps(m))
    assert "leakage.excluded_manifest_unchanged" in codes(validate.validate(path))


def test_validator_catches_a_graph_that_repeats_an_excluded_one(built, tmp_path):
    root, path, _ = _locked_build(built, tmp_path)
    data = root / "data_locked"
    src = next(r for r in built["images"]["train"] if r["variant"] == "root")
    clone = copy.deepcopy(src)
    clone.update({"image_id": "imgCLONE", "graph_group_id": "gCLONE"})
    shutil.copyfile(built["data_dir"] / src["image_path"], data / clone["image_path"])
    rewrite(data / "train.images.jsonl", lambda rows: rows + [clone])
    assert "leakage.no_isomorphic_graph_in_other_group" in codes(validate.validate(path))


def test_locked_test_splits_are_guarded():
    dataset.check_split_allowed("dev", "eval", env={})
    dataset.check_split_allowed("train", "train", env={})
    with pytest.raises(SystemExit):
        dataset.check_split_allowed("test_id", "eval", env={})
    with pytest.raises(SystemExit):
        dataset.check_split_allowed("test_style_ood", "eval", env={dataset.UNLOCK_ENV: "yes"})
    dataset.check_split_allowed("test_id", "eval", env={dataset.UNLOCK_ENV: dataset.UNLOCK_VALUE})
    with pytest.raises(SystemExit):  # never, not even when unlocked
        dataset.check_split_allowed("test_id", "train", env={dataset.UNLOCK_ENV: dataset.UNLOCK_VALUE})


def test_node_quota_equal_gives_exact_counts_per_size(tmp_path):
    cfg = copy.deepcopy(dataset.DEFAULTS)
    cfg.update({"name": "quota", "seed": 11,
                "splits": {"train": dict(ID, roots=8, nodes=[5, 8], node_quota="equal")}})
    dataset.build_dataset(cfg, tmp_path / "data", tmp_path / "manifests" / "quota.json", log=lambda *_: None)
    _, _, _, images = dataset.load_dataset(tmp_path / "manifests" / "quota.json")
    roots = [i for i in images["train"] if i["variant"] == "root"]
    sizes = sorted(len(i["nodes"]) for i in roots)
    assert sizes == [5, 5, 6, 6, 7, 7, 8, 8]


def test_node_count_for_default_matches_the_original_draw():
    from trace_vr.rng import Rng
    sc = {"nodes": [7, 12]}
    for idx in range(20):
        rng = Rng("x", idx)
        assert dataset.node_count_for(sc, idx, rng) == Rng("x", idx).child("n").randint(7, 12)
        assert dataset.node_count_for(dict(sc, node_quota="equal"), idx, rng) == 7 + idx % 6


def test_unknown_node_quota_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: q\nseed: 1\nsplits:\n  train:\n    roots: 2\n    families: {tree_plus: 1}\n"
                 "    nodes: [5, 6]\n    styles: [a]\n    node_quota: balanced\n")
    with pytest.raises(ValueError, match="node_quota"):
        dataset.load_config(p)
