import importlib.util, json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_cards", ROOT / "scripts/model_cards.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

TRAIN = {"arm": "D", "base_id": "Qwen/Qwen3.5-2B", "revision": "r", "seed": 2027, "epochs": 3, "lr": 0.0002}
METRICS = {
    "adapter": "runs/x/adapter", "eval_root_graphs": 200, "samples": 3, "input_mode": "image",
    "max_new_tokens": 96, "truncated_at_limit": 0,
    "metrics": {"per_task": {"reachability": {"n": 3, "accuracy": 2 / 3, "format_failure_rate": 0.0}},
                "non_edit_macro_accuracy": 0.5},
}


def _fake(tmp_path, metrics=METRICS):
    rec = {"path": "job9/runs/x/adapter/adapter_model.safetensors", "bytes": 5, "sha256": "ab" * 32}
    (tmp_path / "m.json").write_text(json.dumps({"files": [rec, {"path": "job9/job9.log", "bytes": 1, "sha256": "0"}]}))
    d = tmp_path / "k/job9/runs/x"
    d.mkdir(parents=True)
    (d / "train_summary.json").write_text(json.dumps(TRAIN))
    if metrics is not None:
        (d / "metrics_dev.json").write_text(json.dumps(metrics))
    return mc.build(kaggle=tmp_path / "k", manifest_path=tmp_path / "m.json")


def test_card_copies_numbers_and_hash(tmp_path):
    cards = _fake(tmp_path)
    assert set(cards) == {"x.md", "README.md"}
    text = cards["x.md"]
    assert "ab" * 32 in text and "seed 2027" in text and "**0.500**" in text and "| reachability | 3 | 0.667 |" in text
    assert "| [`x`](x.md) | D | 2027 | job9 | 0.500 |" in cards["README.md"]


def test_card_without_own_eval(tmp_path):
    cards = _fake(tmp_path, metrics=None)
    assert "No dev evaluation" in cards["x.md"] and "| job9 | - |" in cards["README.md"]


def test_metrics_for_other_adapter_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _fake(tmp_path, metrics=dict(METRICS, adapter="runs/other/adapter"))


@pytest.mark.skipif(not (ROOT / "runs/kaggle").exists(), reason="Kaggle outputs not downloaded here")
def test_committed_cards_match_generator():
    for name, text in mc.build().items():
        assert (ROOT / "docs/model_cards" / name).read_text() == text, name


def test_identical_adapter_copy_in_an_evaluation_kernel_is_ignored():
    f = lambda job, sha: {"path": "%s/runs/x/adapter/adapter_model.safetensors" % job, "sha256": sha, "bytes": 1}
    got = mc.adapters({"files": [f("job9", "a" * 64), f("jobV0", "a" * 64)]})
    assert got["x"][0] == "job9"
    with pytest.raises(SystemExit, match="two different adapters"):
        mc.adapters({"files": [f("job9", "a" * 64), f("jobV0", "b" * 64)]})


def _fake_tested(tmp_path, parser=False):
    queue = {"jobs": [{"job": "jobF99", "arm": "E" if parser else "D", "seed": 2027,
                       "run": "e_run" if parser else "x", **({"parser": "x"} if parser else {})}]}
    arm = {"label": "E" if parser else "D", "per_seed": {"2027": {
        "minimal_graph_repair": 0.25, "reachability": 0.75, "critical_edge": 0.5, "non_edit_macro_accuracy": 0.625}}}
    table = {"splits": ["test_id"], "tables": {"test_id": {"root_graphs": 7, "arms": [arm]}}}
    (tmp_path / "q.json").write_text(json.dumps(queue))
    (tmp_path / "t.json").write_text(json.dumps(table))
    _fake(tmp_path)  # writes inputs
    return mc.build(kaggle=tmp_path / "k", manifest_path=tmp_path / "m.json",
                    queue_path=tmp_path / "q.json", table_path=tmp_path / "t.json")


def test_locked_test_section_copies_own_seed(tmp_path):
    cards = _fake_tested(tmp_path)
    text = cards["x.md"]
    assert "Evaluated once in kernel `trace-jobF99` (arm **D**, seed 2027)" in text
    assert "| test_id | 7 | 0.250 | 0.750 | 0.500 | 0.625 |" in text
    assert "Not evaluated on the locked test" not in text and "Evaluated once on the locked test" in text
    assert "| 0.500 | jobF99 |" in cards["README.md"]


def test_parser_card_says_numbers_are_arm_e(tmp_path):
    cards = _fake_tested(tmp_path, parser=True)
    assert "graph parser of arm **E**" in cards["x.md"] and "jobF99 (as E parser)" in cards["README.md"]


def test_missing_seed_in_table_refused(tmp_path):
    cards_dir = tmp_path
    (cards_dir / "q.json").write_text(json.dumps({"jobs": [{"job": "jobF99", "arm": "D", "seed": 1, "run": "x"}]}))
    (cards_dir / "t.json").write_text(json.dumps({"splits": ["test_id"], "tables": {"test_id": {"root_graphs": 1, "arms": []}}}))
    _fake(tmp_path)
    with pytest.raises(SystemExit, match="no D seed 1"):
        mc.build(kaggle=tmp_path / "k", manifest_path=tmp_path / "m.json",
                 queue_path=tmp_path / "q.json", table_path=tmp_path / "t.json")


@pytest.mark.skipif(not (ROOT / "reports/final/final_table.json").exists(), reason="locked test not opened")
def test_committed_cards_no_longer_say_untested():
    for run in ("pilot3_B", "pilot3_D_s9173", "pilot3_C_s2027", "pilot3_C_desc_s2027"):
        text = (ROOT / "docs/model_cards" / f"{run}.md").read_text()
        assert "## Locked-test evaluation" in text and "Not evaluated on the locked test" not in text, run
    assert "Not evaluated on the locked test" in (ROOT / "docs/model_cards/pilot3_D2.md").read_text()
