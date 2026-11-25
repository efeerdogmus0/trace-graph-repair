"""demo/walkthrough.py and the generated CPU demo notebook.

The walk-through must show the verifier's own verdicts (metrics.score_sample), its candidate
labels must be true of the candidates, and the notebook's code cells must run without Jupyter
and print the picked root's samples."""
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest

from trace_vr import metrics, tasks
from test_dataset import built  # noqa: F401  (module-scoped fixture)

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "train"  # the tiny build has no dev split; "train" is not locked


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


walkthrough = _load("walkthrough", ROOT / "demo" / "walkthrough.py")
builder = _load("build_demo_notebook", ROOT / "scripts" / "build_demo_notebook.py")


def _by_id(built):
    return {s["sample_id"]: s for s in built["samples"][SPLIT]}


def test_every_verdict_is_the_scorers_and_labels_hold(built):
    samples, by_id = built["samples"][SPLIT], _by_id(built)
    seen = set()
    for index in range(3):
        secs = walkthrough.walkthrough(samples, index)
        assert [s["task"] for s in secs] == list(walkthrough.MAIN_TASKS)
        assert len({by_id[s["base"]["sample_id"]]["graph_group_id"] for s in secs}) == 1
        for sec in secs:
            for side in ("base", "twin"):
                sample = by_id[sec[side]["sample_id"]]
                for row in sec[side]["rows"]:
                    assert row["correct"] == bool(metrics.score_sample(sample, row["text"])["correct"])
            base_rows = {r["label"]: r for r in sec["base"]["rows"]}
            assert base_rows["the exact answer"]["correct"]
            assert not base_rows["a sentence instead of the JSON object"]["format_ok"]
            assert sum(r["correct"] for r in sec["base"]["rows"]) == 1  # only the exact answer is right
            # answer-flipping twin: the base's exact answer is wrong on the twin image, the twin's own is right
            assert [r["correct"] for r in sec["twin"]["rows"]] == [False, True]
            assert sec["twin"]["image"] != sec["base"]["image"]
            seen.update(r["error"] for r in sec["base"]["rows"] if sec["task"] == tasks.REPAIR)
    assert {"addition_not_allowed", "edge_not_present"} <= seen


def test_render_names_the_candidates_as_hand_written(built):
    text = walkthrough.render(walkthrough.walkthrough(built["samples"][SPLIT], 0)[2])
    assert "hand-written, not model outputs" in text and "CORRECT" in text and "WRONG" in text
    assert "legal=" in text and "minimal=" in text


def test_locked_split_and_bad_index_are_refused(built):
    with pytest.raises(SystemExit, match="locked"):
        walkthrough.load(str(built["manifest_path"]), "test_id")
    with pytest.raises(SystemExit, match="root index"):
        walkthrough.pick_root(built["samples"][SPLIT], 10 ** 6)


def test_tracked_notebook_equals_the_generator():
    assert (ROOT / "notebooks" / "demo_verifier_cpu.ipynb").read_text(encoding="utf-8") == builder.notebook()
    nb = json.loads(builder.notebook())
    assert nb["nbformat"] == 4 and all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    intro = "".join(nb["cells"][0]["source"])
    assert "No model runs here" in intro and "not model outputs" in intro


def test_notebook_code_cells_run_and_show_the_picked_root(built, monkeypatch):
    monkeypatch.setenv("TRACE_DEMO_MANIFEST", str(built["manifest_path"]))
    monkeypatch.setenv("TRACE_DEMO_SPLIT", SPLIT)
    monkeypatch.chdir(ROOT / "notebooks")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "walkthrough", raising=False)
    nb = json.loads((ROOT / "notebooks" / "demo_verifier_cpu.ipynb").read_text(encoding="utf-8"))
    ns, out = {}, io.StringIO()
    with contextlib.redirect_stdout(out):
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), "<demo cell>", "exec"), ns)
    text = out.getvalue()
    assert "rebuilt the smoke set" not in text  # the tiny build was used as given
    secs = walkthrough.walkthrough(built["samples"][SPLIT], 0)
    for sec in secs:
        assert "=== %s ===" % sec["task"] in text
        assert sec["base"]["sample_id"] in text and sec["twin"]["sample_id"] in text
    sheet = ns["walkthrough"].side_by_side(ns["data_dir"], ns["sections"][0])
    assert sheet.width > sheet.height and os.path.isdir(ns["data_dir"])
