"""VGCure external evaluation (M7): scoring fixed in docs/external_eval.md, image lookup in zips,
letterboxing to the dev canvas, the eval script on a stub model, and the guarded Kaggle kernel builder."""
import importlib.util, io, json, sys
from pathlib import Path

import pytest

from trace_vr import tasks, vgcure

ROOT = Path(__file__).resolve().parents[1]
RECORD = json.loads((ROOT / "reports/external/vgcure_subset.json").read_text())


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _s(i, task, answer, source="er", neg=None):
    return {"sample_id": "vgcure/%s/%d/%d" % (source, i // 3, i % 3), "root_id": "vgcure/%s/%d" % (source, i // 3),
            "source": source, "task": task, "answer": answer, "negative_kind": neg,
            "query": {"type": task, "src": "E1", "dst": "E2"}, "image": "VGCure/images/%s_directed_akg_images/%d.png" % (source, i // 3)}


T, F = '{"answer": true}', '{"answer": false}'


def test_scoring_balanced_reachability_and_macro():
    rows = [_s(0, tasks.EDGE_EXISTS, True), _s(1, tasks.EDGE_EXISTS, False, neg="reversed_edge"),
            _s(2, tasks.REACHABILITY, True), _s(5, tasks.REACHABILITY, True), _s(8, tasks.REACHABILITY, False)]
    # always-"true" answers: edge 1/2, reach yes 2/2, no 0/1 -> balanced 0.5 although raw accuracy is 2/3
    res = vgcure.evaluate(rows, {r["sample_id"]: T for r in rows})
    assert res["edge_exists_accuracy"] == 0.5 and res["reachability_balanced_accuracy"] == 0.5
    assert abs(res["reachability_accuracy"] - 2 / 3) < 1e-9 and res["external_macro"] == 0.5
    assert res["reversed_edge_negative_accuracy"] == 0.0 and res["format_ok"] == 1.0
    # format failures and missing answers are wrong, never crashes
    res = vgcure.evaluate(rows, {rows[0]["sample_id"]: "yes", rows[1]["sample_id"]: F})
    assert res["edge_exists_accuracy"] == 0.5 and res["format_ok"] == 0.2
    fl = vgcure.flags(rows, {rows[1]["sample_id"]: F})
    assert fl["correct"] == "01000" and fl["missing_responses"] == 4 and len(fl["order_sha"]) == 16


def test_image_member_with_and_without_folder():
    im = "VGCure/images/er_directed_akg_images/3.png"
    assert vgcure.image_member(["er_directed_akg_images/3.png", "er_directed_akg_images/33.png"], im) == "er_directed_akg_images/3.png"
    assert vgcure.image_member(["x/er_directed_akg_images/3.png"], im) == "x/er_directed_akg_images/3.png"
    assert vgcure.image_member(["3.png", "13.png"], im) == "3.png"
    with pytest.raises(ValueError):
        vgcure.image_member(["a/3.png", "b/3.png"], im)


def test_hub_files_and_zip_hashes_cover_every_source():
    files = vgcure.hub_files()
    assert len(files) == 3 * len(vgcure.SOURCES) and set(vgcure.ZIP_SHA256) == set(vgcure.SOURCES)
    assert all(len(h) == 64 for h in vgcure.ZIP_SHA256.values())


def test_letterbox_to_dev_canvas(tmp_path):
    from PIL import Image
    prep = _load("vgcure_prepare", "scripts/vgcure_prepare.py")
    buf = io.BytesIO(); Image.new("RGB", (1600, 800), "black").save(buf, format="PNG")
    assert prep.letterbox(buf.getvalue(), tmp_path / "a.png") == (1600, 800)
    out = Image.open(tmp_path / "a.png")
    assert out.size == (512, 512) and out.getpixel((256, 10)) == (255, 255, 255) and out.getpixel((256, 256)) == (0, 0, 0)


def _fake_subset(tmp_path):
    rows = []
    for sid in RECORD["sample_ids"]:
        _, source, g, k = sid.split("/")
        k = int(k)
        task = tasks.REACHABILITY if k == 2 else tasks.EDGE_EXISTS
        rows.append({"sample_id": sid, "root_id": "vgcure/%s/%s" % (source, g), "source": source, "task": task,
                     "query": {"type": task, "src": "E1", "dst": "E2"}, "answer": k != 1,
                     "instruction": "q", "negative_kind": "same_direction" if k == 1 else None,
                     "image": "VGCure/images/%s_directed_akg_images/%s.png" % (source, g)})
    (tmp_path / "subset_full.json").write_text(json.dumps({"subset_sha256": RECORD["subset_sha256"], "revision": RECORD["revision"], "samples": rows}))
    return rows


@pytest.mark.parametrize("parse", [False, True])
def test_eval_script_on_stub_model(tmp_path, monkeypatch, capsys, parse):
    pytest.importorskip("torch")  # trace_vr.model imports torch; the CPU-only .venv skips, .venv-model runs it
    from trace_vr import model as M
    rows = _fake_subset(tmp_path)
    calls = []
    class Stub:
        def eval(self):
            return self
    monkeypatch.setattr(M, "load", lambda *a, **k: (Stub(), object(), "cpu"))
    monkeypatch.setattr(M, "build_inference_inputs", lambda proc, path, prompt: (path, prompt))
    def gen(model, proc, inputs, max_new):
        calls.append(inputs)
        text = '{"nodes": ["E1", "E2"], "edges": [["E1", "E2"]]}' if parse else '{"answer": true}'
        return {"text": text, "new_tokens": 5, "hit_limit": False}
    monkeypatch.setattr(M, "generate", gen)
    ev = _load("eval_vgcure", "scripts/eval_vgcure.py")
    ev.main([str(tmp_path), str(tmp_path / "run"), "none"] + (["--parse-solve"] if parse else []))
    summary = json.loads((tmp_path / "run/metrics_vgcure.json").read_text())
    m = summary["metrics"]
    assert len(calls) == (500 if parse else 1500)          # E transcribes each image once
    # every answer is "true": negatives wrong, reachability items (all answer True here) right
    assert m["edge_exists_accuracy"] == 0.5 and m["reachability_accuracy"] == 1.0 and m["n"] == 1500
    out = capsys.readouterr().out
    flags = json.loads(out.split("FLAGS-SUMMARY ", 1)[1].splitlines()[0])
    assert flags["n"] == 1500 and flags["correct"].count("1") == 1000 and flags["subset_sha256"] == RECORD["subset_sha256"]


def test_kernel_builder_guard_and_build():
    mvk = _load("make_vgcure_kernel", "scripts/kaggle/make_vgcure_kernel.py")
    with pytest.raises(SystemExit):
        mvk.main(["test_vg_build", "base"])
    with pytest.raises(SystemExit):
        mvk.parse_spec("job1:pilot3_E:parse=pilot3_E")
    files = json.loads((ROOT / "runs/kaggle_artifacts.json").read_text())["files"]
    rec = next(f for f in files if f["path"].endswith("/adapter/adapter_model.safetensors"))
    source, _, run = rec["path"].split("/")[:3]
    out = mvk.main(["test_vg_build", "--after-final-evaluation", "base", "%s:%s" % (source, run)])
    try:
        text = (out / "kernel.py").read_text()
        meta = json.loads((out / "kernel-metadata.json").read_text())
        assert rec["sha256"] in text and "vgcure_prepare.py" in text and "eval_vgcure.py" in text
        assert meta["kernel_sources"] == ["yiitefeerdomu/trace-%s" % source] and meta["is_private"] is True
        assert meta["dataset_sources"] == []
        import base64, re, tarfile
        b64 = re.search(r'PAYLOAD = "([^"]+)"', text).group(1)
        names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64)), mode="r:gz").getnames()
        assert "reports/external/vgcure_subset.json" in names and not any(n.startswith("external/") for n in names)
    finally:
        import shutil
        shutil.rmtree(out)
