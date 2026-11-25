"""The Kaggle final-evaluation kernel builder: refuses without the confirm flag, ties the adapter to its
recorded sha256, attaches the source kernel, and final_eval.sh no longer hard-codes /content."""
import importlib.util, json, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("make_final_kernel", ROOT / "scripts/kaggle/make_final_kernel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_refuses_without_confirm():
    with pytest.raises(SystemExit):
        load().main(["jobF1", "job14", "pilot3_B_s2027"])


def test_adapter_sha_from_manifest(tmp_path):
    m = load()
    man = tmp_path / "a.json"
    man.write_text(json.dumps({"files": [{"path": "job14/runs/pilot3_B_s2027/adapter/adapter_model.safetensors",
                                          "sha256": "ab" * 32, "bytes": 1}]}))
    assert m.adapter_sha("job14", "pilot3_B_s2027", man) == "ab" * 32
    with pytest.raises(SystemExit):
        m.adapter_sha("job14", "pilot3_D_s2027", man)


def test_build_kernel_for_recorded_adapter():
    m = load()
    files = json.loads((ROOT / "runs/kaggle_artifacts.json").read_text())["files"]
    rec = next(f for f in files if f["path"].endswith("/adapter/adapter_model.safetensors"))
    source, _, run = rec["path"].split("/")[:3]
    out = m.main(["test_final_build", source, run, "--final-evaluation"])
    try:
        text = (out / "kernel.py").read_text()
        meta = json.loads((out / "kernel-metadata.json").read_text())
        assert rec["sha256"] in text and "--final-evaluation" in text and "final_eval.sh" in text
        assert meta["kernel_sources"] == ["yiitefeerdomu/trace-%s" % source] and meta["is_private"]
        compile(text, "kernel.py", "exec")
    finally:
        for p in out.iterdir():
            p.unlink()
        out.rmdir()


def test_final_eval_paths_configurable():
    sh = (ROOT / "scripts/colab/final_eval.sh").read_text()
    body = "\n".join(l for l in sh.splitlines() if not l.startswith("#"))
    assert 'TRACE_HOME=${TRACE_HOME:-/content/trace}' in body and 'cd "$TRACE_HOME"' in body
    assert "/content" not in body.replace("${TRACE_HOME:-/content/trace}", "").replace("${TRACE_TMP:-/content}", "")


def test_build_parse_solve_kernel_for_arm_e():
    """Arm E's final kernel checks and copies the PARSER adapter (pilot3_C_desc) and calls final_eval.sh in
    parse-and-solve mode; outputs still go to runs/pilot3_E."""
    m = load()
    files = json.loads((ROOT / "runs/kaggle_artifacts.json").read_text())["files"]
    rec = next(f for f in files if f["path"].endswith("runs/pilot3_C_desc/adapter/adapter_model.safetensors"))
    source = rec["path"].split("/")[0]
    out = m.main(["test_final_e", source, "pilot3_E", "--final-evaluation", "--parse-solve=pilot3_C_desc"])
    try:
        text = (out / "kernel.py").read_text()
        assert rec["sha256"] in text
        assert "runs/pilot3_C_desc/adapter/adapter_model.safetensors" in text
        assert '"test_final_e", "pilot3_E", "--final-evaluation", "--parse-solve=pilot3_C_desc"]' in text
        compile(text, "kernel.py", "exec")
    finally:
        for p in out.iterdir():
            p.unlink()
        out.rmdir()


@pytest.mark.parametrize("extra", [["--parse-solve=pilot3_E"], ["--parse-solve=other"], ["--bogus"],
                                   ["--parse-solve=pilot3_C_desc", "--parse-solve=pilot3_C"]])
def test_parse_solve_option_rejects_bad_input(extra):
    with pytest.raises(SystemExit):
        load().main(["jobX", "job13", "pilot3_E", "--final-evaluation"] + extra)


def test_final_eval_parse_solve_branch():
    sh = (ROOT / "scripts/colab/final_eval.sh").read_text()
    assert "--parse-solve=pilot3_*) PARSE_SOLVE=1; ADAPTER=runs/${MODE#--parse-solve=}/adapter" in sh
    for sset in ("main", "invariance"):
        assert ('python scripts/eval_parse_solve.py "$MAN" "$SPLIT" "$ROOTS" "$RUN" "$ADAPTER" $MAXNEW_E %s' % sset) in sh
    assert "MAXNEW_E=512" in sh and "MAXNEW=512" in (ROOT / "scripts/colab/job_e.sh").read_text()
    assert '*) echo "unknown fourth argument $MODE"; exit 2' in sh


@pytest.mark.parametrize("sample_set", ["main", "invariance"])
def test_eval_parse_solve_with_oracle_extraction(tmp_path, monkeypatch, sample_set):
    """eval_parse_solve.py end to end with the model replaced by a perfect transcriber: every answer is right,
    files carry the run_suffix, and the invariance copies are scored."""
    import runpy
    pytest.importorskip("torch")  # trace_vr.model imports torch; the CPU-only .venv skips, .venv-model runs it
    sys.path.insert(0, str(ROOT / "src"))
    from trace_vr import arms, dataset, model as M, parse_solve as ps, tasks
    manifest = ROOT / "manifests/smoke.json"
    if not (ROOT / "data/smoke").exists():
        pytest.skip("smoke data not built")
    man, data_dir, by_split, _ = dataset.load_dataset(manifest)
    split = "dev" if "dev" in by_split else sorted(by_split)[0]
    rows = arms.select_samples(by_split[split], 4, sample_set)
    if not rows:
        pytest.skip("no %s samples in smoke %s" % (sample_set, split))
    truth = {}
    for s in rows:
        t = s["private_truth"]
        truth[dataset.to_model_example(s, data_dir)["image"]] = tasks.format_target(
            tasks.solve_query(ps.extraction_query(), t["nodes"], t["edges"]))
    monkeypatch.setattr(M, "load", lambda *a, **k: (type("Fake", (), {"eval": lambda self: None})(), None, "cpu"))
    monkeypatch.setattr(M, "build_inference_inputs", lambda processor, image, prompt: image)
    monkeypatch.setattr(M, "generate", lambda model, processor, image, max_new: {"text": truth[image], "hit_limit": 0, "new_tokens": 1})
    monkeypatch.setattr(sys, "argv", ["eval_parse_solve.py", str(manifest), split, "4", str(tmp_path), "none", "512", sample_set])
    runpy.run_path(str(ROOT / "scripts/eval_parse_solve.py"), run_name="__main__")
    suffix = arms.run_suffix(split, "image", sample_set)
    summary = json.loads((tmp_path / ("metrics_%s.json" % suffix)).read_text())
    assert summary["sample_set"] == sample_set and summary["samples"] == len(rows)
    assert summary["extraction_strict"]["exact"] == 1.0
    assert (tmp_path / ("predictions_%s.jsonl" % suffix)).exists() and (tmp_path / ("extraction_%s.jsonl" % suffix)).exists()
    if sample_set == "invariance":
        acc = summary["metrics"]["invariance_copy_accuracy"]
        assert acc and all(v["accuracy"] == 1.0 for v in acc.values())
    else:
        assert summary["metrics"]["non_edit_macro_accuracy"] == 1.0


# --- arm A (frozen base) on the locked test: no adapter, prompt variant and limit taken from the dev choice ---

def _choice(tmp_path, chosen="direct", limit=96, split="dev"):
    p = tmp_path / "base_prompt.json"
    p.write_text(json.dumps({"chosen": chosen, "chosen_limit": limit, "split": split,
                             "variants": {v: {"chosen_limit": limit} for v in ("common", "direct", "system")}}))
    return p


def test_build_base_kernel_uses_the_dev_choice(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "CHOICE", _choice(tmp_path, "system", 128))
    out = m.main(["test_final_a", "none", "A", "--final-evaluation", "--base-prompt"])
    try:
        text = (out / "kernel.py").read_text()
        meta = json.loads((out / "kernel-metadata.json").read_text())
        assert '"test_final_a", "base_prompt_system", "--final-evaluation", "--base-prompt=system:128"]' in text
        assert "/kaggle/input" not in text and "adapter ok" not in text and "no adapter: frozen base" in text
        assert meta["kernel_sources"] == [] and meta["is_private"] and meta["enable_gpu"]
        compile(text, "kernel.py", "exec")
    finally:
        for p in out.iterdir():
            p.unlink()
        out.rmdir()


def test_base_kernel_refused_without_a_dev_choice(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "CHOICE", tmp_path / "absent.json")
    with pytest.raises(SystemExit):
        m.main(["jobX", "none", "A", "--final-evaluation", "--base-prompt"])
    for bad in (_choice(tmp_path, "direct", 96, split="test_id"), _choice(tmp_path, "prefill", 96)):
        monkeypatch.setattr(m, "CHOICE", bad)
        with pytest.raises(SystemExit):
            m.main(["jobX", "none", "A", "--final-evaluation", "--base-prompt"])
    assert not (ROOT / "dist/kaggle/jobX").exists()


@pytest.mark.parametrize("argv", [["jobX", "none", "A", "--final-evaluation"],
                                  ["jobX", "job11", "pilot3_B", "--final-evaluation", "--base-prompt"],
                                  ["jobX", "none", "pilot3_B", "--final-evaluation", "--base-prompt"],
                                  ["jobX", "none", "A", "--final-evaluation", "--base-prompt", "--parse-solve=pilot3_C_desc"],
                                  ["jobX", "none", "A", "--final-evaluation", "--base-prompt=direct:96"]])
def test_base_kernel_rejects_mixed_arguments(argv, tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "CHOICE", _choice(tmp_path))
    with pytest.raises(SystemExit):
        m.main(argv)


@pytest.mark.parametrize("run,mode", [("base_prompt_direct", "--base-prompt=bogus:96"),
                                      ("base_prompt_direct", "--base-prompt=direct:abc"),
                                      ("base_prompt_direct", "--base-prompt=direct:"),
                                      ("base_prompt_common", "--base-prompt=direct:96"),
                                      ("pilot3_B", "--base-prompt=direct:96")])
def test_final_eval_refuses_bad_base_prompt_before_anything_runs(run, mode, tmp_path):
    import subprocess
    res = subprocess.run(["bash", str(ROOT / "scripts/colab/final_eval.sh"), "jobX", run, "--final-evaluation", mode],
                         capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "TRACE_HOME": str(tmp_path)})
    assert res.returncode == 2 and "FINAL-START" not in res.stdout


def test_final_eval_base_prompt_branch():
    sh = (ROOT / "scripts/colab/final_eval.sh").read_text()
    assert "VARIANT=common" in sh and "ADAPTER=none" in sh
    assert '[ "$ADAPTER" = none ] || [ -d "$ADAPTER" ]' in sh
    for sset in ("main", "invariance"):
        assert ('python scripts/eval_arm.py "$MAN" "$SPLIT" "$ROOTS" "$RUN" "$ADAPTER" $MAXNEW image %s "$VARIANT"' % sset) in sh
    # the variants the shell accepts are exactly the ones base_prompt.py defines
    sys.path.insert(0, str(ROOT / "src"))
    from trace_vr import base_prompt
    assert 'case "$VARIANT" in %s) ;;' % "|".join(base_prompt.VARIANTS) in sh


def test_collaborator_account_with_adapter_dataset():
    import shutil
    m = load()
    files = json.loads((ROOT / "runs/kaggle_artifacts.json").read_text())["files"]
    rec = next(f for f in files if f["path"].endswith("/adapter/adapter_model.safetensors"))
    source, _, run = rec["path"].split("/")[:3]
    out = m.main(["test_final_owner", source, run, "--final-evaluation", "--owner=some-user",
                  "--adapter-dataset=some-user/trace-adapters"])
    try:
        meta = json.loads((out / "kernel-metadata.json").read_text())
        assert meta["id"] == "some-user/trace-test_final_owner" and meta["is_private"] is True
        assert meta["dataset_sources"] == ["some-user/trace-adapters"] and meta["kernel_sources"] == []
        assert rec["sha256"] in (out / "kernel.py").read_text()   # the adapter is still pinned to its recorded digest
        text = (out / "kernel.py").read_text()
        assert '/kaggle/input/**/%s/adapter/' % run in text and "/runs/%s/adapter/" % run not in text   # dataset drops runs/
        assert out.parent.name == "some-user"
    finally:
        shutil.rmtree(out.parent)
    with pytest.raises(SystemExit):
        m.main(["x", source, run, "--final-evaluation", "--adapter-dataset=no-slash"])
