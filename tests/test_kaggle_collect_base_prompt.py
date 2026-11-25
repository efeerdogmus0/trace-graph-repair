"""collect_base_prompt.py: all three frozen-base variants or nothing; flags and the choice must match the kernel log.

The fake kernels below hold constant answers, not model output."""
import json, subprocess, sys
from pathlib import Path
import pytest
from trace_vr import arms, base_prompt, dataset

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests/pilot.json"
pytestmark = pytest.mark.skipif(not (ROOT / "data/pilot/dev.samples.jsonl").exists(), reason="pilot data not built")
MACRO = {"common": 0.10, "direct": 0.55, "system": 0.50}


def _run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})


def _fake_kernel(tmp_path, job="jobA9", skip=None, end=True, tamper=None, choice_line=None):
    man, _, by_split, _ = dataset.load_dataset(MAN)
    rows = arms.select_samples(by_split["dev"], 3, "main")
    kdir = tmp_path / "k"
    log = ["JOB-START %s arm=A base_prompt roots=3 limit=768 2026-01-01T00:00:00Z" % job, "peft 0.21.2 transformers 5.22.2"]
    for v in base_prompt.VARIANTS:
        if v == skip:
            continue
        run = kdir / "runs" / base_prompt.run_name(v); run.mkdir(parents=True)
        with open(run / "predictions_dev.jsonl", "w") as fh:
            for s in rows:
                fh.write(json.dumps({"sample_id": s["sample_id"], "task": s["task"], "text": '{"answer": true}'}) + "\n")
        (run / "metrics_dev.json").write_text(json.dumps({
            "split": "dev", "eval_root_graphs": 3, "samples": len(rows), "adapter": "none", "max_new_tokens": 768,
            "input_mode": "image", "sample_set": "main", "prompt_variant": "common" if tamper == "variant" else v,
            "content_digest": man["content_digest"], "eval_seconds": 5.5}))
        (run / "token_budget_dev.json").write_text(json.dumps({
            "run_limit": 768, "trunc_max": 0.02, "chosen_limit": 96, "split": "dev", "eval_root_graphs": 3,
            "samples": len(rows), "content_digest": man["content_digest"],
            "limits": {"96": {"non_edit_macro": MACRO[v], "non_edit_macro_lenient": MACRO[v], "max_truncation_rate": 0.0}}}))
        res = _run("scripts/export_flags.py", str(MAN), "dev", "3", str(run), "image")
        assert res.returncode == 0, res.stderr
        line = [l for l in res.stdout.splitlines() if l.startswith("FLAGS-SUMMARY ")][-1]
        if tamper == "flags" and v == "direct":
            d = json.loads(line[14:]); d["predictions_sha256"] = "0" * 64; line = "FLAGS-SUMMARY " + json.dumps(d)
        log.append(line)
    log.append("BASE-PROMPT " + json.dumps(choice_line or {"chosen": "direct", "limit": 96, "macro": MACRO}))
    if end:
        log.append("JOB-END %s 2026-01-01T01:00:00Z" % job)
    kdir.mkdir(exist_ok=True)
    (kdir / (job + ".log")).write_text("\n".join(log) + "\n")
    return kdir


def _collect(kdir, out, *extra):
    return _run("scripts/kaggle/collect_base_prompt.py", "jobA9", "--kernel-dir", str(kdir), "--out-root", str(out), *extra)


def test_collect_writes_all_three_variants(tmp_path):
    kdir, out = _fake_kernel(tmp_path), tmp_path / "out"
    res = _collect(kdir, out)
    assert res.returncode == 0, res.stderr
    budget = json.loads(res.stdout.splitlines()[0])
    assert budget["gpu_seconds"] == 16.5 and budget["run_id"] == "base_prompt_jobA9" and budget["device"] is None
    assert "chosen direct at 96" in res.stdout
    for v in base_prompt.VARIANTS:
        d = out / base_prompt.run_name(v)
        assert json.loads((d / "summary.json").read_text())["prompt_variant"] == v
        assert json.loads((d / "flags_dev.json").read_text())["run"] == base_prompt.run_name(v)
    # the tracked copies are what base_prompt_choice.py reads
    res = _run("scripts/base_prompt_choice.py", str(out))
    assert res.returncode == 0 and '"chosen": "direct"' in res.stdout
    again = _collect(kdir, out)
    assert again.returncode != 0 and "--force" in (again.stderr + again.stdout)
    assert _collect(kdir, out, "--force").returncode == 0


@pytest.mark.parametrize("kw,needle", [
    ({"skip": "system"}, "all variants are needed"),
    ({"end": False}, "did not finish"),
    ({"tamper": "flags"}, "predictions_sha256"),
    ({"tamper": "variant"}, "prompt_variant"),
    ({"choice_line": {"chosen": "system", "limit": 96, "macro": MACRO}}, "BASE-PROMPT"),
])
def test_collect_refuses_and_writes_nothing(tmp_path, kw, needle):
    kdir, out = _fake_kernel(tmp_path, **kw), tmp_path / "out"
    res = _collect(kdir, out)
    assert res.returncode != 0 and needle in (res.stderr + res.stdout)
    assert not out.exists()
