"""Frozen-base prompt variants and the dev-only selection rule (trace_vr/base_prompt.py)."""
import json, os, subprocess, sys
from pathlib import Path

import pytest

from trace_vr import base_prompt as bp

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))


def scan(macro, limit=96, trunc=0.0, **over):
    res = {"run_limit": 768, "trunc_max": 0.02, "chosen_limit": limit, "split": "dev", "eval_root_graphs": 30,
           "samples": 180, "content_digest": "d",
           "limits": {str(limit): {"non_edit_macro": macro, "non_edit_macro_lenient": macro,
                                   "max_truncation_rate": trunc, "per_task_accuracy": {}, "per_task_format_failure": {}}}}
    res.update(over)
    return res


def test_variants_change_only_the_added_sentence():
    q = "Look at the graph.\nAnswer with a single JSON object and nothing else."
    assert bp.apply("common", q) == (None, q)
    assert bp.apply("direct", q) == (None, q + "\n" + bp.DIRECT_SENTENCE)
    assert bp.apply("system", q) == (bp.DIRECT_SENTENCE, q)
    with pytest.raises(KeyError):
        bp.apply("prefill", q)
    assert bp.VARIANTS[0] == "common"


def test_variants_are_for_the_frozen_base_only():
    bp.check_allowed("common", "runs/x/adapter")
    bp.check_allowed("direct", "none")
    with pytest.raises(SystemExit):
        bp.check_allowed("direct", "runs/x/adapter")
    with pytest.raises(SystemExit):
        bp.check_allowed("other", "none")


def test_highest_strict_macro_wins_and_ties_go_to_the_earlier_variant():
    c = bp.choose({"common": scan(0.10, 768, 0.4), "direct": scan(0.55), "system": scan(0.50)})
    assert c["chosen"] == "direct" and c["chosen_limit"] == 96
    assert c["variants"]["common"]["max_truncation_rate"] == 0.4
    assert bp.choose({"common": scan(0.5), "direct": scan(0.5), "system": scan(0.5)})["chosen"] == "common"
    assert bp.choose({"common": scan(0.4), "direct": scan(0.5), "system": scan(0.5)})["chosen"] == "direct"


def test_lenient_score_is_not_used_to_pick():
    hi = scan(0.2); hi["limits"]["96"]["non_edit_macro_lenient"] = 0.9
    assert bp.choose({"common": scan(0.3), "direct": hi, "system": scan(0.1)})["chosen"] == "common"


def test_refuses_incomplete_or_mismatched_runs():
    with pytest.raises(ValueError):
        bp.choose({"common": scan(0.1), "direct": scan(0.5)})
    with pytest.raises(ValueError):
        bp.choose({"common": scan(0.1), "direct": scan(0.5), "system": scan(0.5, eval_root_graphs=10)})
    with pytest.raises(ValueError):
        bp.choose({v: scan(0.5, split="test_id") for v in bp.VARIANTS})


def test_choice_script(tmp_path):
    cmd = [sys.executable, "scripts/base_prompt_choice.py", str(tmp_path), "--out", str(tmp_path / "c.json"),
           "--out-md", str(tmp_path / "c.md")]
    for v, m in (("common", 0.1), ("direct", 0.6)):
        d = tmp_path / bp.run_name(v); d.mkdir()
        (d / "token_budget_dev.json").write_text(json.dumps(scan(m)))
    bad = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV)
    assert bad.returncode != 0 and "all variants are needed" in bad.stderr and not (tmp_path / "c.json").exists()
    d = tmp_path / bp.run_name("system"); d.mkdir()
    (d / "token_budget_dev.json").write_text(json.dumps(scan(0.3)))
    ok = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout.split("BASE-PROMPT ", 1)[1])["chosen"] == "direct"
    assert json.loads((tmp_path / "c.json").read_text())["chosen"] == "direct"
    assert "| direct (chosen) | 96 | 0.6000 |" in (tmp_path / "c.md").read_text()


def test_job_script_runs_every_variant_in_rule_order():
    sh = (ROOT / "scripts/colab/base_prompt.sh").read_text()
    assert "for VARIANT in %s; do" % " ".join(bp.VARIANTS) in sh
    assert 'RUN=runs/base_prompt_${VARIANT}' in sh and bp.run_name("direct") == "base_prompt_direct"
    assert ' dev "$ROOTS" "$RUN" none "$LIMIT" image main "$VARIANT"' in sh
    assert "test" not in sh.replace("latest", "")   # dev only


def test_make_kernel_arm_a():
    base = [sys.executable, "scripts/kaggle/make_kernel.py"]
    bad = subprocess.run(base + ["jobBaseT", "A", "30:edges"], cwd=ROOT, capture_output=True, text=True, env=ENV)
    assert bad.returncode != 0 and "arm A" in bad.stderr
    ok = subprocess.run(base + ["jobBaseT", "A", "30:image"], cwd=ROOT, capture_output=True, text=True, env=ENV)
    out = ROOT / "dist/kaggle/jobBaseT"
    try:
        assert ok.returncode == 0, ok.stderr
        assert "base_prompt.sh" in (out / "kernel.py").read_text()
    finally:
        for f in out.glob("*"):
            f.unlink()
        if out.exists():
            out.rmdir()


def test_eval_arm_passes_the_system_message(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from trace_vr import model as M
    seen = []

    class Proc:
        def apply_chat_template(self, messages, **kw):
            seen.append(messages)
            return {"input_ids": torch.zeros((1, 1), dtype=torch.long)}

    from PIL import Image
    img = tmp_path / "a.png"; Image.new("RGB", (8, 8), "white").save(img)
    M.build_inference_inputs(Proc(), str(img), "q")
    M.build_inference_inputs(Proc(), str(img), "q", system="s")
    assert [m["role"] for m in seen[0]] == ["user"]
    assert [m["role"] for m in seen[1]] == ["system", "user"] and seen[1][0]["content"][0]["text"] == "s"
    assert seen[1][1] == seen[0][0]
