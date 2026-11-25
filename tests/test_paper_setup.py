"""paper/sections/setup.tex must state only numbers that the code and generated docs fix."""
import inspect, re
from pathlib import Path

from trace_vr import bootstrap, token_budget

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper" / "sections" / "setup.tex").read_text()
TRAIN = (ROOT / "scripts" / "train_arm.py").read_text()
JOB = (ROOT / "scripts" / "colab" / "job.sh").read_text()
MODEL = (ROOT / "src" / "trace_vr" / "model.py").read_text()
PROTOCOL = (ROOT / "docs" / "protocol.md").read_text()
EXPOSURE = (ROOT / "docs" / "exposure_pilot_train.md").read_text()


def tex_int(s: str) -> int:
    return int(s.replace("{,}", ""))


def test_training_constants_match_train_script():
    lr, warm, r = re.search(r"LR, WARMUP, R = ([\d.e-]+), ([\d.]+), (\d+)", TRAIN).groups()
    assert float(lr) == 2e-4 and r"$2\times10^{-4}$" in TEX
    assert float(warm) == 0.03 and r"$3\%$ of" in TEX
    assert int(r) == 16 and "$r=16$, $\\alpha=16$" in TEX
    assert "lora_alpha=R, lora_dropout=0.0" in TRAIN and "no dropout" in TEX
    assert "weight_decay=0.01" in TRAIN and "weight decay $0.01$" in TEX
    assert "clip_grad_norm_(params, 1.0)" in TRAIN and "clipping at $1.0$" in TEX
    assert "ACCUM=4" in JOB and "4-step gradient" in TEX
    # LoRA targets only the language model, so the vision encoder is frozen
    assert r'TARGETS = r"model\.language_model\.' in TRAIN and "vision encoder is frozen" in TEX


def test_base_model_and_seeds():
    assert 'BASE_ID = "Qwen/Qwen3.5-2B"' in MODEL
    rev = re.search(r'BASE_REVISION = "(\w+)"', MODEL).group(1)
    assert "\\texttt{Qwen/Qwen3.5-2B}" in TEX and "\\texttt{%s}" % rev[:8] in TEX
    assert "(3407, 2027, 9173)" in PROTOCOL and "3407, 2027 and 9173" in TEX
    assert "SEED=3407" in JOB


def test_decoding_and_bootstrap_rules():
    assert token_budget.CANDIDATES == (96, 192, 384, 768)
    assert "96, 192, 384\nand 768" in TEX or "96, 192, 384 and 768" in TEX.replace("\n", " ")
    assert token_budget.TRUNC_MAX == 0.02 and r"$2\%$ truncation" in TEX
    assert inspect.signature(bootstrap.cluster_bootstrap).parameters["n_boot"].default == 2000
    assert "2{,}000 resamples" in TEX


def test_exposure_table_matches_generated_doc():
    rows = {}
    for line in EXPOSURE.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 11 and cells[0] in ("B", "C", "D", "D2"):
            rows[cells[0]] = cells
    assert set(rows) == {"B", "C", "D", "D2"}
    seen = set()
    for m in re.finditer(r"^(B|C|D|D2)\s*&(.*?)\\\\", TEX, re.M):
        arm, vals = m.group(1), [tex_int(v.strip()) for v in m.group(2).split("&")]
        doc = rows[arm]
        # samples, root groups, unique images, instruction tokens, target tokens
        assert vals == [int(doc[1]), int(doc[2]), int(doc[3]), int(doc[8]), int(doc[9])], arm
        seen.add(arm)
    assert seen == set(rows)
    assert rows["B"][4] == "546" and "shares 546 of D's" in TEX
    assert all(int(r[5]) == 256 * int(r[1]) for r in rows.values()) and "256 image tokens" in TEX


def test_arm_a_sentence_matches_the_dev_choice():
    """Every number in the arm A item is the one in reports/pilot/base_prompt.json (jobA1, collected from files)."""
    import json
    from pathlib import Path
    from trace_vr import base_prompt
    root = Path(__file__).resolve().parents[1]
    c = json.loads((root / "reports/pilot/base_prompt.json").read_text())
    tex = " ".join((root / "paper/sections/setup.tex").read_text().split())
    item = tex[tex.index(r"\item[A, frozen base.]"):tex.index(r"\item[B, QA-SFT.]")]
    v = c["variants"]
    assert c["split"] == "dev" and c["chosen"] == "system" and "system message at %d new tokens" % c["chosen_limit"] in item
    assert "%d development roots" % c["eval_root_graphs"] in item
    assert "scored $%.4f$ at %d new tokens (lenient parse $%.4f$" % (
        v["common"]["non_edit_macro"], v["common"]["chosen_limit"], v["common"]["non_edit_macro_lenient"]) in item
    assert "truncation up to $%.2f$" % v["common"]["max_truncation_rate"] in item
    assert "$%.4f$ after the question and $%.4f$ as a system message, both at %d new tokens with no truncation" % (
        v["direct"]["non_edit_macro"], v["system"]["non_edit_macro"], v["direct"]["chosen_limit"]) in item
    assert v["direct"]["chosen_limit"] == v["system"]["chosen_limit"]
    assert v["direct"]["max_truncation_rate"] == v["system"]["max_truncation_rate"] == 0
    assert base_prompt.DIRECT_SENTENCE.replace(". ", ". ") in item.replace("``", "").replace("''", "")
    # 80 non-edit questions: two questions = 0.025 of the macro when the two tasks have 40 each
    assert abs(v["system"]["non_edit_macro"] - v["direct"]["non_edit_macro"] - 2 / 80) < 1e-9


def test_scale_paragraph_matches_manifests_protocol_and_job():
    import json
    tex = " ".join(TEX.split())
    para = tex[tex.index(r"\paragraph{Scale.}"):tex.index(r"\paragraph{Decoding and scoring.}")]
    pilot = json.loads((ROOT / "manifests/pilot.json").read_text())["config"]["splits"]
    test = json.loads((ROOT / "manifests/pilot_test.json").read_text())["config"]["splits"]
    assert pilot["train"]["roots"] == 512 and pilot["train"]["nodes"] == [5, 10] and pilot["dev"]["roots"] == 200
    assert "512 training root graphs with 5 to 10 nodes for 3 epochs" in para and "EPOCHS=3" in JOB
    assert "200 development roots" in para
    order = ("test_id", "test_style_ood", "test_struct_ood", "test_combined_ood")
    assert [test[s]["roots"] for s in order] == [200, 200, 200, 100]
    assert "200, 200, 200 and 100 root graphs (ID, style-OOD, structure-OOD, combined-OOD)" in para
    assert test["test_struct_ood"]["nodes"] == test["test_combined_ood"]["nodes"] == [11, 16]
    assert "structure-OOD and combined-OOD graphs have 11 to 16 nodes" in para
    assert "main train about 5,000" in PROTOCOL and "ID test 500, style-OOD test 500, structure-OOD test 500, combined-OOD test 250" in " ".join(PROTOCOL.split())
    assert "about 5{,}000 training roots and locked splits of 500, 500, 500 and 250 roots; that build was not run" in para
    assert "pilot-scale study" in para


def test_benchmark_caption_no_longer_promises_the_main_build():
    bench = (ROOT / "paper/sections/benchmark.tex").read_text()
    assert "not built yet" not in bench and "was not built" in bench
