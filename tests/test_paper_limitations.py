"""paper/sections/limitations.tex: every number traces to an evidence file; no result claims."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "paper" / "sections" / "limitations.tex").read_text()
BODY = "\n".join(l for l in RAW.splitlines() if not l.lstrip().startswith("%"))
TEX = " ".join(BODY.split())
MAIN = (ROOT / "paper" / "main.tex").read_text()
SECTIONS = "".join(p.read_text() for p in (ROOT / "paper" / "sections").glob("*.tex"))


def test_wired_into_main_once():
    assert MAIN.count("\\input{sections/limitations}") == 1
    assert "\\section{Limitations}" not in MAIN


def test_refs_resolve():
    labels = set(re.findall(r"\\label\{([^}]+)\}", SECTIONS))
    for ref in re.findall(r"\\ref\{([^}]+)\}", BODY):
        assert ref in labels, ref


def test_ceilings_from_validator():
    v = json.loads((ROOT / "reports/pilot/validation.json").read_text())["text_only_shortcuts"]["dev"]
    for t in ("critical_edge", "minimal_graph_repair", "reachability"):
        assert v[t]["text_only_ceiling"] == 0.5
    assert "$0.50$" in TEX
    assert f"${v['edge_exists']['text_only_ceiling']}$ on edge existence" in TEX


def test_target_tokens_from_exposure_table():
    rows = {}
    for line in (ROOT / "docs/exposure_pilot_train.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 11 and cells[0] in {"D", "B", "C"}:
            rows[cells[0]] = int(cells[9])
    for arm, phrase in [("D", "D has"), ("B", "B"), ("C", "C")]:
        assert f"${rows[arm]:,}$".replace(",", "{,}") in TEX, arm


def test_topology_capacity_numbers():
    doc = (ROOT / "docs/datasheet.md").read_text()
    assert "about 100 five-node layered_dag classes, 255 tree_plus, 422" in " ".join(doc.split())
    assert "about $100$ five-node classes for \\texttt{layered\\_dag}, $255$ for \\texttt{tree\\_plus} and $422$" in TEX


def test_base_model_and_resolution():
    assert 'BASE_ID = "Qwen/Qwen3.5-2B"' in (ROOT / "src/trace_vr/model.py").read_text()
    assert "Qwen3.5-2B" in TEX
    assert "PNG drawings (512 x 512)" in (ROOT / "docs/datasheet.md").read_text()


def test_only_known_numbers():
    text = re.sub(r"\\(ref|label|cite)\{[^}]*\}", "", BODY).replace("{,}", "")
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    assert nums <= {"512", "2", "3.5", "0.50", "0.895", "16384", "5120", "28592", "100", "255", "422", "4",
                    "5000", "3", "200", "5", "10"}, nums


def test_no_result_claims():
    low = TEX.lower()
    for phrase in ["we show", "we find", "we found", "outperform", "improves", "significant",
                   "state-of-the-art", "our results"]:
        assert phrase not in low, phrase


def _para(name):
    start = TEX.index("\\paragraph{%s}" % name)
    end = TEX.find("\\paragraph{", start + 1)
    return TEX[start:end if end > 0 else None]


def test_pilot_scale_paragraph_matches_manifests_and_runs():
    para = _para("Pilot scale.")
    pilot = json.loads((ROOT / "manifests/pilot.json").read_text())["splits"]
    test = json.loads((ROOT / "manifests/pilot_test.json").read_text())["splits"]
    assert pilot["train"]["roots"] == 512 and "trained on $512$ root graphs" in para
    assert [test[s]["roots"] for s in ("test_id", "test_style_ood", "test_struct_ood", "test_combined_ood")] == [200, 200, 200, 100]
    assert "$200$ root graphs per split ($100$ for the combined split)" in para
    # every run that enters the dev comparison or the locked test trained for 3 epochs
    queue = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())
    epochs = {json.loads(p.read_text())["train"]["epochs"] for p in (ROOT / "runs").glob("pilot3_[BCDE]*/summary.json")}
    assert epochs == {3} and "for $3$ epochs" in para and queue
    protocol = (ROOT / "docs/protocol.md").read_text()
    assert "main train about 5,000" in protocol and "about $5{,}000$ training root graphs" in para
    assert "was not run. The locked test evaluates the" in " ".join(protocol.split()) and "It was not run" in para
    assert 5000 / 512 > 9.7 and "ten times the training data is not measured" in para


def test_controls_not_run_paragraph_matches_protocol_record():
    para = _para("Controls not run.")
    protocol = " ".join((ROOT / "docs/protocol.md").read_text().split())
    assert "Question-only / blank image; wrong image;" in protocol
    assert "Not run on any model: question-only / blank image and wrong image" in protocol
    assert "question-only (blank image) control and a wrong-image control. Neither was run on any model" in para
    assert "path length and crossings have no slice" in protocol and "no slice by path length or by edge crossings" in para
    # no evaluation entry point offers such a control
    for script in ("scripts/eval_arm.py", "scripts/colab/final_eval.sh"):
        low = (ROOT / script).read_text().lower()
        assert "blank" not in low and "wrong_image" not in low and "wrong-image" not in low, script
    # oracle (edges as text): 100 dev roots for B and for D, same sample order
    b = json.loads((ROOT / "runs/pilot3_B_kaggle_edges/bootstrap_edges_vs_image_100.json").read_text())
    fb = json.loads((ROOT / "runs/pilot3_B_kaggle_edges/flags_dev_edges.json").read_text())
    fd = json.loads((ROOT / "runs/pilot3_oracle/flags_D_edges_dev.json").read_text())
    assert b["root_graphs"] == 100 and fb["n"] == fd["n"] and fb["order_sha"] == fd["order_sha"]
    assert "evaluated on $100$ development roots for arms B and D and not on the locked test" in para
    assert "edges" not in (ROOT / "scripts/colab/final_eval.sh").read_text()   # the locked test has no oracle mode


def test_generator_paragraph_says_planned_main_build():
    para = _para("Generator coverage and selection.")
    cfg = json.loads((ROOT / "manifests/pilot.json").read_text())["config"]["splits"]
    assert cfg["train"]["nodes"] == [5, 10] == cfg["dev"]["nodes"]
    assert "planned main build therefore starts at seven nodes; the pilot graphs used here have 5 to 10" in para
