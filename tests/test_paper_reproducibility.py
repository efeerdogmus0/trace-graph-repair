"""paper/sections/reproducibility.tex: digests, versions and commands match the repository."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "paper" / "sections" / "reproducibility.tex").read_text()
BODY = "\n".join(l for l in RAW.splitlines() if not l.lstrip().startswith("%"))
TEX = " ".join(BODY.split())
MAIN = (ROOT / "paper" / "main.tex").read_text()
JOB = (ROOT / "scripts/colab/job.sh").read_text()


def test_wired_into_main_once_as_appendix():
    assert MAIN.count("\\input{sections/reproducibility}") == 1
    assert MAIN.index("\\appendix") < MAIN.index("\\input{sections/reproducibility}")


def test_digests_and_configs_match_manifests():
    rows = re.findall(r"^(\S+) & \\texttt\{data\\_(\S+)\.yaml\} & \\texttt\{([0-9a-f]{16})\}", BODY, re.M)
    assert len(rows) == 4
    for name, cfg, prefix in rows:
        name = name.replace("\\_", "_")
        assert cfg.replace("\\_", "_") == name
        m = json.loads((ROOT / f"manifests/{name}.json").read_text())
        assert m["content_digest"].startswith(prefix), name
        assert (ROOT / f"configs/data_{name}.yaml").exists()
        assert m["generator_version"] in TEX.replace("\\_", "_") or m["generator_version"] in TEX


def test_generator_version():
    assert 'GENERATOR_VERSION = "trace-vr-gen-1"' in (ROOT / "src/trace_vr/__init__.py").read_text()
    assert "trace-vr-gen-1" in TEX


def test_job_hyperparameters():
    assert "EPOCHS=3; ACCUM=4; SEED=3407; CKPT=256; MAXNEW=96" in JOB
    for n in ("$3$ epochs", "accumulation $4$", "seed $3407$", "every $256$ steps", "most $96$ new tokens"):
        assert n in TEX, n
    assert "content digest differs" in JOB


def test_pins_and_unlock():
    mk = (ROOT / "scripts/kaggle/make_kernel.py").read_text()
    assert 'PINS = "transformers==5.22.2 peft==0.21.2"' in mk
    assert "transformers==5.22.2" in TEX and "peft==0.21.2" in TEX
    ds = (ROOT / "src/trace_vr/dataset.py").read_text()
    assert 'UNLOCK_ENV = "TRACE_UNLOCK_TEST"' in ds and 'UNLOCK_VALUE = "final-evaluation"' in ds
    assert "TRACE\\_UNLOCK\\_TEST" in TEX and "\\texttt{final-evaluation}" in TEX
    fe = (ROOT / "scripts/colab/final_eval.sh").read_text()
    assert len(re.search(r'SPLITS="([^"]+)"', fe).group(1).split()) == 6
    assert "six locked splits" in TEX
    assert "local flags differ" in (ROOT / "scripts/kaggle/collect.py").read_text()


def test_no_model_results():
    text = re.sub(r"\\(?:texttt|url)\{[^}]*\}", "", BODY)
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    assert nums <= {"3", "4", "3407", "256", "96", "256", "16", "1", "4.0", "2.0"}, nums


def test_licences_match_files():
    assert ROOT.joinpath("LICENSE").read_text().startswith("MIT License")
    data = ROOT.joinpath("LICENSE-DATA.md").read_text()
    assert "CC BY 4.0" in data and "Apache-2.0" in data and "LICENSE_DEJAVU" in data
    assert ROOT.joinpath("THIRD_PARTY.md").exists()
    for s in ("MIT licence (\\texttt{LICENSE})", "CC BY 4.0 (\\texttt{LICENSE-DATA.md})", "Apache-2.0", "DejaVu",
              "\\texttt{THIRD\\_PARTY.md}"):
        assert s in TEX, s


def test_gpu_environment_matches_kernel_env():
    path = ROOT / "reports/final/kernel_env.json"
    if not path.exists():
        import pytest
        pytest.skip("reports/final/kernel_env.json not written yet")
    ke = json.loads(path.read_text())
    rec = {j: e for j, e in ke["jobs"].items() if e is not None}
    assert ke["differing_fields"] == [] and rec
    env = next(iter(rec.values()))
    words = {6: "six", 5: "five", 7: "seven", 8: "eight"}
    assert "The %s final kernels that print their environment" % words[len(rec)] in TEX
    for v in (env["torch"], env["cuda"], env["python"]):
        assert "\\texttt{%s}" % v in TEX or "\\texttt{torch %s}" % v in TEX, v
    assert set(env["gpus"]) == {"Tesla T4"} and "Tesla T4 GPUs" in TEX
    assert "jobR1" in ke["not_recorded"] and "clean-clone rerun, did not record it" in TEX


def test_code_url_is_the_release_repo():
    import importlib.util
    spec = importlib.util.spec_from_file_location("release_export", ROOT / "scripts/release_export.py")
    rx = importlib.util.module_from_spec(spec); spec.loader.exec_module(rx)
    url = "https://github.com/efeerdogmus0/%s" % rx.NAME
    assert "\\url{%s}" % url in TEX
    assert "\\url{%s}" % url in (ROOT / "paper/sections/intro.tex").read_text()
