import importlib.util, io, json, re, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_payload", ROOT / "scripts" / "build_payload.py")
bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
JOB_SCRIPTS = ["scripts/colab/job.sh", "scripts/colab/final_eval.sh", "scripts/colab/base_budget.sh",
               "scripts/colab/base_prompt.sh"]


def _referenced(text):
    return set(re.findall(r"\b((?:scripts|configs|manifests)/[\w./-]+\.(?:py|yaml|json|sh))", text))


def test_payload_contains_everything_the_job_scripts_call():
    data = bp.build(ROOT)
    names = set(tarfile.open(fileobj=io.BytesIO(data)).getnames())
    for js in JOB_SCRIPTS:
        assert js in names
        text = (ROOT / js).read_text()
        # final_eval.sh builds config/manifest paths from a loop variable; expand them.
        for name in re.findall(r"for NAME in ([\w ]+); do", text):
            for n in name.split():
                text += " configs/data_%s.yaml manifests/%s.json" % (n, n)
        needed = _referenced(text)
        assert needed and needed <= names, (js, needed - names)
    assert "src/trace_vr/resume.py" in names and "pyproject.toml" in names
    assert not any("__pycache__" in n or n.startswith("data/") for n in names)


def test_payload_is_byte_reproducible():
    import time
    a = bp.build(ROOT); time.sleep(1.1)
    assert a == bp.build(ROOT)


def test_final_eval_covers_every_locked_split_with_all_roots():
    text = (ROOT / "scripts/colab/final_eval.sh").read_text()
    items = [p.split(":") for p in re.search(r'SPLITS="([^"]+)"', text).group(1).split()]
    want = []
    for name in ("pilot_test", "pilot_diag"):
        man = json.loads((ROOT / ("manifests/%s.json" % name)).read_text())
        want += [[name, s, str(man["splits"][s]["roots"])] for s in man["split_order"]]
        assert re.search(r"for NAME in [^;]*\b%s\b" % name, text), name
    assert items == want
    assert "TRACE_UNLOCK_TEST=final-evaluation" in text and "--final-evaluation" in text


def test_diag_build_is_isomorphism_disjoint_from_pilot_and_test():
    man = json.loads((ROOT / "manifests/pilot_diag.json").read_text())
    rec = man["excluded_manifests"]
    for name in ("pilot", "pilot_test"):
        other = json.loads((ROOT / ("manifests/%s.json" % name)).read_text())
        assert rec[name + ".json"] == other["content_digest"]
    assert all(s.startswith("test") for s in man["split_order"])  # locked by check_split_allowed


def test_every_script_that_scores_a_split_checks_the_lock():
    # Any script taking a split argument and reading labels must refuse locked test splits.
    for p in sorted((ROOT / "scripts").glob("*.py")):
        src = p.read_text()
        if re.search(r"\bsplit\b.*=.*sys\.argv", src) and "load_dataset" in src:
            assert "check_split_allowed" in src, p.name
