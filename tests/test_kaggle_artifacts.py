"""scripts/kaggle/artifact_manifest.py: hashes files, skips ckpt/, ties predictions to tracked flags files."""
import hashlib, importlib.util, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("artifact_manifest", ROOT / "scripts/kaggle/artifact_manifest.py")
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)


def _tree(tmp, pred_bytes, flags_sha):
    k = tmp / "runs/kaggle/job99/runs/pilot3_X"
    (k / "adapter").mkdir(parents=True)
    (k / "ckpt").mkdir()
    (k / "adapter/adapter_model.safetensors").write_bytes(b"w" * 10)
    (k / "ckpt/state.pt").write_bytes(b"skip me")
    (k / "predictions_dev.jsonl").write_bytes(pred_bytes)
    (tmp / "runs/pilot3_X").mkdir(parents=True)
    (tmp / "runs/pilot3_X/flags_dev.json").write_text(json.dumps({"predictions_sha256": flags_sha}))
    return tmp / "runs/kaggle"


def test_hashes_skip_ckpt_and_match_flags(tmp_path):
    pred = b'{"id": 1}\n'
    kaggle = _tree(tmp_path, pred, hashlib.sha256(pred).hexdigest())
    m = am.build(kaggle, tmp_path)
    paths = [f["path"] for f in m["files"]]
    assert not any("ckpt" in p for p in paths) and len(paths) == 2
    p = next(f for f in m["files"] if f["path"].endswith("predictions_dev.jsonl"))
    assert p["sha256"] == hashlib.sha256(pred).hexdigest() and p["bytes"] == len(pred)
    assert p["flags_files"] == ["runs/pilot3_X/flags_dev.json"]
    assert am.unmatched_predictions(m) == []


def test_unmatched_predictions_reported(tmp_path):
    kaggle = _tree(tmp_path, b"changed\n", "0" * 64)
    m = am.build(kaggle, tmp_path)
    assert am.unmatched_predictions(m) == ["job99/runs/pilot3_X/predictions_dev.jsonl"]


def test_vgcure_predictions_are_hashed_without_a_flags_file(tmp_path):
    kaggle = _tree(tmp_path, b"changed\n", "0" * 64)
    v = kaggle / "jobV9/runs/pilot3_X"; v.mkdir(parents=True)
    (v / "predictions_vgcure.jsonl").write_bytes(b"ext\n")
    m = am.build(kaggle, tmp_path)
    e = next(f for f in m["files"] if f["path"] == "jobV9/runs/pilot3_X/predictions_vgcure.jsonl")
    assert e["external"] == "vgcure" and "flags_files" not in e and e["sha256"] == hashlib.sha256(b"ext\n").hexdigest()
    assert am.unmatched_predictions(m) == ["job99/runs/pilot3_X/predictions_dev.jsonl"]


def test_tracked_manifest_consistent():
    m = json.loads((ROOT / "runs/kaggle_artifacts.json").read_text())
    assert m["generated_by"] == "scripts/kaggle/artifact_manifest.py"
    assert am.unmatched_predictions(m) == []
    flags = am.flag_hashes(ROOT)
    for f in m["files"]:
        if "flags_files" in f:  # every recorded prediction hash is still the one a tracked flags file names
            assert flags.get(f["sha256"]) == f["flags_files"], f["path"]


def _final_manifest(tmp, entry, drop=None, adapter=b"w" * 10, collected=True):
    """Fake downloaded final kernel (constant bytes, no model output) + its source training kernel."""
    kaggle = tmp / "runs/kaggle"
    src = kaggle / entry["source"] / "runs" / entry.get("parser", entry["run"]) / "adapter"
    src.mkdir(parents=True); (src / "adapter_model.safetensors").write_bytes(b"w" * 10)
    cp = kaggle / entry["job"] / "runs" / entry.get("parser", entry["run"]) / "adapter"
    cp.mkdir(parents=True); (cp / "adapter_model.safetensors").write_bytes(adapter)
    run = kaggle / entry["job"] / "runs" / entry["run"]; run.mkdir(parents=True, exist_ok=True)
    dev = tmp / entry["dev_dir"]; dev.mkdir(parents=True)
    for _, split, _ in am.cf.SPLITS:
        for ss in am.cf.SAMPLE_SETS:
            suffix = am.cf.suffix_of(split, ss)
            if suffix == drop:
                continue
            body = ("fake %s\n" % suffix).encode()
            (run / ("predictions_%s.jsonl" % suffix)).write_bytes(body)
            if collected:
                (dev / ("flags_%s.json" % suffix)).write_text(json.dumps({"predictions_sha256": hashlib.sha256(body).hexdigest()}))
    return am.build(kaggle, tmp)


ENTRY_B = {"job": "jobF01", "arm": "B", "seed": 3407, "source": "job11", "run": "pilot3_B", "dev_dir": "runs/pilot3_B_kaggle"}
ENTRY_E = {"job": "jobF03", "arm": "E", "seed": 3407, "source": "job13", "run": "pilot3_E", "parser": "pilot3_C_desc",
           "dev_dir": "runs/pilot3_E"}
OTHER = {"job": "jobF02", "arm": "D", "seed": 3407, "source": "job20", "run": "pilot3_D", "dev_dir": "runs/pilot3_D_kaggle"}


def test_final_job_complete_and_undownloaded_jobs_are_not_problems(tmp_path):
    m = _final_manifest(tmp_path, ENTRY_B)
    assert am.final_problems(m, {"jobs": [ENTRY_B, OTHER]}) == (["jobF01"], [])


def test_final_job_arm_e_checks_the_parser_adapter(tmp_path):
    assert am.final_problems(_final_manifest(tmp_path / "a", ENTRY_E), {"jobs": [ENTRY_E]}) == (["jobF03"], [])
    present, problems = am.final_problems(_final_manifest(tmp_path / "b", ENTRY_E, adapter=b"other"), {"jobs": [ENTRY_E]})
    assert problems == ["jobF03: evaluated adapter pilot3_C_desc differs from the one trained by job13"]


def test_final_job_missing_evaluation_or_uncollected_is_a_problem(tmp_path):
    _, problems = am.final_problems(_final_manifest(tmp_path / "a", ENTRY_B, drop="test_size_ood_inv"), {"jobs": [ENTRY_B]})
    assert problems == ["jobF01: predictions_test_size_ood_inv.jsonl missing"]
    m = _final_manifest(tmp_path / "b", ENTRY_B, collected=False)
    _, problems = am.final_problems(m, {"jobs": [ENTRY_B]})
    assert len(problems) == 12 and all("run collect_final.py" in p for p in problems)
    assert len(am.unmatched_predictions(m)) == 12   # the generic check stops main() too


def test_real_queue_matches_tracked_manifest():
    m = json.loads((ROOT / "runs/kaggle_artifacts.json").read_text())
    queue = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())
    present, problems = am.final_problems(m, queue)
    assert problems == []
    paths = {f["path"] for f in m["files"]}
    for e in queue["jobs"]:   # every adapter the queue will evaluate is already hashed from its training kernel
        assert "%s/runs/%s/adapter/adapter_model.safetensors" % (e["source"], e.get("parser", e["run"])) in paths, e["job"]


BASE = {"job": "jobF13", "arm": "A", "choice": "reports/pilot/base_prompt.json"}
ENTRY_A = {"job": "jobF13", "arm": "A", "run": "base_prompt_direct", "dev_dir": "runs/base_prompt_direct"}


def _base_manifest(tmp, drop=None, adapter=False, collected=True):
    """Fake downloaded frozen-base final kernel (constant bytes, no model output)."""
    kaggle = tmp / "runs/kaggle"
    run = kaggle / "jobF13/runs/base_prompt_direct"; run.mkdir(parents=True)
    dev = tmp / "runs/base_prompt_direct"; dev.mkdir(parents=True)
    if adapter:
        (run / "adapter").mkdir(); (run / "adapter/adapter_model.safetensors").write_bytes(b"w")
    for _, split, _ in am.cf.SPLITS:
        for ss in am.cf.SAMPLE_SETS:
            suffix = am.cf.suffix_of(split, ss)
            if suffix == drop:
                continue
            body = ("base %s\n" % suffix).encode()
            (run / ("predictions_%s.jsonl" % suffix)).write_bytes(body)
            if collected:
                (dev / ("flags_%s.json" % suffix)).write_text(json.dumps({"predictions_sha256": hashlib.sha256(body).hexdigest()}))
    return am.build(kaggle, tmp)


def test_frozen_base_final_job_is_checked_without_an_adapter(tmp_path):
    q = {"jobs": [ENTRY_B], "base": BASE}
    assert am.final_problems(_base_manifest(tmp_path / "a"), q, base_entry=ENTRY_A) == (["jobF13"], [])
    _, problems = am.final_problems(_base_manifest(tmp_path / "b", drop="test_id_inv"), q, base_entry=ENTRY_A)
    assert problems == ["jobF13: predictions_test_id_inv.jsonl missing"]
    _, problems = am.final_problems(_base_manifest(tmp_path / "c", collected=False), q, base_entry=ENTRY_A)
    assert len(problems) == 12 and all("run collect_final.py" in p for p in problems)
    _, problems = am.final_problems(_base_manifest(tmp_path / "d", adapter=True), q, base_entry=ENTRY_A)
    assert len(problems) == 1 and "contains adapter files" in problems[0]
    # not downloaded: the base job is neither present nor a problem, whether or not a dev choice exists
    assert am.final_problems(_final_manifest(tmp_path / "e", ENTRY_B), q) == (["jobF01"], [])


def test_downloaded_base_job_without_a_readable_dev_choice_is_a_problem(tmp_path, monkeypatch):
    def refuse():
        raise SystemExit("missing choice")
    monkeypatch.setattr(am.cf, "base_entry", refuse)
    present, problems = am.final_problems(_base_manifest(tmp_path), {"jobs": [], "base": BASE})
    assert present == ["jobF13"] and len(problems) == 1 and "dev choice cannot be read" in problems[0]
