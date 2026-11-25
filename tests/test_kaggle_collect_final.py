"""collect_final.py: all twelve final evaluations must be present, finished and equal to the log before anything
is written. The fake kernel answers a constant, so no model output on the locked test is involved."""
import importlib.util, json, re, subprocess, sys
from pathlib import Path
import pytest
from conftest import require_data
from trace_vr import arms, dataset

ROOT = Path(__file__).resolve().parents[1]
UNLOCK = {"TRACE_UNLOCK_TEST": "final-evaluation"}
SMALL = [("pilot_test", "test_id", 3), ("pilot_diag", "test_size_ood", 2)]
needs_data = pytest.mark.skipif(not (ROOT / "data/pilot_test/test_id.samples.jsonl").exists()
                                or not (ROOT / "data/pilot_diag/test_size_ood.samples.jsonl").exists(),
                                reason="locked test data not built")


def load():
    spec = importlib.util.spec_from_file_location("collect_final", ROOT / "scripts/kaggle/collect_final.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_kernel(tmp_path, tamper=None, drop=None, end=True):
    require_data(*sorted({m for m, _, _ in SMALL}))
    kdir = tmp_path / "k"; run = kdir / "runs/pilot3_X"; run.mkdir(parents=True)
    lines = ["FINAL-START jobFX run=runs/pilot3_X", "adapter ok /kaggle/input/x"]
    for manifest, split, roots in SMALL:
        man = ROOT / "manifests" / (manifest + ".json")
        _, _, by_split, _ = dataset.load_dataset(man)
        for sample_set in ("main", "invariance"):
            suffix = arms.run_suffix(split, "image", sample_set)
            if drop == suffix:
                continue
            rows = arms.select_samples(by_split[split], roots, sample_set)
            with open(run / ("predictions_%s.jsonl" % suffix), "w") as fh:
                for s in rows:
                    fh.write(json.dumps({"sample_id": s["sample_id"], "task": s["task"], "text": '{"answer": true}'}) + "\n")
            (run / ("metrics_%s.json" % suffix)).write_text(json.dumps({"eval_root_graphs": roots, "eval_seconds": 1.5}))
            res = subprocess.run([sys.executable, "scripts/export_flags.py", str(man), split, str(roots), str(run), "image", sample_set],
                                 cwd=ROOT, capture_output=True, text=True,
                                 env=dict({"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}, **UNLOCK))
            assert res.returncode == 0, res.stderr
            line = [l for l in res.stdout.splitlines() if l.startswith("FLAGS-SUMMARY ")][-1]
            d = json.loads(line[14:])
            d["images_digest"] = "f" * 64          # another machine's renderer: recorded, not compared
            if tamper == suffix:
                d["predictions_sha256"] = "0" * 64
            lines.append("FLAGS-SUMMARY " + json.dumps(d))
    if end:
        lines.append("FINAL-END jobFX 2026-01-01T00:00:00Z")
    (kdir / "jobFX.log").write_text("\n".join(lines) + "\n")
    return kdir


@needs_data
def test_collect_final_writes_all_flags(tmp_path):
    m = load()
    kdir, out = _fake_kernel(tmp_path), tmp_path / "out"
    budget, written = m.collect("jobFX", "pilot3_X", kdir, out, splits=SMALL)
    assert budget["gpu_seconds"] == 6.0 and budget["run_id"] == "pilot3_X_jobFX"
    names = sorted(p.name for p in written)
    assert names == ["final_summary.json", "flags_test_id.json", "flags_test_id_inv.json",
                     "flags_test_size_ood.json", "flags_test_size_ood_inv.json"]
    flags = json.loads((out / "flags_test_id_inv.json").read_text())
    assert flags["split"] == "test_id" and flags["sample_set"] == "invariance" and flags["eval_root_graphs"] == 3
    assert flags["content_digest"] == json.loads((ROOT / "manifests/pilot_test.json").read_text())["content_digest"]
    summary = json.loads((out / "final_summary.json").read_text())
    assert len(summary["evaluations"]) == 4 and summary["images_digest"]["test_id"]["kernel"] == "f" * 64
    assert summary["images_digest"]["test_id"]["local"] != "f" * 64
    with pytest.raises(SystemExit, match="--force"):
        m.collect("jobFX", "pilot3_X", kdir, out, splits=SMALL)


@needs_data
@pytest.mark.parametrize("kw, msg", [({"tamper": "test_size_ood_inv"}, "predictions_sha256"),
                                     ({"drop": "test_size_ood_inv"}, "did not finish"),
                                     ({"end": False}, "FINAL-END")])
def test_collect_final_writes_nothing_on_any_failure(tmp_path, kw, msg):
    m = load()
    kdir, out = _fake_kernel(tmp_path, **kw), tmp_path / "out"
    with pytest.raises(SystemExit, match=msg):
        m.collect("jobFX", "pilot3_X", kdir, out, splits=SMALL)
    assert not out.exists()


def test_wrong_root_count_refused(tmp_path):
    m = load()
    if not (ROOT / "data/pilot_test/test_id.samples.jsonl").exists():
        pytest.skip("locked test data not built")
    kdir = _fake_kernel(tmp_path)
    bigger = [(a, b, c + 1) for a, b, c in SMALL]
    with pytest.raises(SystemExit, match="expected"):
        m.collect("jobFX", "pilot3_X", kdir, tmp_path / "out", splits=bigger)


def test_split_list_equals_final_eval_sh():
    m = load()
    sh = (ROOT / "scripts/colab/final_eval.sh").read_text()
    items = re.search(r'^SPLITS="([^"]+)"', sh, re.M).group(1).split()
    assert [tuple(i.split(":")[:2]) + (int(i.split(":")[2]),) for i in items] == m.SPLITS


def test_queue_entries_resolve_and_unknown_job_refused():
    m = load()
    jobs = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())["jobs"]
    for e in jobs:
        got = m.queue_entry(e["job"])
        assert got["run"].startswith("pilot3_") and got["dev_dir"].startswith("runs/pilot3_")
    assert len({e["dev_dir"] for e in jobs}) == len(jobs)      # no two final runs write into one directory
    with pytest.raises(SystemExit):
        m.queue_entry("job22")


def _choice(tmp_path, chosen="direct", limit=96):
    p = tmp_path / "choice.json"
    p.write_text(json.dumps({"chosen": chosen, "chosen_limit": limit, "split": "dev",
                             "variants": {v: {"chosen_limit": limit} for v in ("common", "direct", "system")}}))
    return p


def test_base_entry_comes_from_the_dev_choice_only(tmp_path):
    """Arm A (frozen base): no entry without the dev choice; run, directory and required metrics follow the choice."""
    m = load()
    base = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())["base"]
    jobs = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())["jobs"]
    assert base["job"] not in {e["job"] for e in jobs} and base["arm"] == "A" and len(jobs) == 12
    with pytest.raises(SystemExit, match="missing"):
        m.queue_entry(base["job"], choice=tmp_path / "absent.json")
    e = m.queue_entry(base["job"], choice=_choice(tmp_path, "system", 192))
    assert e["run"] == "base_prompt_system" and e["dev_dir"] == "runs/base_prompt_system"
    assert e["expect"] == {"adapter": "none", "prompt_variant": "system", "max_new_tokens": 192}
    assert e["dev_dir"] not in {j["dev_dir"] for j in jobs}


@needs_data
def test_collect_final_checks_required_metric_values(tmp_path):
    """The fake kernel's metrics carry no prompt variant, so a base entry's requirements refuse it and write nothing;
    requirements the metrics do meet pass."""
    m = load()
    kdir, out = _fake_kernel(tmp_path), tmp_path / "out"
    with pytest.raises(SystemExit, match="prompt_variant"):
        m.collect("jobFX", "pilot3_X", kdir, out, splits=SMALL, expect={"adapter": "none", "prompt_variant": "direct"})
    assert not out.exists()
    for manifest, split, roots in SMALL:
        _, _, by_split, _ = dataset.load_dataset(ROOT / "manifests" / (manifest + ".json"))
        for sample_set in ("main", "invariance"):
            suffix = arms.run_suffix(split, "image", sample_set)
            mp = kdir / "runs/pilot3_X" / ("metrics_%s.json" % suffix)
            mp.write_text(json.dumps(dict(json.loads(mp.read_text()), adapter="none", prompt_variant="direct", max_new_tokens=96)))
    want = {"adapter": "none", "prompt_variant": "direct", "max_new_tokens": 96}
    with pytest.raises(SystemExit, match="max_new_tokens"):
        m.collect("jobFX", "pilot3_X", kdir, out, splits=SMALL, expect=dict(want, max_new_tokens=768))
    assert not out.exists()
    budget, written = m.collect("jobFX", "pilot3_X", kdir, out, splits=SMALL, expect=want)
    assert len(written) == 5 and budget["device"] is None      # the fake log names no GPU
    assert json.loads((out / "final_summary.json").read_text())["required_metric_values"] == want
