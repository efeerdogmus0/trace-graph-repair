import json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
import final_plan  # noqa: E402

QUEUE = json.loads((ROOT / "scripts/kaggle/final_queue.json").read_text())["jobs"]
TRAIN_QUEUE = json.loads((ROOT / "scripts/kaggle/queue.json").read_text())["jobs"]


def test_queue_covers_every_arm_seed_once_in_seed_rounds():
    pairs = [(j["arm"], j["seed"]) for j in QUEUE]
    assert len(pairs) == len(set(pairs)) == 12
    assert {a for a, _ in pairs} == {"B", "C", "D", "E"}
    seeds = [j["seed"] for j in QUEUE]
    assert seeds == [3407] * 4 + [2027] * 4 + [9173] * 4
    for k in range(0, 12, 4):
        assert [j["arm"] for j in QUEUE[k:k + 4]] == ["B", "D", "E", "C"]
    assert len({j["job"] for j in QUEUE}) == 12


def test_sources_are_the_training_kernels_of_each_run():
    by_out = {(j["arm"], j["seed"]): j for j in TRAIN_QUEUE}
    for j in QUEUE:
        if (j["arm"], j["seed"]) in by_out:
            t = by_out[(j["arm"], j["seed"])]
            assert j["source"] == t["job"]
            assert j["dev_dir"] == t["out"]
        if j["arm"] == "E":
            want = "pilot3_C_desc" if j["seed"] == 3407 else "pilot3_C_desc_s%d" % j["seed"]
            assert j["parser"] == want
        else:
            assert "parser" not in j


def test_collected_adapters_have_a_recorded_sha():
    files = {f["path"] for f in json.loads((ROOT / "runs/kaggle_artifacts.json").read_text())["files"]}
    for j in QUEUE:
        if not (ROOT / "runs/kaggle" / j["source"]).exists():
            continue   # training kernel not collected yet (job22)
        adapter = j.get("parser", j["run"])
        assert "%s/runs/%s/adapter/adapter_model.safetensors" % (j["source"], adapter) in files


def test_plan_formula_and_no_test_samples_read():
    assert final_plan.final_roots() == 900
    p = final_plan.plan(ROOT / "scripts/kaggle/final_queue.json", ROOT / "runs/kaggle", ROOT,
                        datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc), 30.0, "job22")
    assert abs(p["scale"] - 900 / 200 * (1 + p["inv_per_main"])) < 1e-3
    for r in p["jobs"]:
        want = r["dev_eval_seconds"] * p["scale"] + p["setup_seconds_assumed"]
        assert abs(r["estimate_hours"] - want / 3600) < 0.01
        assert not r["over_11h"]
    assert not any(r["fits_this_week"] for r in p["jobs"] if r["done"])   # a collected kernel is not planned again
    week = [r["fits_this_week"] for r in p["jobs"] if not r["done"]]
    assert week == sorted(week, reverse=True)   # open kernels packed in queue order, no skipping ahead
    assert p["inv_per_main_source"] in ("measured", "recorded")
    if p["inv_per_main_source"] == "measured":
        assert abs(final_plan.inv_per_main()[0] - final_plan.INV_PER_MAIN_RECORDED) < 1e-12


def test_base_row_is_unknown_until_the_dev_choice_and_then_an_upper_bound(tmp_path):
    """Arm A: outside the twelve rows and the weekly packing; fake choice and fake dev seconds, no model output."""
    queue = tmp_path / "q.json"
    queue.write_text(json.dumps({"base": {"job": "jobF13", "arm": "A", "choice": "reports/pilot/base_prompt.json"}, "jobs": []}))
    row = final_plan.base_row(queue, tmp_path, 5.0, 1000.0)
    assert "estimate_hours" not in row and "no dev choice yet" in row["state"] and not row["done"]
    choice = tmp_path / "reports/pilot/base_prompt.json"; choice.parent.mkdir(parents=True)
    choice.write_text(json.dumps({"chosen": "direct", "chosen_limit": 96}))
    assert "summary.json missing" in final_plan.base_row(queue, tmp_path, 5.0, 1000.0)["state"]
    run = tmp_path / "runs/base_prompt_direct"; run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"eval_dev": {"eval_seconds": 360.0, "eval_root_graphs": 20, "max_new_tokens": 768}}))
    row = final_plan.base_row(queue, tmp_path, 5.0, 1000.0)
    assert row["dev_eval_seconds_per_200_roots"] == 3600.0 and row["upper_bound"] and not row["over_11h"]
    assert abs(row["estimate_hours"] - (3600.0 * 5.0 + 1000.0) / 3600) < 0.01
    choice.write_text(json.dumps({"chosen": "direct", "chosen_limit": 768}))
    assert not final_plan.base_row(queue, tmp_path, 5.0, 1000.0)["upper_bound"]
    assert final_plan.base_row(queue, tmp_path, 12.0, 1000.0)["over_11h"]
    md = final_plan.to_md(dict(final_plan.plan(ROOT / "scripts/kaggle/final_queue.json", ROOT / "runs/kaggle", ROOT,
                                               datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc), 30.0, "job22"), base=row))
    assert "Frozen base (arm A, jobF13" in md and "upper bound" not in md.split("Frozen base")[1].split("estimate from")[0]


def test_real_plan_has_twelve_rows_and_a_separate_base_row():
    p = final_plan.plan(ROOT / "scripts/kaggle/final_queue.json", ROOT / "runs/kaggle", ROOT,
                        datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc), 30.0, "job22")
    assert len(p["jobs"]) == 12 and p["base"]["job"] == "jobF13" and p["base"]["arm"] == "A"
    assert all(r["job"] != "jobF13" for r in p["jobs"])


def test_measured_final_seconds_and_calibration_never_below_plain(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/budget.jsonl").write_text(
        json.dumps({"run_id": "pilot3_E_s9173_jobF11", "gpu_seconds": 100.0,
                    "status": "complete; final locked-test evaluation"}) + "\n"
        + json.dumps({"run_id": "pilot3_E_job13", "gpu_seconds": 5.0, "status": "complete; training"}) + "\n")
    assert final_plan.measured_final_seconds(tmp_path) == {"jobF11": 100.0}
    p = final_plan.plan(ROOT / "scripts/kaggle/final_queue.json", ROOT / "runs/kaggle", ROOT,
                        datetime(2026, 11, 15, 4, 0, tzinfo=timezone.utc), 30.0, None)
    for r in p["jobs"]:
        assert r["arm_ratio"] >= 1.0
        assert r["calibrated_hours"] >= r["estimate_hours"] - 0.01
        assert r["over_11h"] == (r["calibrated_hours"] * 3600 > final_plan.FLAG_S + 36)


def test_gate_counts_running_final_kernel_once_with_its_calibrated_estimate():
    now = datetime(2026, 11, 21, 1, 0, tzinfo=timezone.utc)   # a fresh quota week
    idle = final_plan.plan(ROOT / "scripts/kaggle/final_queue.json", ROOT / "runs/kaggle", ROOT, now, 30.0, None)
    pending = [r for r in idle["jobs"] if not r["done"]]
    if not pending:
        assert idle["verdict"] == "done"
        return
    first = pending[0]
    assert idle["next"] == first["job"] and idle["verdict"] == ("ok" if first["fits_this_week"] else "wait")
    busy = final_plan.plan(ROOT / "scripts/kaggle/final_queue.json", ROOT / "runs/kaggle", ROOT, now, 30.0, first["job"])
    row = next(r for r in busy["jobs"] if r["job"] == first["job"])
    assert not row["fits_this_week"]                        # not packed a second time
    assert abs(busy["running_estimate_hours"] - first["calibrated_hours"]) < 0.01
    assert busy["next"] != first["job"]
