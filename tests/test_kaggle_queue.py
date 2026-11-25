import json, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
import next_job  # noqa: E402
import make_kernel  # noqa: E402


def test_queue_is_consistent_with_launcher_seeds():
    jobs = next_job.load(ROOT / "scripts/kaggle/queue.json")
    assert set(next_job.SEEDS) == set(make_kernel.SEEDS)
    # every seed of B, C, D appears exactly once (D at 3407 is the Kaggle retrain)
    for arm in "BCD":
        seeds = sorted(j["seed"] for j in jobs if j["arm"] == arm)
        expected = [2027, 3407, 9173] if arm == "D" else [2027, 9173]
        assert seeds == expected, arm
    assert jobs[0]["job"] == "job13" and jobs[0]["arm"] == "E"
    # arm E gets the other two seeds too (parser retrained), after the B/C/D seed jobs
    e = [j for j in jobs if j["arm"] == "E"]
    assert sorted(j["seed"] for j in e) == [2027, 3407, 9173]
    assert [j["out"] for j in e[1:]] == ["runs/pilot3_E_s2027", "runs/pilot3_E_s9173"]
    assert all(jobs.index(j) > max(jobs.index(k) for k in jobs if k["arm"] != "E") for j in e[1:])


def test_next_skips_collected_and_builds_commands(tmp_path, capsys):
    q = {"jobs": [
        {"job": "jobA", "arm": "B", "evals": "200:image", "seed": 2027, "out": "runs/pilot3_B_s2027"},
        {"job": "jobB", "arm": "D", "evals": "200:image", "seed": 3407, "out": "runs/pilot3_D_kaggle"}]}
    qp = tmp_path / "q.json"; qp.write_text(json.dumps(q))
    res = next_job.main(["--queue", str(qp), "--root", str(tmp_path), "--json"])
    assert res["next"]["job"] == "jobA"
    cmds = res["next"]["commands"]
    assert "make_kernel.py jobA B 200:image --seed 2027 --from-head" in cmds[0]
    assert cmds[-1].endswith("collect.py jobA B --seed 2027 --mode image --budget")
    (tmp_path / "runs/pilot3_B_s2027").mkdir(parents=True)
    (tmp_path / "runs/pilot3_B_s2027/summary.json").write_text("{}")
    res = next_job.main(["--queue", str(qp), "--root", str(tmp_path), "--json"])
    assert res["done"] == ["jobA"] and res["next"]["job"] == "jobB"
    assert "--seed" not in res["next"]["commands"][0]
    assert res["next"]["commands"][-1].endswith("collect.py jobB D --mode image --out runs/pilot3_D_kaggle --budget")


def test_rejects_unregistered_seed_and_duplicates(tmp_path):
    for jobs in ([{"job": "j", "arm": "B", "evals": "200:image", "seed": 1, "out": "a"}],
                 [{"job": "j", "arm": "B", "evals": "200:image", "seed": 2027, "out": "a"},
                  {"job": "j", "arm": "C", "evals": "200:image", "seed": 2027, "out": "b"}]):
        qp = tmp_path / "q.json"; qp.write_text(json.dumps({"jobs": jobs}))
        with pytest.raises(SystemExit):
            next_job.load(qp)
