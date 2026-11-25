import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
import quota  # noqa: E402


def write_job(kdir, job, arm, start, seconds):
    d = kdir / job
    d.mkdir(parents=True)
    (d / ("trace-%s.log" % job)).write_text(json.dumps(
        [{"stream_name": "stdout", "time": 1.0, "data": "x"}, {"stream_name": "stderr", "time": seconds, "data": "y"}]))
    (d / ("%s.log" % job)).write_text("JOB-START %s arm=%s roots=200 mode=image %s\nJOB-END %s x\n" % (job, arm, start, job))


def test_window_estimates_and_verdict(tmp_path):
    kdir = tmp_path / "runs/kaggle"
    write_job(kdir, "job1", "B", "2026-11-01T00:00:00Z", 9000.0)   # before the weekly reset (Sat 2026-11-07 00:00Z)
    write_job(kdir, "job2", "B", "2026-11-12T10:00:00Z", 4000.0)
    write_job(kdir, "job3", "E", "2026-11-12T12:00:00Z", 6000.0)
    q = {"jobs": [{"job": "job2", "arm": "B", "evals": "200:image", "seed": 3407, "out": "runs/b"},
                  {"job": "job4", "arm": "B", "evals": "200:image", "seed": 2027, "out": "runs/b2"},
                  {"job": "job5", "arm": "C", "evals": "200:image", "seed": 2027, "out": "runs/c2"}]}
    (tmp_path / "runs/b").mkdir(parents=True)
    (tmp_path / "runs/b/summary.json").write_text("{}")
    qp = tmp_path / "q.json"; qp.write_text(json.dumps(q))
    now = quota.parse_time("2026-11-13T00:00:00Z")
    res = quota.check(kdir, qp, tmp_path, now, quota_hours=30, running="job4")
    assert [s["job"] for s in res["sessions_in_window"]] == ["job2", "job3"]
    assert res["used_hours"] == round(10000 / 3600, 2)
    assert res["running_estimate_hours"] == round(9000 / 3600, 2)   # longest past B session, any date
    assert res["next"] == "job5" and res["next_estimate_hours"] == round(9000 / 3600, 2)  # no C yet -> overall max
    assert res["verdict"] == "ok"
    res = quota.check(kdir, qp, tmp_path, now, quota_hours=7, running="job4")
    assert res["verdict"] == "wait"


def test_real_logs_parse_when_present():
    kdir = ROOT / "runs/kaggle"
    if not (kdir / "job13" / "trace-job13.log").exists():
        return  # runs/kaggle is gitignored; nothing to check on a fresh clone
    by_job = {s["job"]: s for s in quota.sessions(kdir)}
    assert by_job["job13"]["arm"] == "E" and by_job["job13"]["start_utc"] == "2026-11-12T19:01:55Z"
    assert 6000 < by_job["job13"]["session_seconds"] < 7000


def test_kernel_without_job_start_or_with_lowercase_log_still_counts(tmp_path):
    import os
    kdir = tmp_path / "runs/kaggle"
    write_job(kdir, "job2", "B", "2026-11-12T10:00:00Z", 4000.0)
    d = kdir / "jobV0"; d.mkdir()
    (d / "trace-jobv0.log").write_text(json.dumps([{"stream_name": "stdout", "time": 13000.0, "data": "x"}]))
    (d / "jobV0.log").write_text("VGCURE-PREPARED {}\n")
    t = quota.parse_time("2026-11-12T20:00:00Z").timestamp()
    os.utime(d / "trace-jobv0.log", (t, t))
    qp = tmp_path / "q.json"; qp.write_text(json.dumps({"jobs": [
        {"job": "job4", "arm": "B", "evals": "200:image", "seed": 2027, "out": "runs/b2"}]}))
    res = quota.check(kdir, qp, tmp_path, quota.parse_time("2026-11-13T00:00:00Z"), starts={})
    by_job = {s["job"]: s for s in res["sessions_in_window"]}
    assert by_job["jobV0"] == {"job": "jobV0", "arm": None, "start_utc": "2026-11-12T20:00:00Z", "session_seconds": 13000.0}
    assert res["used_hours"] == round(17000 / 3600, 2)
    assert res["next_estimate_hours"] == round(4000 / 3600, 2)      # same-arm estimate ignores the arm-less kernel


def test_week_resets_on_saturday_utc(tmp_path):
    assert quota.week_start(quota.parse_time("2026-11-14T10:00:00Z")) == quota.parse_time("2026-11-14T00:00:00Z")
    assert quota.week_start(quota.parse_time("2026-11-13T23:59:59Z")) == quota.parse_time("2026-11-07T00:00:00Z")
    kdir = tmp_path / "runs/kaggle"
    write_job(kdir, "job2", "B", "2026-11-13T22:00:00Z", 4000.0)   # Friday evening: previous week
    write_job(kdir, "job3", "E", "2026-11-14T01:00:00Z", 6000.0)
    d = kdir / "jobV1"; d.mkdir()
    (d / "trace-jobv1.log").write_text(json.dumps([{"stream_name": "stdout", "time": 5000.0, "data": "x"}]))
    qp = tmp_path / "q.json"; qp.write_text(json.dumps({"jobs": []}))
    res = quota.check(kdir, qp, tmp_path, quota.parse_time("2026-11-14T12:00:00Z"),
                      starts={"jobV1": "2026-11-13T23:00:00Z"})   # recorded push time wins over the file mtime
    assert [s["job"] for s in res["sessions_in_window"]] == ["job3"]
    assert res["week_start_utc"] == "2026-11-14T00:00:00Z"


def test_other_project_sessions_count_toward_used(tmp_path):
    kdir = tmp_path / "runs/kaggle"
    write_job(kdir, "job2", "B", "2026-11-12T10:00:00Z", 4000.0)
    q = {"jobs": [{"job": "job4", "arm": "B", "evals": "200:image", "seed": 2027, "out": "runs/b2"}]}
    qp = tmp_path / "q.json"; qp.write_text(json.dumps(q))
    others = [{"kernel": "u/other-a", "start_utc": "2026-11-12T11:00:00Z", "session_seconds": 1800.0},
              {"kernel": "u/other-b", "start_utc": "2026-11-05T11:00:00Z", "session_seconds": 99999.0}]  # previous week
    now = quota.parse_time("2026-11-13T00:00:00Z")
    res = quota.check(kdir, qp, tmp_path, now, quota_hours=30, others=others)
    assert res["other_sessions_in_window"] == 1 and res["other_used_hours"] == 0.5
    assert res["used_hours"] == round(5800 / 3600, 2)
    assert res["next_estimate_hours"] == round(4000 / 3600, 2)   # other kernels never feed TRACE estimates
    # 4000 + 1800 used + 4000 next = 2.72 h fits 4.8 h - 2 h margin only without the other kernel
    assert quota.check(kdir, qp, tmp_path, now, quota_hours=4.3, others=[])["verdict"] == "ok"
    assert quota.check(kdir, qp, tmp_path, now, quota_hours=4.3, others=others)["verdict"] == "wait"


def test_other_usage_parsing_and_merge():
    import other_usage
    assert other_usage.session_seconds('[{"time": 1.5, "data": "a"}, {"time": 442.2, "data": "b"}]') == 442.2
    assert other_usage.session_seconds("no times") is None
    assert other_usage.to_utc("2026-11-20 08:53:41.173000") == "2026-11-20T08:53:41Z"
    old = [{"kernel": "u/a", "start_utc": "2026-11-20T08:00:00Z", "session_seconds": 10.0}]
    new = [{"kernel": "u/a", "start_utc": "2026-11-20T08:00:00Z", "session_seconds": 99.0},
           {"kernel": "u/a", "start_utc": "2026-11-20T09:00:00Z", "session_seconds": 20.0}]
    assert [e["session_seconds"] for e in other_usage.merge(old, new)] == [10.0, 20.0]
    for e in other_usage.load():
        assert set(e) >= {"kernel", "start_utc", "session_seconds"} and not e["kernel"].split("/")[-1].startswith("trace-")
