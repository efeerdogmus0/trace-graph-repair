import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_audit", ROOT / "scripts" / "release_audit.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)


def _run(files):
    return ra.audit(list(files), lambda f: files[f], lambda f: len(files[f] or b"x"))


def test_secret_and_home_path_fail():
    fake = "KGAT_" + "a1B2" * 6
    res = _run({"a.py": "x = 1\ntoken = '%s'\n" % fake, "b.md": "see /Users/someone/x.txt\n"})
    assert not res["ok"]
    assert res["secrets"] == [("a.py", "kaggle_token", 2)]
    assert res["local_paths"] == [("b.md", "home_path", 1)]


def test_review_items_do_not_fail():
    res = _run({"r.md": "kernel " + "yiitefeerdo" + "mu/trace-job1\nmail me@" + "host.org\n",
                "RUNBOOK.md": "notes\n", "img.png": None})
    assert res["ok"]
    assert {(n, i) for _, n, i in res["review"]} == {("kaggle_account", 1), ("email", 2)}
    assert res["internal"] == ["RUNBOOK.md"] and res["binary"] == 1


def test_shared_kaggle_account_is_a_review_item():
    res = _run({"n.md": "owner " + "efeerdog" + "mus1/trace-jobF07\n"})
    assert res["ok"]
    assert [(n, i) for _, n, i in res["review"]] == [("kaggle_account", 1)]


def test_ci_runner_paths_and_example_mail_are_ignored():
    res = _run({"w.yml": "cd /home/runner/work/x\nuser@example.com\n"})
    assert res["ok"] and not res["review"]


def test_repository_has_no_secrets_or_home_paths():
    files = ra.tracked_files()

    def read(f):
        p = ROOT / f
        if not p.exists():
            return ""
        b = p.read_bytes()
        return None if b"\0" in b[:8192] else b.decode("utf-8", errors="replace")

    res = ra.audit(files, read, lambda f: (ROOT / f).stat().st_size if (ROOT / f).exists() else 0)
    assert res["secrets"] == [] and res["local_paths"] == [], (res["secrets"], res["local_paths"])
