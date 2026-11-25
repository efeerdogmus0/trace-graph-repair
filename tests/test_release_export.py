import subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import release_export as rx  # noqa: E402


def test_identity_rejects_machine_local_email():
    with pytest.raises(SystemExit):
        rx.identity_env("Someone <someone@Some-MacBook.local>")
    with pytest.raises(SystemExit):
        rx.identity_env("no email here")
    env = rx.identity_env(rx.DEFAULT_AUTHOR)
    assert env["GIT_AUTHOR_EMAIL"] == env["GIT_COMMITTER_EMAIL"] == "trace-authors@example.invalid"


def test_snapshot_drops_private_files_and_uses_explicit_author(tmp_path):
    if not all((ROOT / p).exists() for p in rx.PRIVATE):
        pytest.skip("running inside a release snapshot: the private files are already left out")
    dst, kept = rx.export(tmp_path)
    assert kept > 0
    for p in rx.PRIVATE:
        assert not (dst / p).exists(), p
    who = subprocess.run(["git", "log", "-1", "--format=%an <%ae>|%cn <%ce>"], cwd=dst, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert who == "%s|%s" % (rx.DEFAULT_AUTHOR, rx.DEFAULT_AUTHOR)


def test_commit_into_updates_a_clone_without_rewriting(tmp_path):
    if not all((ROOT / p).exists() for p in rx.PRIVATE):
        pytest.skip("running inside a release snapshot")
    first, _ = rx.export(tmp_path / "a")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(first), str(clone)], check=True)
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True, check=True).stdout.strip()
    assert rx.commit_into(clone, first, rx.DEFAULT_AUTHOR) is None            # identical tree: nothing to commit
    second, _ = rx.export(tmp_path / "b")
    (second / "README.md").write_text("changed\n"); (second / "LICENSE").unlink()
    assert rx.commit_into(clone, second, rx.DEFAULT_AUTHOR)
    log = subprocess.run(["git", "log", "--format=%H %an <%ae>"], cwd=clone, capture_output=True, text=True, check=True).stdout.split("\n")
    assert log[1].split()[0] == base and rx.DEFAULT_AUTHOR in log[0]           # parent kept, explicit author
    assert (clone / "README.md").read_text() == "changed\n" and not (clone / "LICENSE").exists()
