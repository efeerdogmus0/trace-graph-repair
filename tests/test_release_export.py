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
