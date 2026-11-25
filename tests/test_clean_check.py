"""require_data skips on a fresh checkout, fails under TRACE_REQUIRE_DATA=1; clean_check refuses a used workdir."""
import importlib.util
from pathlib import Path

import pytest

import conftest

ROOT = Path(__file__).resolve().parents[1]


def test_missing_dataset_skips_with_build_command(monkeypatch):
    monkeypatch.delenv("TRACE_REQUIRE_DATA", raising=False)
    assert not conftest.data_built("no_such_dataset")
    with pytest.raises(pytest.skip.Exception, match="configs/data_no_such_dataset.yaml"):
        conftest.require_data("no_such_dataset")


def test_missing_dataset_fails_in_strict_mode(monkeypatch):
    monkeypatch.setenv("TRACE_REQUIRE_DATA", "1")
    with pytest.raises(pytest.fail.Exception, match="not built"):
        conftest.require_data("no_such_dataset")


def test_data_built_reads_the_manifest_file_list(tmp_path, monkeypatch):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "data" / "x").mkdir(parents=True)
    (tmp_path / "manifests" / "x.json").write_text(
        '{"data_dir": "../data/x", "splits": {"dev": {"samples_file": "dev.s", "images_file": "dev.i"}}}')
    monkeypatch.setattr(conftest, "ROOT", tmp_path)
    assert not conftest.data_built("x")
    (tmp_path / "data" / "x" / "dev.s").write_text("")
    assert not conftest.data_built("x")
    (tmp_path / "data" / "x" / "dev.i").write_text("")
    assert conftest.data_built("x")


def test_clean_check_refuses_nonempty_workdir(tmp_path):
    spec = importlib.util.spec_from_file_location("clean_check", ROOT / "scripts" / "clean_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "left_over").write_text("x")
    with pytest.raises(SystemExit, match="must be empty"):
        mod.main(["--workdir", str(tmp_path)])
