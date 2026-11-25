"""Shared test helpers.

Some tests read generated datasets (data/ is not tracked; it is rebuilt from configs/).
They call `require_data(name)`: on a fresh checkout without that dataset the test is
skipped with the command that builds it. Set TRACE_REQUIRE_DATA=1 to turn those skips
into failures (scripts/clean_check.py does, so a clean-checkout check cannot pass by
skipping).
"""
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def data_built(name):
    """True if every split file listed in manifests/<name>.json exists under its data_dir."""
    manifest = ROOT / "manifests" / (name + ".json")
    if not manifest.exists():
        return False
    m = json.loads(manifest.read_text())
    data_dir = (manifest.parent / m["data_dir"]).resolve()
    files = [info[k] for info in m["splits"].values() for k in ("samples_file", "images_file") if k in info]
    return bool(files) and all((data_dir / f).exists() for f in files)


def require_data(*names):
    missing = [n for n in names if not data_built(n)]
    if not missing:
        return
    how = "; ".join("python -m trace_vr generate --config configs/data_%s.yaml" % n for n in missing)
    msg = "dataset(s) %s not built (run: %s)" % (", ".join(missing), how)
    if os.environ.get("TRACE_REQUIRE_DATA") == "1":
        pytest.fail(msg)
    pytest.skip(msg)
