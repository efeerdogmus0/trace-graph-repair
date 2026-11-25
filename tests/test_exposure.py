"""Exposure accounting (protocol open item 3): every arm sees the same number of samples,
root groups and images; the committed table matches the current manifest."""
import importlib.util
import re
from pathlib import Path

import pytest

from conftest import require_data
from trace_vr import arms, dataset

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "pilot.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("exposure", ROOT / "scripts" / "exposure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def table():
    require_data("pilot")
    mod = _load_script()
    _, _, by_split, _ = dataset.load_dataset(MANIFEST)
    t = {arm: mod.exposure(by_split["train"], arm) for arm in arms.ARMS}
    mod.add_overlap(t)
    return t


def test_equal_samples_groups_images(table):
    first = table["D"]
    for arm, e in table.items():
        assert e["samples"] == first["samples"], arm
        assert e["root_groups"] == first["root_groups"], arm
        assert e["unique_images"] == e["samples"], arm  # one image per sample, no repeats
        assert e["image_tokens"] == e["samples"] * 256


def test_c_and_d2_use_exactly_d_images(table):
    for arm in ("C", "D2"):
        assert table[arm]["images_shared_with_D"] == table["D"]["unique_images"]
    # B's twins come from non-edit pairs: only root images (and few twins) coincide with D.
    assert table["B"]["root_groups"] <= table["B"]["images_shared_with_D"] < table["D"]["unique_images"]


def test_committed_table_is_current(table):
    doc = ROOT / "docs" / "exposure_pilot_train.md"
    text = doc.read_text(encoding="utf-8")
    assert dataset.load_dataset(MANIFEST)[0]["content_digest"] in text
    for arm, e in table.items():
        row = re.search(r"^\| %s \| (.*)$" % arm, text, re.M).group(1).split(" | ")
        assert [int(x) for x in row[:7]] == [
            e["samples"], e["root_groups"], e["unique_images"], e["images_shared_with_D"],
            e["image_tokens"], e["instruction_chars"], e["target_chars"]], arm
