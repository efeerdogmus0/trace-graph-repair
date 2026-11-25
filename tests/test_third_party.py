"""THIRD_PARTY.md is the generator's output, and what it says about the repository holds."""
import hashlib, importlib.util, json, re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("third_party", ROOT / "scripts/third_party.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _tracked(mod):
    try:
        return mod.tracked()
    except Exception:
        pytest.skip("not a git checkout")


def test_file_is_generated_and_linked():
    mod = _mod()
    files = _tracked(mod)
    assert (ROOT / mod.OUT).read_text() == mod.render(mod.facts(files=files))
    assert mod.OUT in files and "`THIRD_PARTY.md`" in (ROOT / "README.md").read_text()
    assert "`THIRD_PARTY.md`" in (ROOT / "RELEASE_CHECKLIST.md").read_text()


def test_fonts_are_the_only_third_party_files():
    mod = _mod()
    files = _tracked(mod)
    fonts = [f for f in files if f.startswith("assets/fonts/")]
    assert len(fonts) == 7 and sum(f.endswith(".ttf") for f in fonts) == 6
    text = (ROOT / mod.OUT).read_text()
    for f in fonts:
        assert "| `%s` | `%s` |" % (f, hashlib.sha256((ROOT / f).read_bytes()).hexdigest()) in text
    # the kernels check the same six hashes against matplotlib's copies
    kernel = (ROOT / "scripts/kaggle/make_kernel.py").read_text()
    assert '(ROOT / "assets" / "fonts").glob("*.ttf")' in kernel
    licence = (ROOT / "assets/fonts/LICENSE_DEJAVU").read_text()
    for phrase in ("to reproduce and distribute the", "shall\nbe included in all copies",
                   "may be sold by itself", "only if the fonts\nare renamed"):
        assert phrase in licence


def test_new_file_types_and_external_paths_are_refused():
    mod = _mod()
    files = _tracked(mod)
    for extra in ("vendor/lib.so", "docs/photo.jpg", "src/OTHER_LICENSE", "docs/Extra.ttf", "external/vgcure_meta/metrics.py"):
        with pytest.raises(SystemExit):
            mod.facts(files=files + [extra])
    assert "/external/" in (ROOT / ".gitignore").read_text().split()
    assert mod.suffix(".gitignore") == ".gitignore" and mod.suffix("a/b/LICENSE_DEJAVU") == "" and mod.suffix("a.b/c.tar.gz") == ".gz"


def test_vgcure_record_holds_ids_and_hashes_only():
    mod = _mod()
    record = json.loads((ROOT / mod.VGCURE_RECORD).read_text())
    assert sorted(record) == mod.VGCURE_KEYS
    assert all(re.fullmatch(r"vgcure/[a-z_]+/\d+/\d+", s) for s in record["sample_ids"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in record["source_files_sha256"].values())
    facts = mod.facts(files=_tracked(mod))
    assert record["revision"] == facts["vgcure_revision"]
    assert "revision `%s` | none stated" % facts["vgcure_revision"] in (ROOT / mod.OUT).read_text()
    assert "Licence: none stated" in (ROOT / "docs/external_eval.md").read_text()
    # no other tracked file is a VGCure data file
    assert not [f for f in _tracked(mod) if "vgcure" in f.lower() and f.endswith((".zip", ".jsonl", ".png"))]


def test_base_model_row_matches_the_code_and_the_cards():
    mod = _mod()
    facts = mod.facts(files=_tracked(mod))
    source = (ROOT / "src/trace_vr/model.py").read_text()
    assert 'BASE_ID = "%s"' % facts["base_id"] in source and 'BASE_REVISION = "%s"' % facts["base_revision"] in source
    text = (ROOT / mod.OUT).read_text()
    assert "| Base model `%s` | revision `%s` | `apache-2.0` |" % (facts["base_id"], facts["base_revision"]) in text
    card = (ROOT / "docs/model_cards/pilot3_B.md").read_text()
    assert facts["base_id"] in card and facts["base_revision"] in card


def test_every_pinned_package_is_listed():
    mod = _mod()
    facts = mod.facts(files=_tracked(mod))
    text = (ROOT / mod.OUT).read_text()
    assert len(facts["lock"]) == 6 and set(facts["pins"]) == {"transformers", "peft"}
    for name, version in list(facts["lock"].items()) + list(facts["pins"].items()):
        assert "| %s | %s | %s |" % (name, version, mod.PACKAGES[(name, version)]) in text
    assert set(mod.ROLE) == {n for n, _ in mod.PACKAGES}


def test_recorded_licences_match_installed_metadata():
    mod = _mod()
    compared = 0
    for (name, version), licence in mod.PACKAGES.items():
        found = mod.declared_licence(name)
        if found is None or found[0] != version:
            continue                      # another environment; the record is for the pinned version
        assert found[1] == licence, name
        compared += 1
    if compared == 0:
        pytest.skip("none of the recorded versions is installed here")


def test_fonts_equal_matplotlib_copies_when_that_version_is_installed():
    mod = _mod()
    found = mod.declared_licence("matplotlib")
    if found is None or found[0] != "3.9.4":
        pytest.skip("matplotlib 3.9.4 not installed")
    bundled = Path(importlib.util.find_spec("matplotlib").origin).parent / "mpl-data/fonts/ttf"
    for f in sorted((ROOT / "assets/fonts").iterdir()):
        assert (bundled / f.name).read_bytes() == f.read_bytes(), f.name


def test_root_licence_is_own_but_a_nested_one_is_not():
    mod = _mod()
    files = list(_tracked(mod))
    mod.facts(ROOT, files=files + ["LICENSE"] if "LICENSE" not in files else files)   # root LICENSE accepted
    with pytest.raises(SystemExit):
        mod.facts(ROOT, files=files + ["src/LICENSE"])
