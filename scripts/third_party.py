"""Inventory of third-party material: what is redistributed, what is only used, and what each licence
statement rests on.

usage: python scripts/third_party.py [--check]      -> writes / checks THIRD_PARTY.md

Brief section 23: review the licence of every piece of code, data, font, model and notebook separately
and add a third-party notice if needed. This script writes that notice. It does not decide anything
legal; for each component it records the source, the pinned version, whether a copy is in the
repository, and the evidence the licence statement rests on (a licence file that was read, package
metadata, a model card, or "none stated").

Computed from the repository on every run: the tracked files (font hashes, tracked file types, no
tracked path under external/), the base model id and revision (src/trace_vr/model.py), the VGCure
revision (src/trace_vr/vgcure.py), the locked package versions (requirements-lock.txt) and the kernel
pins (scripts/kaggle/make_kernel.py). Recorded by hand, with the date they were read: the declared
licences below. `declared_licence` reads the same field from an installed package, so the tests can
compare the record with the environment they run in. `--check` exits 1 if THIRD_PARTY.md is stale.
"""
import hashlib, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = "THIRD_PARTY.md"
READ_ON = "2026-11-16"

# (package, version) -> licence as declared in that version's package metadata, read READ_ON from the
# two local environments. This is the publisher's declaration, not a reading of the licence text.
PACKAGES = {
    ("pillow", "11.3.0"): "MIT-CMU",
    ("PyYAML", "6.0.3"): "MIT License",
    ("pytest", "8.4.2"): "MIT License",
    ("networkx", "3.2.1"): "BSD License",
    ("numpy", "2.0.2"): "BSD License",
    ("matplotlib", "3.9.4"): "Python Software Foundation License",
    ("transformers", "5.22.2"): "Apache 2.0 License",
    ("peft", "0.21.2"): "Apache Software License",
    ("torch", "2.15.0"): "Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND "
                         "BSL-1.0 AND ISC AND MIT AND Zlib",
}
ROLE = {"pillow": "renders every image", "PyYAML": "reads configs", "pytest": "test suite",
        "networkx": "tests only: independent graph oracle", "numpy": "dependency of matplotlib; figures",
        "matplotlib": "report figures; source of the font files",
        "transformers": "model loading, generation (GPU runs)", "peft": "LoRA adapters (GPU runs)",
        "torch": "training and inference (GPU runs)"}
# torch is not pinned by the kernels: they use the version of the Kaggle / Colab image.
UNPINNED = {"torch": "local `.venv-model` only; the Kaggle and Colab runs used the image's torch, whose version "
                     "was not logged for the kernels collected so far (`docs/environment.md`)"}

# Read READ_ON from the pinned snapshot in the local Hugging Face cache (not tracked here).
BASE_CARD_LICENCE = "apache-2.0"
BASE_LICENCE_FILE_SHA256 = "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
FONT_VERSION = "Version 2.35"          # name table, id 5, of all six files (fontTools, READ_ON)
FONT_SOURCE = "matplotlib 3.9.4, `mpl-data/fonts/ttf/` (byte-identical, licence file included; compared " + READ_ON + ")"

# Tracked file types. Text types are written in this project; a new type must be placed here by hand.
OWN_TEXT = {".py", ".md", ".json", ".jsonl", ".tex", ".yaml", ".toml", ".txt", ".sh", ".ipynb", ".xml", ".svg",
            ".gitignore"}
OWN_RENDERED = {".png", ".pdf"}        # figures, gallery cards, contact sheets drawn by this repository's code
THIRD_PARTY_FILES = {".ttf", ""}       # "" = assets/fonts/LICENSE_DEJAVU
OWN_FILES = {"LICENSE"}               # exact paths of own files without a suffix (project licence, MIT)
# keys of the only tracked file derived from VGCure: ids, counts and hashes, no question text, no image
VGCURE_RECORD = "reports/external/vgcure_subset.json"
VGCURE_KEYS = ["counts", "dataset", "n_samples", "per_source", "revision", "sample_ids", "source_files_sha256",
               "subset_sha256"]


def tracked(root: Path = ROOT):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return sorted(f for f in out.decode().split("\0") if f)


def suffix(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name.startswith(".") and name.count(".") == 1:
        return name
    return "." + name.rsplit(".", 1)[1] if "." in name else ""


def declared_licence(name: str):
    """(version, declared licence) of an installed package, or None. Same rule the record was made with."""
    import importlib.metadata as md
    try:
        meta = md.metadata(name)
    except md.PackageNotFoundError:
        return None
    lic = meta.get("License-Expression")
    if not lic:
        cls = [c.split(" :: ")[-1] for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
        lic = "; ".join(cls) if cls else (meta.get("License") or "").strip().split("\n")[0]
    return meta["Version"], lic


def _const(path: str, name: str, root: Path) -> str:
    m = re.search(r'^%s\s*=\s*"([^"]+)"' % re.escape(name), (root / path).read_text(), re.M)
    if not m:
        raise SystemExit("%s: no constant %s" % (path, name))
    return m.group(1)


def facts(root: Path = ROOT, files=None):
    files = tracked(root) if files is None else files
    fonts = [f for f in files if f.startswith("assets/fonts/")]
    files_x = [f for f in files if f not in OWN_FILES]
    unknown = sorted(f for f in files_x if suffix(f) not in OWN_TEXT | OWN_RENDERED | THIRD_PARTY_FILES)
    misplaced = sorted(f for f in files_x if suffix(f) in THIRD_PARTY_FILES and not f.startswith("assets/fonts/"))
    if unknown or misplaced:
        raise SystemExit("tracked files without a place in the inventory: %s" % ", ".join(unknown + misplaced))
    external = [f for f in files if f.startswith("external/")]
    if external:
        raise SystemExit("external/ must stay untracked (VGCure states no licence): %s" % ", ".join(external))
    if "assets/fonts/LICENSE_DEJAVU" not in fonts:
        raise SystemExit("font licence file missing")
    lock = {}
    for line in (root / "requirements-lock.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.strip().split("==")
            lock[name] = version
    pins = dict(p.split("==") for p in _const("scripts/kaggle/make_kernel.py", "PINS", root).split())
    for name, version in list(lock.items()) + list(pins.items()):
        if (name, version) not in PACKAGES:
            raise SystemExit("no licence record for %s==%s" % (name, version))
    return {
        "fonts": [(f, hashlib.sha256((root / f).read_bytes()).hexdigest()) for f in fonts],
        "lock": lock, "pins": pins,
        "base_id": _const("src/trace_vr/model.py", "BASE_ID", root),
        "base_revision": _const("src/trace_vr/model.py", "BASE_REVISION", root),
        "vgcure_revision": _const("src/trace_vr/vgcure.py", "HF_REVISION", root),
    }


def render(f) -> str:
    L = ["# Third-party material", "",
         "Generated by `scripts/third_party.py` from the tracked files; do not edit by hand. Nothing is "
         "published. Licence statements below say what they rest on; none of them is legal advice. "
         "Licence fields were read on %s." % READ_ON, "",
         "## Redistributed in this repository", "",
         "Only the DejaVu fonts. Source: %s. Font version in the files' name table: `%s`. Licence text: "
         "`assets/fonts/LICENSE_DEJAVU` (Bitstream Vera licence with the DejaVu and Arev additions; read in "
         "full: it allows copying and redistribution, also with other software, if the copyright and "
         "permission notices are kept, forbids selling the fonts by themselves, and requires modified fonts "
         "to be renamed; the files here are unmodified). Any copy of the "
         "repository or of a data build that includes the fonts must keep that file." % (FONT_SOURCE, FONT_VERSION), "",
         "| File | sha256 |", "|---|---|"]
    L += ["| `%s` | `%s` |" % row for row in f["fonts"]]
    L += ["",
          "The Kaggle kernels do not carry the fonts: they link matplotlib's bundled copies and check these "
          "hashes (`scripts/kaggle/make_kernel.py`).", "",
          "The tracked PNG / PDF files (figures, gallery cards, contact sheets) were drawn by this "
          "repository's code. They show text set in DejaVu as pixels or embedded glyph outlines; they contain "
          "no file from another project. Every other tracked file is text written in this project "
          "(the brief `TRACE_ASTRA_MASTER_PROMPT.md` is the owner's).", "",
          "## Used, not redistributed", "",
          "| Component | Pinned version | Licence statement | Rests on | In this repository |", "|---|---|---|---|---|",
          "| Base model `%s` | revision `%s` | `%s` | `license:` field of the model card and the `LICENSE` file "
          "(Apache License 2.0 text, sha256 `%s...`) in the pinned snapshot | nothing; downloaded by "
          "`trace_vr.model.load` |" % (f["base_id"], f["base_revision"], BASE_CARD_LICENCE,
                                       BASE_LICENCE_FILE_SHA256[:16]),
          "| VGCure dataset `AAAndyZ/VGCure` | revision `%s` | none stated | dataset card and GitHub repository, "
          "API `license: null`, checked 2026-11-13 (`docs/external_eval.md`) | `%s` only: sample ids, counts "
          "and file hashes; no image, no question text |" % (f["vgcure_revision"], VGCURE_RECORD),
          "| VGCure / MCDGraph code `github.com/AAAndy-Zhu/VGCure` | as downloaded 2026-11-13 | none stated | "
          "same check | nothing. Arm C is our own adaptation written from the paper (`src/trace_vr/arms.py`); "
          "the VGCure scoring rules are re-implemented (`src/trace_vr/vgcure.py`). Their scorer and `src/` + "
          "`scripts/` share no line of 30 or more characters (compared %s) |" % READ_ON,
          "", "Consequences, as long as these statements stand:", "",
          "- Adapters trained here are LoRA weights for the base model above; an adapter release names the "
          "base model and its revision and ships no base weights (`docs/model_cards/`).",
          "- No VGCure file (image, question, answer, code) may be added to the repository or to a release. "
          "`external/` is git-ignored and this script refuses a tracked path under it. Results on VGCure are "
          "reported as numbers with sample ids; users download the dataset from its own page.", "",
          "## Python packages (installed by the user, not redistributed)", "",
          "| Package | Version | Declared licence (package metadata) | Used for | Pinned by |", "|---|---|---|---|---|"]
    for (name, version), lic in PACKAGES.items():
        if f["lock"].get(name) == version:
            where = "`requirements-lock.txt`"
        elif f["pins"].get(name) == version:
            where = "kernel pins (`scripts/kaggle/make_kernel.py`)"
        else:
            where = UNPINNED[name]
        L.append("| %s | %s | %s | %s | %s |" % (name, version, lic, ROLE[name], where))
    L += ["",
          "The licence column is what each package's own metadata declares for that version, not a reading of "
          "its licence text. Packages these pull in (tokenizers, safetensors, huggingface-hub and others) are "
          "not listed; nothing of them is copied here.", "",
          "## Tools", "",
          "Tectonic 0.17.0 builds the paper PDF; Kaggle and Colab ran the GPU jobs. No file of theirs is in the "
          "repository; their licences were not reviewed.", "",
          "## Project licence", "",
          "Chosen by the owner 2026-11-21: code MIT (`LICENSE`); data, images, figures, text and adapter weights "
          "CC BY 4.0 (`LICENSE-DATA.md`).", "",
          "## Open", "",
          "- VGCure states no licence. Only the authors can change that; the plan above (numbers and ids only) "
          "does not depend on it.", ""]
    return "\n".join(L)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = render(facts())
    path = ROOT / OUT
    if "--check" in argv:
        if not path.exists() or path.read_text() != text:
            print("%s is stale: rerun scripts/third_party.py" % OUT)
            return 1
        print("%s up to date" % OUT)
        return 0
    path.write_text(text)
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
