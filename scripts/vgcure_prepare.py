"""Fetch the pre-registered VGCure subset and its 500 images (run on Kaggle; ~3.1 GB, never on the Mac).

usage: python scripts/vgcure_prepare.py <work_dir> [--local <dir with understanding/ reasoning/ zips/>]

Downloads the 20 test JSONL files and 10 directed image zips at the pinned revision (or uses --local),
checks every zip against its LFS sha256 (vgcure.ZIP_SHA256), rebuilds the subset with vgcure.build_subset
and refuses unless its digest equals the selection record (reports/external/vgcure_subset.json, or the
exploratory dev record named by TRACE_VGCURE_RECORD), extracts only the selected
images, letterboxed to the dev canvas (512x512, aspect kept, white pad), to <work_dir>/VGCure/images/<dir>/<k>.png, deletes each zip after use, and writes
<work_dir>/subset_full.json. Prints VGCURE-PREPARED.
"""
import hashlib, json, os, shutil, sys, zipfile
from pathlib import Path

from trace_vr import vgcure

ROOT = Path(__file__).resolve().parents[1]
CANVAS = 512   # dev/test images are 512x512 (configs/data_pilot.yaml image_size); fixed before any VGCure output


def letterbox(raw: bytes, out: Path, canvas: int = CANVAS):
    """Scale the longest side to `canvas` (aspect kept, LANCZOS), centre on a white square. Returns the original size."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    k = canvas / max(w, h)
    small = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    bg = Image.new("RGB", (canvas, canvas), "white")
    bg.paste(small, ((canvas - small.size[0]) // 2, (canvas - small.size[1]) // 2))
    bg.save(out, format="PNG")
    return w, h


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fetch(work: Path, local: Path = None) -> Path:
    src = work / "src"
    for hub, rel in vgcure.hub_files().items():
        dst = src / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if local is not None:
            shutil.copyfile(local / rel, dst)
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(vgcure.HF_DATASET, hub, repo_type="dataset", revision=vgcure.HF_REVISION,
                                local_dir=str(work / "hub"))
            shutil.move(p, dst)
    return src


def main(argv):
    work = Path(argv[0]).resolve()
    local = Path(argv[argv.index("--local") + 1]).resolve() if "--local" in argv else None
    record = json.loads((ROOT / os.environ.get("TRACE_VGCURE_RECORD", vgcure.TEST_RECORD)).read_text())
    src = fetch(work, local)
    sub = vgcure.build_subset(src, record["per_source"], record.get("skip", 0))
    if sub["subset_sha256"] != record["subset_sha256"] or sub["source_files_sha256"] != record["source_files_sha256"]:
        raise SystemExit("VGCure subset differs from the pre-registered record; refusing to evaluate")
    by_source, sizes = {}, []
    for s in sub["samples"]:
        by_source.setdefault(s["source"], set()).add(s["image"])
    for source, images in sorted(by_source.items()):
        z = src / ("zips/%s_directed_akg_images.zip" % source)
        if images and all((work / im).exists() for im in images):
            continue
        if sha_file(z) != vgcure.ZIP_SHA256[source]:
            raise SystemExit("%s: zip sha256 differs from the pinned revision" % z.name)
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            for im in sorted(images):
                out = work / im
                out.parent.mkdir(parents=True, exist_ok=True)
                raw = zf.read(vgcure.image_member(names, im))
                sizes.append(letterbox(raw, out))
        if local is None:
            os.remove(z)
    (work / "subset_full.json").write_text(json.dumps(sub, sort_keys=True) + "\n")
    n_img = sum(len(v) for v in by_source.values())
    big = [max(w, h) for w, h in sizes]
    print("VGCURE-PREPARED " + json.dumps({"n": len(sub["samples"]), "images": n_img, "sha256": sub["subset_sha256"],
                                           "canvas": CANVAS, "orig_longest_side_min": min(big) if big else None,
                                           "orig_longest_side_max": max(big) if big else None}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
