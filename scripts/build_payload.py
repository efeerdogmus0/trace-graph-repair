"""Pack the code the Colab VM needs into dist/payload.tar.gz and dist/index.html (base64 in a <pre>).

The page is served by the `payload` launch config and moved into the notebook via window.name
(RUNBOOK, Colab section). Data is not shipped: the VM regenerates it and checks the content digest.
Usage: python scripts/build_payload.py   -> prints member count, size and sha256."""
import base64, gzip, hashlib, io, sys, tarfile
from pathlib import Path

INCLUDE = ["pyproject.toml", "src/trace_vr", "scripts", "configs", "manifests", "tests", "assets"]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def members(root: Path):
    for item in INCLUDE:
        p = root / item
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        for f in files:
            rel = f.relative_to(root)
            if not SKIP_PARTS & set(rel.parts) and f.suffix != ".pyc":
                yield f, rel.as_posix()


def build(root: Path) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for f, name in members(root):
            info = tar.gettarinfo(str(f), arcname=name)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(f, "rb") as fh:
                tar.addfile(info, fh)
    return buf.getvalue()


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    data = build(root)
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    (dist / "payload.tar.gz").write_bytes(data)
    b64 = base64.b64encode(data).decode()
    (dist / "index.html").write_text("<!doctype html><title>payload</title><pre id=p>%s</pre>" % b64)
    print("members %d  bytes %d  sha256 %s" % (sum(1 for _ in members(root)), len(data), hashlib.sha256(data).hexdigest()))
