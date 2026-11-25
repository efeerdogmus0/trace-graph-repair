"""Build the arXiv source package of the paper. Builds a local tarball only; it uploads nothing.

usage: python scripts/arxiv_package.py --out <dir> [--compile]

arXiv compiles the source in one flat directory and refuses paths that climb out of it, so the
generated result files the paper reads from ../reports/final/ are copied into final/ and every
reference is rewritten. Full-line comments (internal notes) are dropped; they never typeset.
Writes <out>/trace-arxiv/ and <out>/trace-arxiv.tar.gz and prints the tarball's sha256.

Blockers (exit 2, the package is still written for review):
  - the title or the first line still says "draft";
  - a generated input the paper waits for is missing (e.g. tables/vgcure_final.tex);
  - a \\input or \\includegraphics target is missing from the package, or a path still climbs out.
--compile also builds the package alone with tectonic (nice 19) in a scratch copy and blocks on
missing files or undefined references.
"""
import gzip, hashlib, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER, FINAL = ROOT / "paper", ROOT / "reports" / "final"
WAITED_FOR = ["tables/vgcure_final.tex"]          # inputs behind \IfFileExists that a finished paper needs
TECTONIC = Path.home() / ".local/tectonic/tectonic"


def strip_comments(text):
    return "".join(l for l in text.splitlines(keepends=True) if not l.lstrip().startswith("%"))


def build(out):
    pkg = out / "trace-arxiv"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "final").mkdir(parents=True)
    files = {}
    for p in sorted(PAPER.rglob("*")):
        rel = p.relative_to(PAPER)
        if p.is_file() and p.suffix in {".tex", ".pdf", ".png"}:
            files[str(rel)] = p
    for p in sorted(FINAL.glob("*.tex")):
        files["final/" + p.name] = p
    for rel, src in files.items():
        dst = pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".tex":
            text = strip_comments(src.read_text()).replace("../reports/final/", "final/")
            if rel == "main.tex" and "\\pdfoutput" not in text:
                text = "\\pdfoutput=1\n" + text          # arXiv: force pdflatex (PDF/PNG figures)
            dst.write_text(text)
        else:
            shutil.copyfile(src, dst)
    return pkg


def blockers(pkg):
    out = []
    main_src = (PAPER / "main.tex").read_text()
    title = re.search(r"\\title\{(.*?)\}\s*\n\\author", main_src, re.S)
    if title and "draft" in title.group(1).lower():
        out.append("title still says draft")
    if "draft" in main_src.splitlines()[0].lower():
        out.append("main.tex first line still marks a draft")
    for rel in WAITED_FOR:
        if not (pkg / rel).exists():
            out.append("missing generated input %s" % rel)
    defs = {}
    for t in pkg.rglob("*.tex"):
        src = t.read_text()
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\[\d\]\{", src):
            depth, i = 1, m.end()
            while depth and i < len(src):
                depth += {"{": 1, "}": -1}.get(src[i], 0) if src[i - 1] != "\\" else 0
                i += 1
            defs[m.group(1)] = src[m.end():i - 1]
    for t in sorted(pkg.rglob("*.tex")):
        text = t.read_text()
        if ".." in "".join(re.findall(r"\\(?:input|includegraphics(?:\[[^]]*\])?|IfFileExists)\{([^}]*)\}", text)):
            out.append("%s: a path climbs out of the package" % t.relative_to(pkg))
        for target in re.findall(r"\\input\{([^}#]+)\}", text):
            if not ((pkg / target).exists() or (pkg / (target + ".tex")).exists()):
                if not any(target in w for w in WAITED_FOR):
                    out.append("%s: \\input{%s} not in package" % (t.relative_to(pkg), target))
        for target in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}#]+)\}", text):
            if not (pkg / target).exists():
                out.append("%s: figure %s not in package" % (t.relative_to(pkg), target))
        for macro, body in defs.items():   # \finaltable{name}... style macros with #1 in a path
            paths = re.findall(r"\\(?:IfFileExists|input|includegraphics(?:\[[^]]*\])?)\{([^{}]*#1[^{}]*)\}", body)
            for arg in re.findall(r"\\%s\{([^}]+)\}" % macro, text):
                for pth in {p.replace("#1", arg) for p in paths}:
                    if not ((pkg / pth).exists() or (pkg / (pth + ".tex")).exists()):
                        out.append("%s: \\%s{%s} needs %s" % (t.relative_to(pkg), macro, arg, pth))
    return sorted(set(out))


def compile_check(pkg):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "src"
        shutil.copytree(pkg, work)
        main_tex = work / "main.tex"   # tectonic is XeTeX, where \pdfoutput is undefined; arXiv runs pdflatex
        main_tex.write_text(main_tex.read_text().replace("\\pdfoutput=1\n", "", 1))
        r = subprocess.run(["nice", "-n", "19", str(TECTONIC), "-X", "compile", "main.tex", "--keep-logs"],
                           cwd=work, capture_output=True, text=True)
        log = (work / "main.log").read_text(errors="replace") if (work / "main.log").exists() else ""
        out = [] if r.returncode == 0 else ["tectonic exit %d: %s" % (r.returncode, r.stderr.strip()[-400:])]
        for pat, msg in ((r"File `[^']+' not found", "missing file"), (r"Reference `[^']+' on page \d+ undefined", "undefined reference"),
                         (r"There were undefined references", "undefined references")):
            out += ["%s: %s" % (msg, m) for m in sorted(set(re.findall(pat, log)))]
        m = re.search(r"Output written on main\.\w+ \((\d+) pages", log)
        if not (work / "main.pdf").exists():
            out.append("no PDF written")
        return out, int(m.group(1)) if m else 0


def main(argv):
    if "--out" not in argv:
        raise SystemExit(__doc__)
    out = Path(argv[argv.index("--out") + 1]).resolve()
    if out == ROOT or ROOT in out.parents and not str(out.relative_to(ROOT)).startswith("dist"):
        raise SystemExit("write the package outside the repo or under dist/")
    out.mkdir(parents=True, exist_ok=True)
    pkg = build(out)
    tar = out / "trace-arxiv.tar.gz"
    with open(tar, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz, \
            tarfile.open(fileobj=gz, mode="w") as tf:   # byte-identical for identical sources
        for p in sorted(pkg.rglob("*")):
            if p.is_file():
                info = tf.gettarinfo(str(p), arcname=str(p.relative_to(pkg)))
                info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                with open(p, "rb") as fh:
                    tf.addfile(info, fh)
    block = blockers(pkg)
    if "--compile" in argv:
        more, pages = compile_check(pkg)
        block += more
        print("standalone compile: %d pages" % pages)
    n = sum(1 for p in pkg.rglob("*") if p.is_file())
    print("package %s: %d files, %.1f kB, sha256 %s" % (tar, n, tar.stat().st_size / 1024,
                                                         hashlib.sha256(tar.read_bytes()).hexdigest()))
    for b in block:
        print("BLOCKER: " + b)
    return 2 if block else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
