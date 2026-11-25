"""Command line entry point: python -m trace_vr <command>."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def _doctor(args) -> int:
    import PIL

    from . import GENERATOR_VERSION, __version__, families, layout, render
    from .rng import Rng
    from .styles import STYLES

    print("trace_vr %s (%s)" % (__version__, GENERATOR_VERSION))
    print("python %s on %s" % (platform.python_version(), platform.platform()))
    print("pillow %s" % PIL.__version__)
    ok = True
    for style in STYLES.values():
        try:
            render.load_font(style.font_file, style.font_size)
        except Exception as exc:  # noqa: BLE001 - report any font problem
            ok = False
            print("FONT PROBLEM (%s): %s" % (style.style_id, exc))
    if ok:
        print("fonts: ok (%s)" % (render.assets_dir() / "fonts"))
    rng = Rng("doctor")
    g = families.generate("sparse_random", 7, rng.child("g"))
    lay, _ = layout.find_layout(g, STYLES["a"], rng.child("layout"), 512)
    if lay is None:
        ok = False
        print("layout self-test: FAILED")
    else:
        labels = list("ABCDEFG")
        img = render.draw_diagram(lay.pos, labels, lay.routes, STYLES["a"], 512)
        nodes = {labels[i]: list(lay.pos[i]) for i in range(7)}
        edges = [{"src": labels[u], "dst": labels[v], "points": lay.routes[(u, v)]} for (u, v) in sorted(lay.routes)]
        problems = render.image_checks(img, STYLES["a"], nodes, edges)
        ok = ok and not problems
        print("render self-test: %s" % ("ok" if not problems else problems))
    try:
        import torch  # type: ignore

        print("torch %s, cuda=%s" % (torch.__version__, torch.cuda.is_available()))
    except Exception:  # noqa: BLE001
        print("torch: not installed (fine for the CPU data core)")
    return 0 if ok else 1


def _generate(args) -> int:
    from . import dataset

    cfg = dataset.load_config(Path(args.config))
    out = Path(args.out) if args.out else Path("data") / cfg["name"]
    manifest_path = Path(args.manifest) if args.manifest else Path("manifests") / ("%s.json" % cfg["name"])
    t0 = time.time()
    manifest = dataset.build_dataset(cfg, out, manifest_path)
    total = sum(s["samples"] for s in manifest["splits"].values())
    images = sum(s["images"] for s in manifest["splits"].values())
    roots = sum(s["roots"] for s in manifest["splits"].values())
    print("built '%s': %d root graphs, %d images, %d samples in %.1fs" % (cfg["name"], roots, images, total, time.time() - t0))
    print("content digest %s" % manifest["content_digest"])
    print("images digest  %s" % manifest["images_digest"])
    print("manifest: %s" % manifest_path)
    return 0


def _validate(args) -> int:
    from . import validate

    t0 = time.time()
    report = validate.validate(Path(args.manifest), check_pixels=not args.no_pixels)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("%d checks, %d errors, %d warnings (%.1fs)" % (report["total_checks"], report["error_count"],
                                                         len(report["warnings"]), time.time() - t0))
    for e in report["errors"][:20]:
        print("  ERROR " + e)
    for w in report["warnings"]:
        print("  warning " + w)
    for split, tasks_ in report["text_only_shortcuts"].items():
        for task, entry in tasks_.items():
            rest = ", ".join("%s=%.2f" % (k, v) for k, v in entry.items() if k != "samples")
            print("  text-only %s/%s (n=%d): %s" % (split, task, entry["samples"], rest))
    return 0 if report["ok"] else 1


def _gallery(args) -> int:
    from . import gallery

    manifest = Path(args.manifest)
    out = Path(args.out) if args.out else Path("reports") / manifest.stem / "gallery"
    for path in gallery.render_gallery(manifest, out):
        print(path)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="trace_vr", description="TRACE CPU data core")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="print the environment and run a small self-test").set_defaults(fn=_doctor)
    g = sub.add_parser("generate", help="build a dataset from a config")
    g.add_argument("--config", required=True)
    g.add_argument("--out", help="data directory (default: data/<name>)")
    g.add_argument("--manifest", help="manifest path (default: manifests/<name>.json)")
    g.set_defaults(fn=_generate)
    v = sub.add_parser("validate-data", help="re-check a dataset from its files")
    v.add_argument("--manifest", required=True)
    v.add_argument("--report", help="write the full JSON report here")
    v.add_argument("--no-pixels", action="store_true", help="skip reading images back")
    v.set_defaults(fn=_validate)
    r = sub.add_parser("render-gallery", help="write contact sheets and an HTML page")
    r.add_argument("--manifest", required=True)
    r.add_argument("--out")
    r.set_defaults(fn=_gallery)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
