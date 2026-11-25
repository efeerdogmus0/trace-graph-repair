"""Score end-to-end arms on drawings arm E reads exactly vs the rest (see src/trace_vr/perception.py).

Usage:
  python scripts/perception_split.py <manifest> <extraction_<split>.jsonl> --ref LABEL LABEL=flags.json ...
         [--out-json f.json] [--out-md f.md] [--boot 2000]
All flag files must cover the same split/roots/sample set (main) and dataset build as the extraction run."""
import json, sys
from pathlib import Path
from trace_vr import arms, bootstrap, dataset, perception


def main(argv):
    args = list(argv)
    opts = {"ref": None, "out-json": None, "out-md": None, "boot": "2000"}
    for k in list(opts):
        if "--" + k in args:
            i = args.index("--" + k); opts[k] = args[i + 1]; del args[i:i + 2]
    if len(args) < 3 or opts["ref"] is None:
        raise SystemExit(__doc__)
    manifest, ext_path, pairs = Path(args[0]), Path(args[1]), [a.split("=", 1) for a in args[2:]]
    man, _, by_split, _ = dataset.load_dataset(manifest)
    key, flags = None, {}
    for label, path in pairs:
        p = json.loads(Path(path).read_text())
        if p["content_digest"] != man["content_digest"]:
            raise SystemExit("%s comes from a different dataset build" % path)
        if p.get("sample_set", "main") != "main" or p.get("missing_responses"):
            raise SystemExit("%s: need a complete main-set flag file" % path)
        k = (p["split"], p["eval_root_graphs"])
        if key not in (None, k):
            raise SystemExit("%s covers different samples %s than %s" % (path, k, key))
        key = k
        rows = arms.eval_samples(by_split[k[0]], k[1])
        flags[label] = bootstrap.decode_flags(rows, p)
    if opts["ref"] not in flags:
        raise SystemExit("--ref %s is not one of the arms" % opts["ref"])
    dataset.check_split_allowed(key[0], "eval")
    readable = perception.readable_images(rows, perception.load_extraction(ext_path))
    t = perception.split_table(rows, readable, flags, opts["ref"], int(opts["boot"]))
    t.update({"split": key[0], "root_graphs": key[1], "extraction": str(ext_path), "content_digest": man["content_digest"]})
    lines = ["Arm E extraction `%s`: %d of %d images read exactly. Readable subset %d samples / %d roots; unreadable %d / %d."
             % (ext_path, t["readable_images"], t["images"], t["subsets"]["readable"]["samples"], t["subsets"]["readable"]["root_graphs"],
                t["subsets"]["unreadable"]["samples"], t["subsets"]["unreadable"]["root_graphs"]), "",
             "| Arm | macro readable | macro unreadable | vs %s readable | vs %s unreadable | reading |" % (opts["ref"], opts["ref"]), "|---|---|---|---|---|---|"]

    def f(e):
        return "-" if not e else "%.3f [%.3f, %.3f]" % (e["point"], e["lo"], e["hi"])
    for label, e in t["arms"].items():
        r, u = e["readable"] or {}, e["unreadable"] or {}
        lines.append("| %s | %s | %s | %s | %s | %s |" % (label, f(r.get("acc")), f(u.get("acc")), f(r.get("vs_ref")), f(u.get("vs_ref")), e.get("reading", "ref")))
    md = "\n".join(lines) + "\n"
    if opts["out-json"]:
        Path(opts["out-json"]).write_text(json.dumps(t, indent=1, sort_keys=True) + "\n")
    if opts["out-md"]:
        Path(opts["out-md"]).write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
