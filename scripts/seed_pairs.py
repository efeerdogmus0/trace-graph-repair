"""Twin-pair and answer-bias counts per arm and seed, from saved flag files (CPU only).

Usage:
  python scripts/seed_pairs.py <manifest> LABEL@SEED=flags.json [...] [--out-json f] [--out-md f]

The descriptive counts that paper/sections/analysis.tex fixed before the runs (both right / one right / both wrong per
answer-flipping twin pair; how often the model said true next to how often true was right), repeated for every
pre-registered seed. Counts are also summed over seeds: every seed scores the same pairs, so the summed rate is the mean
of the per-seed rates. No interval, no reading rule and no reference arm: the decision uses the macro table
(scripts/seed_table.py) only. Inputs are checked exactly as in seed_table.py."""
import json, sys
from pathlib import Path
from trace_vr import analysis, arms as arms_mod, bootstrap, dataset, seeds

CELLS = ("both_right", "one_right", "both_wrong")


def parse(argv):
    args = list(argv)
    opts = {"out-json": None, "out-md": None}
    for key in list(opts):
        flag = "--" + key
        if flag in args:
            i = args.index(flag); opts[key] = args[i + 1]; del args[i:i + 2]
    if len(args) < 2:
        raise SystemExit(__doc__)
    entries = []
    for a in args[1:]:
        if "=" not in a or "@" not in a.split("=", 1)[0]:
            raise SystemExit("expected LABEL@SEED=path, got %r" % a)
        head, path = a.split("=", 1)
        label, s = head.rsplit("@", 1)
        s = int(s)
        if s not in seeds.PREREGISTERED_SEEDS:
            raise SystemExit("seed %d is not pre-registered" % s)
        if (label, s) in [(e[0], e[1]) for e in entries]:
            raise SystemExit("duplicate %s@%d" % (label, s))
        entries.append((label, s, Path(path)))
    return args[0], entries, opts


def build(manifest, entries):
    man, _, by_split, _ = dataset.load_dataset(Path(manifest))
    key, loaded = None, []
    for label, s, path in entries:
        p = json.loads(path.read_text())
        if p["content_digest"] != man["content_digest"]:
            raise SystemExit("%s comes from a different dataset build" % path)
        k = (p["split"], p["eval_root_graphs"], p.get("sample_set", "main"))
        if k[2] != "main":
            raise SystemExit("%s is not a main-sample run (twins are main samples)" % path)
        if key is None:
            key = k
        elif k != key:
            raise SystemExit("%s covers different samples %s than %s" % (path, k, key))
        if p.get("missing_responses"):
            raise SystemExit("%s has %d missing responses" % (path, p["missing_responses"]))
        loaded.append((label, s, path, p))
    rows = arms_mod.eval_samples(by_split[key[0]], key[1])
    table = {"split": key[0], "root_graphs": key[1], "sample_set": key[2], "content_digest": man["content_digest"],
             "note": "descriptive counts; summed = sum over seeds (same pairs in every seed)", "arms": [], "files": []}
    by_label = {}
    for label, s, path, p in loaded:
        flags = bootstrap.decode_flags(rows, p)
        fmt = bootstrap.decode_flags(rows, p, "format_ok")
        by_label.setdefault(label, {})[s] = {"pairs": analysis.pair_table(rows, flags),
                                             "answer_bias": analysis.answer_bias(rows, flags, fmt)}
        table["files"].append({"label": label, "seed": s, "file": str(path),
                               "predictions_sha256": p.get("predictions_sha256")})
    for label, per_seed in by_label.items():
        summed_pairs, summed_bias = {}, {}
        for s in sorted(per_seed):
            for task, cell in per_seed[s]["pairs"].items():
                tot = summed_pairs.setdefault(task, {"pairs": 0, **{c: 0 for c in CELLS}})
                for c in ("pairs",) + CELLS:
                    tot[c] += cell[c]
            for task, e in per_seed[s]["answer_bias"].items():
                tot = summed_bias.setdefault(task, {"n": 0, "no_answer": 0, "said_true": 0, "truth_true": 0})
                answered = e["n"] - e["no_answer"]
                tot["n"] += e["n"]; tot["no_answer"] += e["no_answer"]
                # counts are integers; rates were built from them, so rounding recovers them exactly
                tot["said_true"] += 0 if e["said_true_rate"] is None else int(round(e["said_true_rate"] * answered))
                tot["truth_true"] += int(round(e["truth_true_rate"] * e["n"]))
        for tot in summed_pairs.values():
            tot["both_right_rate"] = tot["both_right"] / tot["pairs"]
        for tot in summed_bias.values():
            answered = tot["n"] - tot["no_answer"]
            tot["said_true_rate"] = (tot["said_true"] / answered) if answered else None
            tot["truth_true_rate"] = tot["truth_true"] / tot["n"]
        table["arms"].append({"label": label, "seeds": sorted(per_seed),
                              "per_seed": {str(s): per_seed[s] for s in sorted(per_seed)},
                              "summed": {"pairs": summed_pairs, "answer_bias": summed_bias}})
    return table


def markdown(t):
    lines = ["Split `%s`, %d root graphs, main samples. Descriptive counts from the tracked flag files; no intervals. "
             "\"Both right\" = both images of an answer-flipping twin pair answered correctly." % (t["split"], t["root_graphs"]),
             "", "## Twin pairs", "",
             "| Arm | Task | Pairs per seed | Both right per seed | Summed both / one / none right | Both-right rate |",
             "|---|---|---|---|---|---|"]
    for a in t["arms"]:
        for task, tot in a["summed"]["pairs"].items():
            per = ", ".join("%s: %d" % (s, a["per_seed"][s]["pairs"][task]["both_right"]) for s in a["per_seed"])
            lines.append("| %s | %s | %d | %s | %d / %d / %d | %.3f |" % (
                a["label"], task, a["per_seed"][str(a["seeds"][0])]["pairs"][task]["pairs"], per,
                tot["both_right"], tot["one_right"], tot["both_wrong"], tot["both_right_rate"]))
    lines += ["", "## Answer bias (yes/no tasks)", "",
              "| Arm | Task | Said true per seed | Said true, summed | True was right | No answer (summed) |",
              "|---|---|---|---|---|---|"]
    for a in t["arms"]:
        for task, tot in a["summed"]["answer_bias"].items():
            per = ", ".join("%s: %s" % (s, "n/a" if a["per_seed"][s]["answer_bias"][task]["said_true_rate"] is None
                                        else "%.3f" % a["per_seed"][s]["answer_bias"][task]["said_true_rate"])
                            for s in a["per_seed"])
            lines.append("| %s | %s | %s | %s | %.3f | %d of %d |" % (
                a["label"], task, per, "n/a" if tot["said_true_rate"] is None else "%.3f" % tot["said_true_rate"],
                tot["truth_true_rate"], tot["no_answer"], tot["n"]))
    return "\n".join(lines) + "\n"


def main(argv):
    manifest, entries, opts = parse(argv)
    t = build(manifest, entries)
    md = markdown(t)
    for key, text in (("out-json", json.dumps(t, indent=1) + "\n"), ("out-md", md)):
        if opts[key]:
            Path(opts[key]).parent.mkdir(parents=True, exist_ok=True)
            Path(opts[key]).write_text(text)
    print(md, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
