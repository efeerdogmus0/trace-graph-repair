"""Choose the reference baseline (B or C) on dev from a pooled seed table (CPU only).

Usage:
  python scripts/reference_baseline.py <seeds_dev.json> [--out-json f] [--out-md f]

Rule: src/trace_vr/seeds.py:choose_reference. Run once, before the locked test is opened."""
import json, sys
from pathlib import Path
from trace_vr import seeds

args = sys.argv[1:]
outs = {}
for key in ("--out-json", "--out-md"):
    if key in args:
        i = args.index(key); outs[key] = Path(args[i + 1]); del args[i:i + 2]
if len(args) != 1:
    raise SystemExit(__doc__)
table = json.loads(Path(args[0]).read_text())
res = seeds.choose_reference(table)
res["source"] = args[0]
lines = ["# Reference baseline (dev)", "",
         "Rule: %s. Source: `%s` (split %s)." % (res["rule"], res["source"], res["split"]), "",
         "| Candidate | Macro per seed | Pooled macro |", "|---|---|---|"]
for c in res["candidates"]:
    per = ", ".join("%s: %.3f" % (s, v) for s, v in sorted(res["per_seed_macro"][c].items()))
    lines.append("| %s | %s | %.3f |" % (c, per, res["pooled_macro"][c]))
lines += ["", "Chosen reference: **%s**." % res["chosen"]]
for c, g in (res.get("gaps_vs_chosen") or {}).items():
    if g:
        lines.append("%s - %s pooled: %.3f [%.3f, %.3f]." % (c, res["chosen"], g["point"], g["lo"], g["hi"]))
md = "\n".join(lines) + "\n"
if "--out-json" in outs:
    outs["--out-json"].write_text(json.dumps(res, indent=1) + "\n")
if "--out-md" in outs:
    outs["--out-md"].write_text(md)
print(md, end="")
