"""Print the next Kaggle job of scripts/kaggle/queue.json and the exact commands for it (no network).

Usage: python scripts/kaggle/next_job.py [--queue scripts/kaggle/queue.json] [--root DIR] [--json]

An entry is done when <root>/<out>/summary.json exists (written by collect.py). The first entry that is
not done is the one to poll (if already pushed) or to push. The autopilot checks
`kaggle kernels status` itself; this script only keeps the order and the command lines in one place,
so an hourly run cannot skip a seed or reuse a job name.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (3407, 2027, 9173)
OWNER = "yiitefeerdomu"
KAGGLE = "~/.kaggle/venv/bin/kaggle"


def load(queue_path):
    q = json.loads(Path(queue_path).read_text())
    jobs, outs = [j["job"] for j in q["jobs"]], [j["out"] for j in q["jobs"]]
    if len(set(jobs)) != len(jobs) or len(set(outs)) != len(outs):
        raise SystemExit("queue has a duplicate job name or output dir")
    for j in q["jobs"]:
        if j["seed"] not in SEEDS:
            raise SystemExit("%s: seed %s is not pre-registered" % (j["job"], j["seed"]))
        default = "runs/pilot3_" + j["arm"] + ("" if j["seed"] == 3407 else "_s%d" % j["seed"])
        j["default_out"] = default
    return q["jobs"]


def commands(j):
    seed = "" if j["seed"] == 3407 else " --seed %d" % j["seed"]
    mode = j["evals"].split(",")[0].split(":")[1]
    out = "" if j["out"] == j["default_out"] else " --out " + j["out"]
    slug = "%s/trace-%s" % (OWNER, j["job"])
    return [
        "python scripts/kaggle/make_kernel.py %s %s %s%s --from-head" % (j["job"], j["arm"], j["evals"], seed),
        "%s kernels push -p dist/kaggle/%s --accelerator NvidiaTeslaT4" % (KAGGLE, j["job"]),
        "%s kernels status %s" % (KAGGLE, slug),
        "%s kernels output %s -p runs/kaggle/%s" % (KAGGLE, slug, j["job"]),
        "PYTHONPATH=src python scripts/kaggle/collect.py %s %s%s --mode %s%s --budget" % (j["job"], j["arm"], seed, mode, out),
    ]


def main(argv):
    args = list(argv)
    queue, root, as_json = ROOT / "scripts/kaggle/queue.json", ROOT, "--json" in args
    if "--queue" in args:
        queue = Path(args[args.index("--queue") + 1])
    if "--root" in args:
        root = Path(args[args.index("--root") + 1])
    jobs = load(queue)
    done = [j["job"] for j in jobs if (root / j["out"] / "summary.json").exists()]
    pending = [j for j in jobs if j["job"] not in done]
    nxt = pending[0] if pending else None
    result = {"done": done, "pending": [j["job"] for j in pending],
              "next": None if nxt is None else {"job": nxt["job"], "arm": nxt["arm"], "seed": nxt["seed"],
                                                "out": nxt["out"], "commands": commands(nxt)}}
    if as_json:
        print(json.dumps(result, indent=1))
    elif nxt is None:
        print("queue finished: %d jobs collected" % len(done))
    else:
        print("done: %s" % (", ".join(done) or "-"))
        print("next: %s arm %s seed %d -> %s" % (nxt["job"], nxt["arm"], nxt["seed"], nxt["out"]))
        print("\n".join(result["next"]["commands"]))
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
