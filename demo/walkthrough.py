"""CPU walk-through of the benchmark's verifier: one root graph, its twins and its repair task.

No model is involved. Every "candidate response" below is written by this script to show how the
scorer treats it (an exact answer, the opposite answer, a sentence instead of JSON, an edit that
is not allowed, ...). They are illustrations, not model outputs, and nothing here measures a model.

The verdicts come from trace_vr.metrics.score_sample, the same function every evaluation in this
repository uses. On generated data it checks a response against the generator's true graph. For an
arbitrary uploaded picture no true graph is known, so nothing like this verdict could be claimed.

    PYTHONPATH=src python demo/walkthrough.py manifests/smoke.json dev [--root 0]

Locked test splits are refused, as everywhere else.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from trace_vr import dataset, metrics, tasks

MAIN_TASKS = (tasks.REACHABILITY, tasks.CRITICAL_EDGE, tasks.REPAIR)
PROSE = {tasks.REACHABILITY: "Yes, there is a path.", tasks.CRITICAL_EDGE: "Yes, the connection would be lost.",
         tasks.REPAIR: "Add one edge between the two nodes."}


def _pair_id(sample, role):
    for p in sample["pairs"]:
        if p["kind"] == "critical" and p["role"] == role:
            return p["pair_id"]
    return None


def twin_of(samples: List[Dict[str, Any]], base: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The answer-flipping twin: same instruction, an image whose graph differs by one edge."""
    pid = _pair_id(base, "base")
    for s in samples:
        if pid is not None and _pair_id(s, "variant") == pid:
            return s
    return None


def pick_root(samples: List[Dict[str, Any]], index: int = 0) -> Dict[str, Dict[str, Any]]:
    """The index-th root graph (file order) that has all three main tasks with a twin each.

    Selection rule: position only; no response or score is looked at."""
    groups: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        if s["task"] in MAIN_TASKS and _pair_id(s, "base") and twin_of(samples, s) is not None:
            groups.setdefault(s["graph_group_id"], {}).setdefault(s["task"], s)
    full = [g for g in groups.values() if all(t in g for t in MAIN_TASKS)]
    if not 0 <= index < len(full):
        raise SystemExit("root index must be between 0 and %d" % (len(full) - 1))
    return full[index]


def verdict(sample: Dict[str, Any], text: Optional[str]) -> Dict[str, Any]:
    res = metrics.score_sample(sample, text)
    out = {"correct": bool(res["correct"]), "format_ok": bool(res["format_ok"]), "error": res.get("error")}
    if sample["task"] == tasks.REPAIR and res["format_ok"]:
        out.update({k: res.get(k) for k in ("status_correct", "legal", "valid", "minimal")})
    return out


def _edit(op, a, b):
    return tasks.format_target({"status": "edited", "edits": [{"op": op, "src": a, "dst": b}]})


def _repair_candidates(sample):
    """Hand-written repair responses, each kept only if the scorer itself gives it the named verdict."""
    truth, q = sample["private_truth"], sample["public_input"]["query"]
    nodes, edges = truth["nodes"], [tuple(e) for e in truth["edges"]]
    allowed = [tuple(e) for e in q["allowed_additions"]]
    singles = [_edit("add_edge", a, b) for a, b in allowed if (a, b) not in edges]
    singles += [_edit("remove_edge", a, b) for a, b in edges]
    absent = [(a, b) for a in nodes for b in nodes if a != b and (a, b) not in edges]
    wanted = [
        ("a legal edit that does not reach the goal", singles, "constraints_not_satisfied"),
        ("an addition that is not on the allowed list", [_edit("add_edge", a, b) for a, b in absent if (a, b) not in allowed],
         "addition_not_allowed"),
        ("removing an edge that is not in the image", [_edit("remove_edge", a, b) for a, b in absent], "edge_not_present"),
    ]
    out = [("claims nothing needs to change", tasks.format_target({"status": "unchanged", "edits": []})),
           ("claims the goal is impossible", tasks.format_target({"status": "impossible", "edits": []}))]
    out = [c for c in out if json.loads(c[1])["status"] != truth["status"]]
    for label, pool, error in wanted:
        hit = next((t for t in pool if verdict(sample, t)["error"] == error), None)
        if hit is not None:
            out.append((label, hit))
    if truth["status"] == "edited" and int(q["max_edits"]) == 1 and edges:
        two = list(sample["training_target"]["edits"]) + [{"op": "remove_edge", "src": edges[0][0], "dst": edges[0][1]}]
        out.append(("the right edit plus a second one (over the edit budget)",
                    tasks.format_target({"status": "edited", "edits": two})))
    return out


def candidates(sample: Dict[str, Any]) -> List[tuple]:
    exact = tasks.format_target(sample["training_target"])
    out = [("the exact answer", exact)]
    if sample["task"] == tasks.REPAIR:
        out += _repair_candidates(sample)
    else:
        out.append(("the opposite answer", tasks.format_target({"answer": not sample["training_target"]["answer"]})))
    out.append(("a sentence instead of the JSON object", PROSE[sample["task"]]))
    return out


def section(samples: List[Dict[str, Any]], base: Dict[str, Any]) -> Dict[str, Any]:
    twin = twin_of(samples, base)
    exact = tasks.format_target(base["training_target"])
    return {
        "task": base["task"], "instruction": base["public_input"]["instruction"],
        "base": {"sample_id": base["sample_id"], "image": base["image_path"],
                 "rows": [{"label": lab, "text": t, **verdict(base, t)} for lab, t in candidates(base)]},
        # The twin gets the same instruction on a one-edge-different image; the base's exact answer is wrong there.
        "twin": {"sample_id": twin["sample_id"], "image": twin["image_path"],
                 "rows": [{"label": "the base image's exact answer, given for the twin image", "text": exact, **verdict(twin, exact)},
                          {"label": "the twin's own exact answer", "text": tasks.format_target(twin["training_target"]),
                           **verdict(twin, tasks.format_target(twin["training_target"]))}]},
    }


def walkthrough(samples: List[Dict[str, Any]], index: int = 0) -> List[Dict[str, Any]]:
    root = pick_root(samples, index)
    return [section(samples, root[t]) for t in MAIN_TASKS]


def _line(row):
    if row["correct"]:
        mark = "CORRECT"
    elif not row["format_ok"]:
        mark = "WRONG (format failure: %s)" % str(row["error"]).replace("format:", "")
    else:
        mark = "WRONG (%s)" % (row["error"] or "answer differs from the true graph's")
    extra = ""
    if "legal" in row:
        extra = "  [legal=%s goal met=%s minimal=%s]" % (row["legal"], row["valid"], row["minimal"])
    return "  %-58s %s\n      %s%s" % (row["label"], mark, row["text"], extra)


def render(sec: Dict[str, Any]) -> str:
    lines = ["=== %s ===" % sec["task"], sec["instruction"], "",
             "image %s (sample %s); candidate responses are hand-written, not model outputs:" % (sec["base"]["image"], sec["base"]["sample_id"])]
    lines += [_line(r) for r in sec["base"]["rows"]]
    lines += ["", "twin image %s (sample %s), same instruction, one edge different:" % (sec["twin"]["image"], sec["twin"]["sample_id"])]
    lines += [_line(r) for r in sec["twin"]["rows"]]
    return "\n".join(lines)


def side_by_side(data_dir: Path, sec: Dict[str, Any]):
    """Base and twin image next to each other (PIL image)."""
    from PIL import Image
    a, b = (Image.open(Path(data_dir) / sec[k]["image"]).convert("RGB") for k in ("base", "twin"))
    sheet = Image.new("RGB", (a.width + b.width + 16, max(a.height, b.height)), "white")
    sheet.paste(a, (0, 0))
    sheet.paste(b, (a.width + 16, 0))
    return sheet


def load(manifest: str, split: str):
    dataset.check_split_allowed(split, "eval")
    man, data_dir, by_split, _ = dataset.load_dataset(Path(manifest))
    if split not in by_split:
        raise SystemExit("split %r not in %s (has: %s)" % (split, manifest, ", ".join(sorted(by_split))))
    return man, data_dir, by_split[split]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest"); ap.add_argument("split"); ap.add_argument("--root", type=int, default=0)
    args = ap.parse_args(argv)
    _, _, samples = load(args.manifest, args.split)
    print("\n\n".join(render(s) for s in walkthrough(samples, args.root)))


if __name__ == "__main__":
    main()
