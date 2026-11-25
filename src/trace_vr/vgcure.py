"""Pre-registered external evaluation subset on VGCure (Zhu et al., ACL 2025).

VGCure's test files give one image per graph and question/claim strings, not the edge list, so the
adapted subset only uses questions whose answer is fixed by the file itself:

* ``edge_exists`` (true)  from ``neighbor_query.fc_sup``:  "<v> is a out-neighbors of <u> ..." / True
* ``edge_exists`` (false) from ``neighbor_query.fc_ref``:  same claim with a non-neighbour / False
* ``reachability``        from ``connective_path_query.qa``: "Is there a path from <u> to <v>?" / Yes|No

Selection rule (fixed 2026-11-13 before any model output on VGCure): every directed source, graphs
sorted by numeric image id, the first ``per_source`` graphs that carry all three items. Nothing here
looks at model predictions. Instructions are our own templates (tasks.render_instruction), so this
is an adapted protocol, not VGCure's official one.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import tasks

SOURCES = (
    "ba", "dblp", "dbpedia", "er", "openflights", "pubchemqc", "sbm", "sfn", "social_network", "star",
)
HF_DATASET = "AAAndyZ/VGCure"
HF_REVISION = "06d8cf2c10ffc6fb6d4c518eda1275b6eee2f56c"
PER_SOURCE = 50
# Exploratory dev subset (owner instruction 2026-11-14, DECISIONS.md): the DEV_PER_SOURCE usable graphs after
# the pre-registered PER_SOURCE ones, so the two subsets share no graph. Iteration happens only on this one.
DEV_PER_SOURCE = 40
DEV_RECORD = "reports/external/vgcure_dev_subset.json"
TEST_RECORD = "reports/external/vgcure_subset.json"

_NB = re.compile(r"^(\S+) is a (out|in)-neighbors of (\S+) in this graph\.")
_PATH = re.compile(r"^Is there a path from (\S+) to (\S+)\?")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        t = rec["task"]
        rec["task"] = ast.literal_eval(t) if isinstance(t, str) else t
        out.append(rec)
    return out


def _image_key(rec: Dict[str, Any]) -> int:
    return int(Path(rec["graph_image"]).stem)


def _edge_item(entry: Dict[str, Any], key: str) -> Dict[str, Any]:
    fc = entry.get(key) or {}
    m = _NB.match(fc.get("claim", ""))
    if not m or fc.get("label") not in ("True", "False"):
        raise ValueError("unexpected neighbor claim: %r" % (fc,))
    a, kind, b = m.group(1), m.group(2), m.group(3)
    # "a is a out-neighbor of b": edge b->a;  "a is a in-neighbor of b": edge a->b.
    src, dst = (b, a) if kind == "out" else (a, b)
    return {"type": tasks.EDGE_EXISTS, "src": src, "dst": dst, "answer": fc["label"] == "True",
            "_anchor": b, "_other": a, "_kind": kind}


def _reach_item(entry: Dict[str, Any]) -> Dict[str, Any]:
    qa = entry.get("qa") or {}
    m = _PATH.match(qa.get("question", ""))
    ans = qa.get("answer", "")
    if not m or not (ans.startswith("Yes") or ans.startswith("No")):
        raise ValueError("unexpected path question: %r" % (qa,))
    return {"type": tasks.REACHABILITY, "src": m.group(1), "dst": m.group(2), "answer": ans.startswith("Yes")}


def _check_neighbors(nb: Dict[str, Any], pos: Dict[str, Any], neg: Dict[str, Any]) -> None:
    """The claims must agree with the neighbour list in the same record."""
    qa = nb.get("qa") or {}
    m = re.match(r"^Which nodes are (out|in)-neighbors of (\S+) in this graph\?", qa.get("question", ""))
    if not m:
        raise ValueError("unexpected neighbor question: %r" % (qa,))
    listed = {x.strip() for x in qa.get("answer", "").strip("[]").split(",") if x.strip()}
    kind, anchor = m.group(1), m.group(2)
    # fc_ref sometimes flips the direction ("E4 is a in-neighbor of E1" when E1->E4 exists); that is a
    # valid negative which the list cannot confirm, so only same-direction negatives are checked here.
    same_dir_neg_listed = neg["_kind"] == kind and neg["_other"] in listed
    if pos["_kind"] != kind or pos["_anchor"] != anchor or neg["_anchor"] != anchor \
            or pos["_other"] not in listed or same_dir_neg_listed:
        raise ValueError("neighbor claims disagree with the listed neighbours: %r" % (nb,))


def select_source(source: str, understanding: Sequence[Dict], reasoning: Sequence[Dict],
                  per_source: int = PER_SOURCE, skip: int = 0) -> List[Dict[str, Any]]:
    """Samples for one source; deterministic, independent of any model output."""
    by_img = {r["graph_image"]: r for r in reasoning}
    samples: List[Dict[str, Any]] = []
    graphs = 0
    for u in sorted(understanding, key=_image_key):
        if graphs >= per_source:
            break
        r = by_img.get(u["graph_image"])
        nb = u["task"].get("neighbor_query") or {}
        cp = (r or {}).get("task", {}).get("connective_path_query") or {}
        if not (r and nb.get("fc_sup") and nb.get("fc_ref") and cp.get("qa")):
            continue
        items = [_edge_item(nb, "fc_sup"), _edge_item(nb, "fc_ref"), _reach_item(cp)]
        _check_neighbors(nb, items[0], items[1])
        if skip > 0:   # usable graph that belongs to an earlier (disjoint) subset
            skip -= 1
            continue
        root = "vgcure/%s/%d" % (source, _image_key(u))
        for i, q in enumerate(items):
            query = {k: q[k] for k in ("type", "src", "dst")}
            samples.append({
                "sample_id": "%s/%d" % (root, i),
                "root_id": root,
                "source": source,
                "image": "VGCure/images/%s" % u["graph_image"],
                "task": q["type"],
                "query": query,
                "instruction": tasks.render_instruction(query),
                "answer": q["answer"],
                "negative_kind": None if i != 1 else ("same_direction" if q["_kind"] == items[0]["_kind"] else "reversed_edge"),
            })
        graphs += 1
    if graphs < per_source:
        raise ValueError("%s: only %d usable graphs" % (source, graphs))
    return samples


def build_subset(root: Path, per_source: int = PER_SOURCE, skip: int = 0) -> Dict[str, Any]:
    """root holds understanding/<s>_directed_graphs_test.json and reasoning/<same>."""
    root = Path(root)
    samples: List[Dict[str, Any]] = []
    files = {}
    for s in SOURCES:
        name = "%s_directed_graphs_test.json" % s
        up, rp = root / "understanding" / name, root / "reasoning" / name
        for p in (up, rp):
            files["%s/%s" % (p.parent.name, name)] = hashlib.sha256(p.read_bytes()).hexdigest()
        samples.extend(select_source(s, load_jsonl(up), load_jsonl(rp), per_source, skip))
    body = json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
    return {
        "dataset": HF_DATASET,
        "revision": HF_REVISION,
        "per_source": per_source,
        "skip": skip,
        "source_files_sha256": files,
        "subset_sha256": hashlib.sha256(body).hexdigest(),
        "samples": samples,
    }


def summary(subset: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in subset["samples"]:
        c = out.setdefault(s["source"], {"graphs": set(), "edge_true": 0, "edge_false": 0, "reach_yes": 0, "reach_no": 0})
        c["graphs"].add(s["root_id"])
        if s["task"] == tasks.EDGE_EXISTS:
            c["edge_true" if s["answer"] else "edge_false"] += 1
        else:
            c["reach_yes" if s["answer"] else "reach_no"] += 1
    for c in out.values():
        c["graphs"] = len(c["graphs"])
    return out


# ------------------------------------------------------------------ evaluation (M7, run once on Kaggle)
# LFS sha256 of each directed image zip at HF_REVISION (from the Hub tree listing, 2026-11-13).
ZIP_SHA256 = {
    "ba": "15bb4d471413c004874fec0018fd1745292d5d8b8288d33d1864773869e86a5b",
    "dblp": "e31e705c1cd86aeb91704c91d4104911f760832b1c78d42a64aff2c252a27d28",
    "dbpedia": "daa7d54db4f4fe15f3c4e0431b85f004733491ad8a879e931ca40830f707e597",
    "er": "15d57c241d236cbe95166c02179fca33c18fe909b85143d8cb87494bb71395df",
    "openflights": "a6fd490acd8698a9af366203ef4bf137c7a386dfbb8b90a4c20cd3ad4bc244e0",
    "pubchemqc": "c05dfc7be1da22a988b12ea5ca7e242e0c7a3d18581591d18f33d3f92ddc533a",
    "sbm": "58355851e692665809600737fecde836b8955242a10371719c2c971168a2fb04",
    "sfn": "0809449abe627cbeb082b4e440b6eccf38da72799b0877fb2ea5a4ce0838e4bf",
    "social_network": "092336d882e1d7e1bc26592e3642dff9bf152e08d9bd6e5aaf67a5c9784794e2",
    "star": "b764a8e0b31b350ae339c61e439cf399922fef46d172dd53c02f077d3b838ddd",
}


def hub_files() -> Dict[str, str]:
    """Hub path -> local relative path for everything the evaluation downloads (on Kaggle, not the Mac)."""
    out = {}
    for s in SOURCES:
        name = "%s_directed_graphs_test.json" % s
        for part in ("understanding", "reasoning"):
            out["VGCure/test_samples/%s/%s" % (part, name)] = "%s/%s" % (part, name)
        out["VGCure/images/%s_directed_akg_images.zip" % s] = "zips/%s_directed_akg_images.zip" % s
    return out


def image_member(names: Sequence[str], image: str) -> str:
    """The zip member for a subset image path 'VGCure/images/<dir>/<k>.png' (zips may or may not keep <dir>)."""
    folder, base = image.split("/")[-2], image.split("/")[-1]
    hits = [n for n in names if n == "%s/%s" % (folder, base) or n.endswith("/%s/%s" % (folder, base))]
    if not hits:
        hits = [n for n in names if n == base or n.endswith("/" + base)]
    if len(hits) != 1:
        raise ValueError("%s: %d zip members match" % (image, len(hits)))
    return hits[0]


def score_item(sample: Dict[str, Any], text: Optional[str]) -> Dict[str, Any]:
    """Strict JSON parse; correct iff the boolean equals the file's answer. A format failure is wrong."""
    parsed = tasks.parse_response(sample["task"], text)
    if not parsed.ok:
        return {"correct": False, "format_ok": False}
    return {"correct": parsed.value["answer"] == sample["answer"], "format_ok": True}


def _balanced(flags_by_class: Dict[bool, List[bool]]) -> Optional[float]:
    accs = [sum(v) / len(v) for v in flags_by_class.values() if v]
    return sum(accs) / len(accs) if accs else None


def evaluate(samples: Sequence[Dict[str, Any]], responses: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Scores fixed in docs/external_eval.md: edge accuracy, reachability balanced accuracy, macro = their mean."""
    per: Dict[str, Dict[str, Any]] = {}
    edge, reach = [], {True: [], False: []}
    fmt = []
    for s in samples:
        r = score_item(s, responses.get(s["sample_id"]))
        fmt.append(r["format_ok"])
        p = per.setdefault(s["source"], {"edge": [], "reach": {True: [], False: []}, "reversed_neg": []})
        if s["task"] == tasks.EDGE_EXISTS:
            edge.append(r["correct"]); p["edge"].append(r["correct"])
            if s.get("negative_kind") == "reversed_edge":
                p["reversed_neg"].append(r["correct"])
        else:
            reach[s["answer"]].append(r["correct"]); p["reach"][s["answer"]].append(r["correct"])
    mean = lambda v: sum(v) / len(v) if v else None
    edge_acc, reach_bal = mean(edge), _balanced(reach)
    rev = [x for p in per.values() for x in p["reversed_neg"]]
    return {
        "n": len(samples),
        "format_ok": mean(fmt),
        "edge_exists_accuracy": edge_acc,
        "reachability_balanced_accuracy": reach_bal,
        "reachability_accuracy": mean(reach[True] + reach[False]),
        "reachability_yes_accuracy": mean(reach[True]),
        "reachability_no_accuracy": mean(reach[False]),
        "external_macro": (edge_acc + reach_bal) / 2 if edge_acc is not None and reach_bal is not None else None,
        "reversed_edge_negative_accuracy": mean(rev),
        "per_source": {src: {"edge_exists_accuracy": mean(p["edge"]),
                             "reachability_accuracy": mean(p["reach"][True] + p["reach"][False]),
                             "reachability_balanced_accuracy": _balanced(p["reach"])}
                       for src, p in sorted(per.items())},
    }


def flags(samples: Sequence[Dict[str, Any]], responses: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Per-sample strict scores as bit strings in subset order (same layout as bootstrap.score_flags)."""
    from . import bootstrap
    scored = [score_item(s, responses.get(s["sample_id"])) for s in samples]
    return {"n": len(samples), "missing_responses": sum(1 for s in samples if s["sample_id"] not in responses),
            "order_sha": bootstrap.order_sha(samples),
            "correct": "".join("1" if r["correct"] else "0" for r in scored),
            "format_ok": "".join("1" if r["format_ok"] else "0" for r in scored)}
