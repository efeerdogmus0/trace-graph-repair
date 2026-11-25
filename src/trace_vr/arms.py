"""Which training samples each arm sees. Same root graphs, same number of samples and images.
(C and D2 use exactly D's images; B uses the same root images but other twins, so it shares
546 of D's 1,024 pilot-train images: docs/exposure_pilot_train.md.)

D (TRACE):  the repair pair of every root graph (base image + twin image).
B (QA-SFT): one non-edit pair per root graph, alternating reachability and critical edge.
C (structural SFT, MCDGraph-inspired, protocol item 4): on exactly the two images D sees
   (root + repair twin), a balanced cycle of three structural tasks over the 2 x groups image
   slots: graph description (list every node and edge), masked infilling (a third of the
   edges, at least one, are withheld from a listed edge set; name them) and graph matching
   (one image + an edge list; is it exactly the drawn graph? negatives are the other image's
   edge list, which differs by one edit). Matching replaces MCDGraph's two-image contrastive
   discrimination because the pipeline is one image per sample. The edge lists in infilling
   and matching prompts are C's declared extra training input; evaluation inputs are the same
   for every arm. Same samples and images as D; target and prompt tokens differ (reported).
C_desc (ablation; the C built on 2026-11-11): graph description only, on D's images.
D2 (first allowed revision, brief section 20): D with one auxiliary task, "edit outcome":
   given the repair goal and one legal edit, does the goal hold after the edit? Same images
   and sample count as D; per root group one image keeps its repair sample and the other
   gets an edit-outcome sample (alternating which). Removal edits make this close to the
   critical-edge question, so D2 is NOT a pure transfer test and must be reported as such.

Only root and edited images are used; relabelled/restyled copies are evaluation material.
"""
from __future__ import annotations

from typing import Any, Dict, List

from . import semantics as sem, tasks
from .rng import Rng

ARMS = ("D", "B", "C", "D2", "C_desc")
C_CYCLE = (tasks.EXTRACTION, tasks.INFILL, tasks.MATCH)


def training_samples(samples: List[Dict[str, Any]], arm: str) -> List[Dict[str, Any]]:
    if arm not in ARMS:
        raise KeyError("unknown arm: %s" % arm)
    main = [s for s in samples if s["metadata"]["image_variant"] in ("root", "edit")
            and any(p["kind"] == "critical" for p in s["pairs"])]
    groups: List[str] = []
    for s in main:
        if s["graph_group_id"] not in groups:
            groups.append(s["graph_group_id"])
    out: List[Dict[str, Any]] = []
    slot = 0
    n_match = 0
    for i, gid in enumerate(groups):
        if arm == "C_desc":
            out += [_description_sample(s) for s in main
                    if s["graph_group_id"] == gid and s["task"] == tasks.REPAIR]
            continue
        if arm == "C":
            reps = [s for s in main if s["graph_group_id"] == gid and s["task"] == tasks.REPAIR]
            for j, s in enumerate(reps):
                kind = C_CYCLE[slot % 3]
                slot += 1
                if kind == tasks.EXTRACTION:
                    out.append(_description_sample(s))
                elif kind == tasks.INFILL:
                    out.append(_infill_sample(s))
                else:
                    other = reps[1 - j] if len(reps) == 2 else None
                    positive = n_match % 2 == 0 or other is None
                    n_match += 1
                    out.append(_match_sample(s, None if positive else other))
            continue
        if arm == "D2":
            reps = [s for s in main if s["graph_group_id"] == gid and s["task"] == tasks.REPAIR]
            out += [_outcome_sample(s) if j == i % 2 else s for j, s in enumerate(reps)]
            continue
        if arm == "D":
            task = tasks.REPAIR
        else:
            task = tasks.REACHABILITY if i % 2 == 0 else tasks.CRITICAL_EDGE
        out += [s for s in main if s["graph_group_id"] == gid and s["task"] == task]
    return out


def _description_sample(src: Dict[str, Any]) -> Dict[str, Any]:
    """Graph-description training sample on the image of `src` (a repair sample)."""
    truth = src["private_truth"]
    target = {"nodes": sorted(truth["nodes"]),
              "edges": sorted([list(e) for e in truth["edges"]])}
    query = {"type": tasks.EXTRACTION}
    return {
        "sample_id": "c" + src["sample_id"],
        "graph_group_id": src["graph_group_id"],
        "split": src["split"],
        "task": tasks.EXTRACTION,
        "image_id": src["image_id"],
        "image_path": src["image_path"],
        "image_sha256": src["image_sha256"],
        "public_input": {"instruction": tasks.render_instruction(query), "query": query},
        "training_target": target,
        "private_truth": {"answer": target, "nodes": target["nodes"], "edges": target["edges"]},
        "metadata": dict(src["metadata"]),
        "pairs": [],
    }


def _structural_sample(src: Dict[str, Any], prefix: str, query: Dict[str, Any]) -> Dict[str, Any]:
    truth = src["private_truth"]
    answer = tasks.solve_query(query, truth["nodes"], truth["edges"])
    return {
        "sample_id": prefix + src["sample_id"],
        "graph_group_id": src["graph_group_id"],
        "split": src["split"],
        "task": query["type"],
        "image_id": src["image_id"],
        "image_path": src["image_path"],
        "image_sha256": src["image_sha256"],
        "public_input": {"instruction": tasks.render_instruction(query), "query": query},
        "training_target": answer,
        "private_truth": {"answer": answer, "nodes": truth["nodes"], "edges": truth["edges"]},
        "metadata": dict(src["metadata"]),
        "pairs": [],
    }


def _infill_sample(src: Dict[str, Any]) -> Dict[str, Any]:
    """Masked infilling: a third of the edges (at least one) are withheld; deterministic in the id."""
    edges = sorted([list(e) for e in src["private_truth"]["edges"]])
    k = max(1, len(edges) // 3)
    hidden = Rng("c-infill", src["sample_id"]).sample(list(range(len(edges))), k)
    shown = [e for t, e in enumerate(edges) if t not in set(hidden)]
    return _structural_sample(src, "ci", {"type": tasks.INFILL, "shown_edges": shown, "n_missing": k})


def _match_sample(src: Dict[str, Any], other: Any) -> Dict[str, Any]:
    """Graph matching on src's image; the listed edges are src's own (positive) or `other`'s."""
    listed = (other or src)["private_truth"]["edges"]
    return _structural_sample(src, "cm", {"type": tasks.MATCH, "edges": sorted([list(e) for e in listed])})


def _outcome_sample(src: Dict[str, Any]) -> Dict[str, Any]:
    """Edit-outcome sample on the image of repair sample `src`; answer balanced ~50/50.

    The edit is drawn from the declared action space (allowed additions that are absent,
    removal of any drawn edge). With probability 1/2 it is one that satisfies the goal when
    such an edit exists; otherwise one that does not. Deterministic in the sample id.
    """
    truth = src["private_truth"]
    q = src["public_input"]["query"]
    g, idx = tasks.build_graph(truth["nodes"], truth["edges"])
    nodes = truth["nodes"]
    reach = [(idx[a], idx[b]) for a, b in q["must_reach"]]
    forbid = [(idx[a], idx[b]) for a, b in q["must_not_reach"]]
    acts = sem.legal_actions(g, [(idx[a], idx[b]) for a, b in q["allowed_additions"]])
    good = [a for a in acts if sem.satisfied(sem.apply_action(g, a), reach, forbid)]
    bad = [a for a in acts if a not in good]
    rng = Rng("d2-outcome", src["sample_id"])
    pool = good if (good and (not bad or rng.chance(0.5))) else bad
    op, u, v = rng.choice(pool)
    query = {"type": tasks.EDIT_OUTCOME, "must_reach": q["must_reach"], "must_not_reach": q["must_not_reach"],
             "edit": {"op": op, "src": nodes[u], "dst": nodes[v]}}
    answer = tasks.solve_query(query, nodes, truth["edges"])
    return {
        "sample_id": "o" + src["sample_id"],
        "graph_group_id": src["graph_group_id"],
        "split": src["split"],
        "task": tasks.EDIT_OUTCOME,
        "image_id": src["image_id"],
        "image_path": src["image_path"],
        "image_sha256": src["image_sha256"],
        "public_input": {"instruction": tasks.render_instruction(query), "query": query},
        "training_target": answer,
        "private_truth": {"answer": answer, "nodes": truth["nodes"], "edges": truth["edges"]},
        "metadata": dict(src["metadata"]),
        "pairs": [],
    }


def _first_roots(samples: List[Dict[str, Any]], n_roots: int) -> set:
    groups: List[str] = []
    for s in samples:
        if s["graph_group_id"] not in groups:
            groups.append(s["graph_group_id"])
    return set(groups[:n_roots])


SAMPLE_SETS = ("main", "invariance")


def eval_samples(samples: List[Dict[str, Any]], n_roots: int) -> List[Dict[str, Any]]:
    """Main samples (repair, reachability, critical edge) of the first n root graphs."""
    keep = _first_roots(samples, n_roots)
    wanted = (tasks.REPAIR, tasks.REACHABILITY, tasks.CRITICAL_EDGE)
    return [s for s in samples if s["graph_group_id"] in keep and s["task"] in wanted
            and s["metadata"]["image_variant"] in ("root", "edit")]


def invariance_samples(samples: List[Dict[str, Any]], n_roots: int) -> List[Dict[str, Any]]:
    """Relabelled and restyled copies (reachability, critical edge) of the first n root graphs.

    Their pair partners are root samples already inside eval_samples(samples, n_roots), so a
    model is scored on them in a separate run file and invariance pairs are formed afterwards
    (analysis.invariance_table). eval_samples is left unchanged so existing flags files and
    their order hashes stay valid."""
    keep = _first_roots(samples, n_roots)
    wanted = (tasks.REACHABILITY, tasks.CRITICAL_EDGE)
    return [s for s in samples if s["graph_group_id"] in keep and s["task"] in wanted
            and s["metadata"]["image_variant"] in ("relabel", "restyle")]


def select_samples(samples: List[Dict[str, Any]], n_roots: int, sample_set: str = "main") -> List[Dict[str, Any]]:
    if sample_set not in SAMPLE_SETS:
        raise ValueError("unknown sample set %r" % sample_set)
    return eval_samples(samples, n_roots) if sample_set == "main" else invariance_samples(samples, n_roots)


def run_suffix(split: str, input_mode: str = "image", sample_set: str = "main") -> str:
    """File suffix of predictions_/metrics_ files; unchanged for the default main image run."""
    suffix = split if input_mode == "image" else "%s_%s" % (split, input_mode.replace("+", "_"))
    return suffix if sample_set == "main" else suffix + "_inv"
