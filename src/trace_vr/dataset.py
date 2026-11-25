"""Dataset builder: root graphs -> images, paired samples, manifest.

Leakage unit: `graph_group_id`. One root graph and everything derived from it (edited
twins, relabelled or restyled copies, every question) share the id and live in one split.
In addition no two graphs from different groups are isomorphic, so a topology never
appears on both sides of a split even under different labels or drawings.

What the model may see is only `image_path` and `public_input.instruction`
(see `to_model_example`). Everything under `private_truth` and `metadata` is for
labelling, scoring and analysis.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import PIL
import yaml

from . import GENERATOR_VERSION, families, layout as lay, render, semantics as sem, tasks
from .graphs import Graph
from .iso import TopologyRegistry
from .rng import Rng, stable_hash
from .styles import Style, get_style

MAX_ROOT_ATTEMPTS = 80
REPAIR_PROPOSALS = 8
QA_PROPOSALS = 6

DEFAULTS: Dict[str, Any] = {
    "image_size": 512,
    "supersample": 3,
    "invariant_rate": 0.25,
    # A root graph is only used if each main task gets an answer-flipping twin on it.
    "require_pairs": True,
    "repair_pair_types": [
        ["add", "unchanged"],
        ["add", "impossible"],
        ["remove", "unchanged"],
        ["remove", "impossible"],
    ],
    "secondary_tasks": [tasks.EDGE_EXISTS, tasks.SHORTEST_PATH, tasks.EXTRACTION],
}


# How a split picks the node count of root idx (split key `node_quota`).
# "random" (default; every build before 2026-11-12): drawn per attempt, so a rejected candidate
#   may be replaced by one of a different size and hard sizes end up under-represented.
# "equal": sizes cycle lo, lo+1, ..., hi by root index and do not depend on the attempt, so a
#   rejection is retried at the same size and each size gets roots // k (+1) roots.
NODE_QUOTA_MODES = ("random", "equal")


def node_count_for(sc: Dict[str, Any], idx: int, rng: Rng) -> int:
    lo, hi = int(sc["nodes"][0]), int(sc["nodes"][1])
    if sc.get("node_quota", "random") == "equal":
        return lo + idx % (hi - lo + 1)
    return rng.child("n").randint(lo, hi)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = dict(DEFAULTS)
    cfg.update(raw)
    for key in ("name", "seed", "splits"):
        if key not in cfg:
            raise ValueError("config is missing '%s'" % key)
    for split, sc in cfg["splits"].items():
        for key in ("roots", "families", "nodes", "styles"):
            if key not in sc:
                raise ValueError("split '%s' is missing '%s'" % (split, key))
        for fam in sc["families"]:
            if fam not in families.FAMILIES:
                raise ValueError("unknown family '%s' in split '%s'" % (fam, split))
            if sc["nodes"][0] < families.MIN_NODES[fam]:
                raise ValueError("family '%s' needs more nodes than split '%s' allows" % (fam, split))
        for sid in sc["styles"]:
            get_style(sid)
        if sc["nodes"][1] > len(tasks.LABEL_ALPHABET):
            raise ValueError("split '%s' asks for more nodes than there are labels" % split)
        quota = sc.get("node_quota", "random")
        if quota not in NODE_QUOTA_MODES:
            raise ValueError("split '%s': node_quota must be one of %s" % (split, sorted(NODE_QUOTA_MODES)))
    excl = cfg.get("exclude_isomorphic_to", [])
    if not isinstance(excl, list) or not all(isinstance(x, str) for x in excl):
        raise ValueError("'exclude_isomorphic_to' must be a list of manifest paths")
    return cfg


def graph_from_record(record: Dict[str, Any]) -> Graph:
    idx = {label: i for i, label in enumerate(record["nodes"])}
    return Graph(len(idx), [(idx[a], idx[b]) for a, b in record["edges"]])


def load_exclusions(cfg: Dict[str, Any], manifest_dir: Path) -> Tuple[Dict[str, str], List[Tuple[Graph, str]]]:
    """Graphs of earlier datasets that this one must not repeat (up to isomorphism).

    `exclude_isomorphic_to` lists manifest paths relative to the directory of the manifest being
    built (e.g. a locked test build next to `pilot.json`). Returns {path: content digest} and
    (graph, pseudo group id) pairs. Group ids are prefixed so they can never equal a new group's.
    """
    digests: Dict[str, str] = {}
    graphs: List[Tuple[Graph, str]] = []
    for rel in cfg.get("exclude_isomorphic_to", []):
        man, _, _, images = load_dataset(Path(manifest_dir) / rel)
        digests[rel] = man["content_digest"]
        for split in sorted(images):
            for rec in images[split]:
                graphs.append((graph_from_record(rec), "excluded:%s:%s" % (man["name"], rec["graph_group_id"])))
    return digests, graphs


class Reject(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ImageSpec:
    """One picture: a graph, its labels and a concrete drawing."""

    def __init__(self, group_id: str, graph: Graph, labels: Sequence[str], style: Style, layout: lay.Layout,
                 variant: str, edit: Optional[Tuple[str, int, int]]) -> None:
        self.graph = graph
        self.labels = list(labels)
        self.style = style
        self.layout = layout
        self.variant = variant
        self.edit = edit
        self.image_id = "img" + stable_hash(
            group_id, variant, style.style_id, ",".join(self.labels), repr(graph.edges), repr(layout.pos), length=16
        )

    def nodes(self) -> List[str]:
        return sorted(self.labels)

    def edges(self) -> List[List[str]]:
        return sorted([self.labels[u], self.labels[v]] for u, v in self.graph.edges)

    def record(self, group_id: str, family: str, split: str, size: int) -> Dict[str, Any]:
        lab = self.labels
        return {
            "image_id": self.image_id,
            "graph_group_id": group_id,
            "graph_family": family,
            "split": split,
            "variant": self.variant,
            "edit": None if self.edit is None else {"op": self.edit[0], "src": lab[self.edit[1]], "dst": lab[self.edit[2]]},
            "style_id": self.style.style_id,
            "layout": self.layout.layout_name,
            "crossings": self.layout.crossings,
            "size": size,
            "nodes": self.nodes(),
            "edges": self.edges(),
            "positions": {lab[i]: [int(x), int(y)] for i, (x, y) in enumerate(self.layout.pos)},
            "routes": [
                {
                    "src": lab[u],
                    "dst": lab[v],
                    # Full precision: JSON round-trips floats exactly, so the validator re-checks
                    # the very same coordinates the builder accepted.
                    "points": [[px, py] for px, py in self.layout.routes[(u, v)]],
                }
                for (u, v) in sorted(self.layout.routes)
            ],
        }


def _lp(labels: Sequence[str], pair: Sequence[int]) -> List[str]:
    return [labels[pair[0]], labels[pair[1]]]


class _Group:
    """Builds every image and sample that derives from one root graph."""

    def __init__(self, cfg: Dict[str, Any], split: str, idx: int, attempt: int, registry: TopologyRegistry,
                 pair_type_counts: List[int]) -> None:
        sc = cfg["splits"][split]
        self.cfg, self.split, self.sc = cfg, split, sc
        self.size = int(cfg["image_size"])
        self.registry = registry
        self.pair_type_counts = pair_type_counts
        self.seed_parts = [cfg["seed"], cfg["name"], split, idx, attempt]
        self.rng = Rng(*self.seed_parts)
        self.group_id = "g" + stable_hash(cfg["name"], cfg["seed"], split, idx, length=12)
        fams = sorted(sc["families"])
        self.family = self.rng.child("family").weighted_choice(fams, [float(sc["families"][f]) for f in fams])
        self.n = node_count_for(sc, idx, self.rng)
        self.images: Dict[Tuple, Optional[ImageSpec]] = {}
        self.samples: List[Dict[str, Any]] = []
        self.pair_type: Optional[int] = None
        self.unpaired: List[str] = []
        self.missing: List[str] = []

    # ---- construction ----------------------------------------------------------
    def build(self) -> None:
        self.g = families.generate(self.family, self.n, self.rng.child("graph"))
        ok, why = families.family_invariants_ok(self.g)
        if not ok:
            raise RuntimeError("family %s broke an invariant: %s" % (self.family, why))
        if self.registry.find_other_group(self.g, self.group_id) is not None:
            raise Reject("duplicate_topology")
        self.labels = self.rng.child("labels").sample(tasks.LABEL_ALPHABET, self.n)
        self.style = get_style(self.rng.child("style").choice(list(self.sc["styles"])))
        layout, _ = lay.find_layout(self.g, self.style, self.rng.child("layout"), self.size)
        if layout is None:
            raise Reject("layout")
        self.layout = layout
        root = ImageSpec(self.group_id, self.g, self.labels, self.style, layout, "root", None)
        self.images[("root",)] = root
        self._repair(root)
        base_qa = [s for s in (self._qa(root, tasks.REACHABILITY), self._qa(root, tasks.CRITICAL_EDGE)) if s]
        self._secondary(root)
        if self.rng.child("invariants").chance(float(self.cfg["invariant_rate"])):
            self._invariants(root, base_qa)

    def twin_image(self, mod: Tuple[str, int, int]) -> Optional[ImageSpec]:
        """Image of the root graph with one edge changed, on the same node positions."""
        key = ("edit",) + tuple(mod)
        if key not in self.images:
            g2 = sem.apply_mod(self.g, mod)
            spec: Optional[ImageSpec] = None
            if self.registry.find_other_group(g2, self.group_id) is None:
                layout2, _ = lay.layout_for_positions(g2, self.layout.pos, self.style, self.size, self.layout.layout_name)
                if layout2 is not None:
                    spec = ImageSpec(self.group_id, g2, self.labels, self.style, layout2, "edit", tuple(mod))
            self.images[key] = spec
        return self.images[key]

    # ---- samples ---------------------------------------------------------------
    def _sample(self, image: ImageSpec, query: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        task = query["type"]
        nodes, edges = image.nodes(), image.edges()
        public: Dict[str, Any] = {"instruction": tasks.render_instruction(query), "query": query}
        truth: Dict[str, Any] = {"nodes": nodes, "edges": edges}
        sample_id = "s" + stable_hash(self.group_id, image.image_id, task, canonical_json(query), length=16)
        info = dict(info)
        if task == tasks.REPAIR:
            public["max_edits"] = query["max_edits"]
            _, _, sol = tasks.solve_repair_labels(query, nodes, edges)
            solutions = [tasks.edits_to_labels(s, nodes) for s in sol.solutions]
            edits = Rng(self.cfg["seed"], "target", sample_id).choice(solutions) if sol.status == "edited" else []
            target: Any = {"status": sol.status, "edits": edits}
            truth.update(
                {
                    "status": sol.status,
                    "constraints": [{"type": "reach", "src": a, "dst": b} for a, b in query["must_reach"]]
                    + [{"type": "no_reach", "src": a, "dst": b} for a, b in query["must_not_reach"]],
                    "allowed_actions": [{"op": sem.ADD, "src": a, "dst": b} for a, b in query["allowed_additions"]]
                    + [{"op": sem.REMOVE, "scope": "any_existing_edge"}],
                    "all_minimal_solutions": solutions,
                    "minimum_edit_count": sol.min_edits,
                }
            )
            info["category"] = sem.repair_category(sol)
        else:
            target = tasks.solve_query(query, nodes, edges)
            truth["answer"] = target
        return {
            "sample_id": sample_id,
            "graph_group_id": self.group_id,
            "graph_family": self.family,
            "split": self.split,
            "task": task,
            "image_id": image.image_id,
            "image_path": "images/%s/%s.png" % (self.split, image.image_id),
            "image_sha256": None,
            "public_input": public,
            "training_target": target,
            "private_truth": truth,
            "pairs": [],
            "metadata": {
                "generator_version": GENERATOR_VERSION,
                "graph_seed": list(self.seed_parts),
                "style_id": image.style.style_id,
                "layout": image.layout.layout_name,
                "node_count": image.graph.n,
                "edge_count": image.graph.edge_count,
                "crossings": image.layout.crossings,
                "image_variant": image.variant,
                "task_info": info,
            },
        }

    def _link(self, base: Dict[str, Any], variant: Dict[str, Any], kind: str) -> None:
        pair_id = "p" + stable_hash(kind, base["sample_id"], variant["sample_id"], length=16)
        base["pairs"].append({"pair_id": pair_id, "kind": kind, "role": "base"})
        variant["pairs"].append({"pair_id": pair_id, "kind": kind, "role": "variant"})

    def _repair_query(self, labels: Sequence[str], spec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": tasks.REPAIR,
            "must_reach": sorted(_lp(labels, p) for p in spec["must_reach"]),
            "must_not_reach": sorted(_lp(labels, p) for p in spec["must_not"]),
            "allowed_additions": sorted(_lp(labels, p) for p in spec["additions"]),
            "max_edits": 1,
        }

    def _repair(self, root: ImageSpec) -> None:
        types = [tuple(t) for t in self.cfg["repair_pair_types"]]
        r = self.rng.child("repair")
        tie = r.child("tie").shuffled(range(len(types)))
        order = sorted(range(len(types)), key=lambda i: (self.pair_type_counts[i], tie.index(i)))
        for ti in order:
            a, b = types[ti]
            roles = [(a, b), (b, a)] if r.child("role", ti).chance(0.5) else [(b, a), (a, b)]
            for base_cat, twin_cat in roles:
                for k in range(REPAIR_PROPOSALS):
                    spec = sem.propose_repair(self.g, r.child("proposal", ti, base_cat, k), base_cat)
                    if spec is None:
                        continue
                    twins = [m for m, s2 in sem.repair_twin_mods(self.g, spec) if sem.repair_category(s2) == twin_cat]
                    for mod in r.child("twin", ti, base_cat, k).shuffled(twins):
                        img2 = self.twin_image(mod)
                        if img2 is None:
                            continue
                        query = self._repair_query(self.labels, spec)
                        s1 = self._sample(root, query, dict(spec["info"]))
                        s2 = self._sample(img2, query, {"direct": None, "violated_sides": None})
                        if s1["metadata"]["task_info"]["category"] != base_cat or \
                                s2["metadata"]["task_info"]["category"] != twin_cat:
                            raise RuntimeError("repair pair categories disagree with the solver")
                        self._link(s1, s2, "critical")
                        self.samples += [s1, s2]
                        self.pair_type = ti
                        return
        if self.cfg["require_pairs"]:
            raise Reject("no_twin:" + tasks.REPAIR)
        for cat in r.child("fallback").shuffled(list(sem.REPAIR_CATEGORIES)):
            for k in range(REPAIR_PROPOSALS):
                spec = sem.propose_repair(self.g, r.child("single", cat, k), cat)
                if spec is not None:
                    self.samples.append(self._sample(root, self._repair_query(self.labels, spec), dict(spec["info"])))
                    self.unpaired.append(tasks.REPAIR)
                    return
        raise Reject("repair_task")

    def _qa_query(self, task: str, labels: Sequence[str], q: Dict[str, Any]) -> Dict[str, Any]:
        query: Dict[str, Any] = {"type": task, "src": labels[q["src"]], "dst": labels[q["dst"]]}
        if task == tasks.CRITICAL_EDGE:
            query["edge"] = _lp(labels, q["edge"])
        return query

    def _qa(self, root: ImageSpec, task: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Adds a question on the root image and, if possible, its answer-flipping twin.

        Returns (index-level question, base sample) or None if the task has no valid
        question on this graph."""
        r = self.rng.child(task)
        propose = sem.propose_reachability if task == tasks.REACHABILITY else sem.propose_critical_edge
        first = r.child("first").chance(0.5)
        single: Optional[Dict[str, Any]] = None
        for ans in (first, not first):
            for k in range(QA_PROPOSALS):
                q = propose(self.g, r.child("q", ans, k), ans)
                if q is None:
                    continue
                if single is None:
                    single = q
                for mod in q["twin_mods"]:
                    img2 = self.twin_image(mod)
                    if img2 is None:
                        continue
                    query = self._qa_query(task, self.labels, q)
                    s1 = self._sample(root, query, dict(q["info"]))
                    s2 = self._sample(img2, query, {})
                    if s1["training_target"]["answer"] is not ans or s2["training_target"]["answer"] is ans:
                        raise RuntimeError("QA pair answers disagree with the solver")
                    self._link(s1, s2, "critical")
                    self.samples += [s1, s2]
                    return q, s1
        if self.cfg["require_pairs"]:
            raise Reject("no_twin:" + task)
        if single is None:
            self.missing.append(task)
            return None
        s1 = self._sample(root, self._qa_query(task, self.labels, single), dict(single["info"]))
        self.samples.append(s1)
        self.unpaired.append(task)
        return single, s1

    def _secondary(self, root: ImageSpec) -> None:
        wanted = list(self.cfg["secondary_tasks"])
        if tasks.EDGE_EXISTS in wanted:
            q = sem.propose_edge_exists(self.g, self.rng.child(tasks.EDGE_EXISTS))
            self.samples.append(self._sample(root, self._qa_query(tasks.EDGE_EXISTS, self.labels, q), q["info"]))
        if tasks.SHORTEST_PATH in wanted:
            q = sem.propose_shortest_path(self.g, self.rng.child(tasks.SHORTEST_PATH))
            self.samples.append(self._sample(root, self._qa_query(tasks.SHORTEST_PATH, self.labels, q), q["info"]))
        if tasks.EXTRACTION in wanted:
            self.samples.append(self._sample(root, {"type": tasks.EXTRACTION}, {}))

    def _invariants(self, root: ImageSpec, base_qa: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
        """Same graph, same question, different appearance: the answer must not change."""
        if not base_qa:
            return
        r = self.rng.child("invariant")
        new_labels = None
        for k in range(8):
            cand = r.child("labels", k).sample(tasks.LABEL_ALPHABET, self.n)
            if sum(1 for a, b in zip(cand, self.labels) if a != b) * 2 >= self.n:
                new_labels = cand
                break
        variants: List[Tuple[str, ImageSpec]] = []
        if new_labels is not None:
            img = ImageSpec(self.group_id, self.g, new_labels, self.style, self.layout, "relabel", None)
            self.images[("relabel",)] = img
            variants.append(("invariant_relabel", img))
        others = [s for s in self.sc["styles"] if s != self.style.style_id]
        if others:
            style2 = get_style(r.child("style").choice(others))
            layout2, _ = lay.find_layout(self.g, style2, r.child("layout"), self.size)
            if layout2 is not None:
                img = ImageSpec(self.group_id, self.g, self.labels, style2, layout2, "restyle", None)
                self.images[("restyle",)] = img
                variants.append(("invariant_style", img))
        for kind, img in variants:
            for q, base in base_qa:
                s = self._sample(img, self._qa_query(base["task"], img.labels, q), {})
                if s["training_target"] != base["training_target"]:
                    raise RuntimeError("invariant variant changed the answer")
                self._link(base, s, kind)
                self.samples.append(s)

    def used_images(self) -> List[ImageSpec]:
        used = {s["image_id"] for s in self.samples}
        return [img for img in self.images.values() if img is not None and img.image_id in used]


# ------------------------------------------------------------------------- build
def build_dataset(cfg: Dict[str, Any], out_dir: Path, manifest_path: Path, log=print) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    manifest_path = Path(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = TopologyRegistry()
    excluded_digests, excluded_graphs = load_exclusions(cfg, manifest_path.parent)
    for g, gid in excluded_graphs:
        registry.add(g, gid)
    size, ss = int(cfg["image_size"]), int(cfg["supersample"])
    manifest_splits: Dict[str, Any] = {}
    rejections: Dict[str, Dict[str, int]] = {}
    acceptance: Dict[str, Dict[str, Dict[str, int]]] = {}
    all_samples: Dict[str, List[Dict[str, Any]]] = {}
    all_images: Dict[str, List[Dict[str, Any]]] = {}
    content_digests: Dict[str, str] = {}
    image_digests: Dict[str, str] = {}

    for split, sc in cfg["splits"].items():
        content_hash, image_hash = hashlib.sha256(), hashlib.sha256()
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        pair_type_counts = [0] * len(cfg["repair_pair_types"])
        samples: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []
        rej = rejections.setdefault(split, {})
        acc = acceptance.setdefault(split, {})
        unpaired: Dict[str, int] = {}
        missing: Dict[str, int] = {}
        for idx in range(int(sc["roots"])):
            group: Optional[_Group] = None
            for attempt in range(MAX_ROOT_ATTEMPTS):
                cand = _Group(cfg, split, idx, attempt, registry, pair_type_counts)
                slot = acc.setdefault("%s/n=%d" % (cand.family, cand.n), {"accepted": 0, "rejected": 0})
                try:
                    cand.build()
                except Reject as exc:
                    rej[exc.reason] = rej.get(exc.reason, 0) + 1
                    slot["rejected"] += 1
                    continue
                slot["accepted"] += 1
                group = cand
                break
            if group is None:
                raise RuntimeError("could not build root %d of split '%s' in %d attempts" % (idx, split, MAX_ROOT_ATTEMPTS))
            if group.pair_type is not None:
                pair_type_counts[group.pair_type] += 1
            for t in group.unpaired:
                unpaired[t] = unpaired.get(t, 0) + 1
            for t in group.missing:
                missing[t] = missing.get(t, 0) + 1
            sha_by_image: Dict[str, str] = {}
            for img in group.used_images():
                registry.add(img.graph, group.group_id)
                picture = render.draw_diagram(img.layout.pos, img.labels, img.layout.routes, img.style, size, ss)
                rel = "images/%s/%s.png" % (split, img.image_id)
                picture.save(out_dir / rel, format="PNG", optimize=False, compress_level=6)
                digest = sha256_bytes((out_dir / rel).read_bytes())
                sha_by_image[img.image_id] = digest
                rec = img.record(group.group_id, group.family, split, size)
                rec["image_path"] = rel
                rec["image_sha256"] = digest
                images.append(rec)
            for s in group.samples:
                content_hash.update(canonical_json(s).encode("ascii"))  # image hash still None here
                s["image_sha256"] = sha_by_image[s["image_id"]]
                image_hash.update(s["image_sha256"].encode("ascii"))
                samples.append(s)
            if (idx + 1) % 25 == 0 or idx + 1 == int(sc["roots"]):
                log("  %s: %d/%d roots, %d samples, %d images" % (split, idx + 1, sc["roots"], len(samples), len(images)))
        sfile, ifile = "%s.samples.jsonl" % split, "%s.images.jsonl" % split
        _write_jsonl(out_dir / sfile, samples)
        _write_jsonl(out_dir / ifile, images)
        manifest_splits[split] = {
            "roots": int(sc["roots"]),
            "samples": len(samples),
            "images": len(images),
            "samples_file": sfile,
            "images_file": ifile,
            "samples_sha256": sha256_bytes((out_dir / sfile).read_bytes()),
            "images_file_sha256": sha256_bytes((out_dir / ifile).read_bytes()),
            "unpaired_main_samples": unpaired,
            "tasks_without_valid_question": missing,
        }
        all_samples[split], all_images[split] = samples, images
        content_digests[split], image_digests[split] = content_hash.hexdigest(), image_hash.hexdigest()

    manifest = {
        "name": cfg["name"],
        "generator_version": GENERATOR_VERSION,
        "config": cfg,
        "config_sha256": sha256_bytes(canonical_json(cfg).encode("ascii")),
        "data_dir": _relative(out_dir, manifest_path.parent),
        "build_environment": {
            "python": platform.python_version(),
            "pillow": PIL.__version__,
            "platform": platform.platform(),
        },
        "splits": manifest_splits,
        # Splits are generated in this order; later splits are de-duplicated against earlier ones.
        "split_order": list(cfg["splits"].keys()),
        "content_digest": combine_digests(content_digests),
        "images_digest": combine_digests(image_digests),
        # Datasets whose graphs this one avoids up to isomorphism, with their content digests.
        "excluded_manifests": excluded_digests,
        "stats": {split: summarize(all_samples[split], all_images[split]) for split in all_samples},
        "rejections": rejections,
        "acceptance_by_family_and_size": acceptance,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def combine_digests(by_split: Dict[str, str]) -> str:
    """One digest for the whole dataset, independent of the order splits are listed in."""
    h = hashlib.sha256()
    for split in sorted(by_split):
        h.update((split + ":" + by_split[split] + "\n").encode("ascii"))
    return h.hexdigest()


def _relative(target: Path, start: Path) -> str:
    import os

    return Path(os.path.relpath(Path(target).resolve(), Path(start).resolve())).as_posix()


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(canonical_json(row) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_dataset(manifest_path: Path):
    """Returns (manifest, data_dir, samples by split, image records by split)."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = (manifest_path.parent / manifest["data_dir"]).resolve()
    samples = {s: _read_jsonl(data_dir / info["samples_file"]) for s, info in manifest["splits"].items()}
    images = {s: _read_jsonl(data_dir / info["images_file"]) for s, info in manifest["splits"].items()}
    return manifest, data_dir, samples, images


UNLOCK_ENV = "TRACE_UNLOCK_TEST"
UNLOCK_VALUE = "final-evaluation"


def check_split_allowed(split: str, purpose: str, env: Optional[Dict[str, str]] = None) -> None:
    """Locked test splits (names starting with "test") are never trained on, and are evaluated
    only in the final run of the study, which sets TRACE_UNLOCK_TEST=final-evaluation on purpose.
    Development decisions use dev only (brief section 19, M6)."""
    if not split.startswith("test"):
        return
    if purpose == "train":
        raise SystemExit("refusing to train on locked test split %r" % split)
    env = os.environ if env is None else env
    if env.get(UNLOCK_ENV) != UNLOCK_VALUE:
        raise SystemExit("split %r is locked; set %s=%s only for the final evaluation" % (split, UNLOCK_ENV, UNLOCK_VALUE))


# Input conditions. "image" is the only one allowed for training and for the main results.
# The other two are evaluation-only diagnostics that hand the model the true graph as text
# (an oracle): "image+edges" keeps the drawing, "edges" replaces it with a blank image.
# Comparing them with "image" separates reading the drawing from reasoning about the graph.
INPUT_MODES = ("image", "image+edges", "edges")


def oracle_graph_text(sample: Dict[str, Any], blank_image: bool) -> str:
    """One line with the true node and edge lists (sorted). Diagnostic use only."""
    truth = sample["private_truth"]
    nodes = ", ".join(sorted(truth["nodes"]))
    edges = ", ".join("%s->%s" % (a, b) for a, b in sorted(tuple(e) for e in truth["edges"]))
    head = ("The image is blank. The graph is given here as text instead" if blank_image
            else "Exact text description of the graph in the image")
    return "%s. Nodes: %s. Directed edges: %s." % (head, nodes, edges)


def to_model_example(sample: Dict[str, Any], data_dir: Path, with_target: bool = False,
                     input_mode: str = "image") -> Dict[str, Any]:
    """The only view of a sample that may be handed to a model adapter.

    `image` is None for input_mode "edges"; the model adapter then uses a blank image of the
    same size, so the number of image tokens stays the same.
    """
    if input_mode not in INPUT_MODES:
        raise KeyError("unknown input mode: %r" % (input_mode,))
    if input_mode != "image" and with_target:
        raise ValueError("oracle input modes are for evaluation only, never for training")
    prompt = sample["public_input"]["instruction"]
    if input_mode != "image":
        prompt = oracle_graph_text(sample, blank_image=input_mode == "edges") + "\n" + prompt
    example = {
        "image": None if input_mode == "edges" else str(Path(data_dir) / sample["image_path"]),
        "prompt": prompt,
    }
    if with_target:
        example["target"] = tasks.format_target(sample["training_target"])
    return example


# ------------------------------------------------------------------------- stats
def _count(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def summarize(samples: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Distribution report for one split (computed from the stored rows, not from builder state)."""
    roots = [i for i in images if i["variant"] == "root"]
    repair = [s for s in samples if s["task"] == tasks.REPAIR]
    stats: Dict[str, Any] = {
        "root_graphs": len(roots),
        "samples_by_task": _count(s["task"] for s in samples),
        "images_by_variant": _count(i["variant"] for i in images),
        "root_family": _count(i["graph_family"] for i in roots),
        "root_node_count": _count(len(i["nodes"]) for i in roots),
        "root_edge_count": _count(len(i["edges"]) for i in roots),
        "root_style": _count(i["style_id"] for i in roots),
        "root_layout": _count(i["layout"] for i in roots),
        "root_crossings": _count(i["crossings"] for i in roots),
        "repair_category": _count(s["metadata"]["task_info"]["category"] for s in repair),
        "repair_category_on_root_images": _count(
            s["metadata"]["task_info"]["category"] for s in repair if s["metadata"]["image_variant"] == "root"
        ),
        "repair_direct_share_on_root_images": _count(
            s["metadata"]["task_info"].get("direct") for s in repair if s["metadata"]["image_variant"] == "root"
        ),
        "pairs_by_kind": _count(
            p["kind"] + ":" + s["task"] for s in samples for p in s["pairs"] if p["role"] == "base"
        ),
    }
    for task in tasks.BOOLEAN_TASKS:
        rows = [s for s in samples if s["task"] == task]
        stats["answer_balance:" + task] = _count(s["training_target"]["answer"] for s in rows)
        stats["answer_on_root_images:" + task] = _count(
            s["training_target"]["answer"] for s in rows if s["metadata"]["image_variant"] == "root"
        )
    stats["shortest_path_answers"] = _count(
        s["training_target"]["answer"] for s in samples if s["task"] == tasks.SHORTEST_PATH
    )
    return stats
