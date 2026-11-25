"""Exact task semantics on node indices: the repair solver and task proposal logic.

Nothing in this module is learned or approximate. With at most one edit and small graphs
every allowed action is enumerated, so "minimal" and "impossible" are exact statements
about the declared action space, not heuristics.

Twin design: every main sample is proposed together with single-edge modifications of
the graph that keep the instruction text valid but change the correct answer. A model
that ignores the image cannot answer both members of such a pair correctly.

Root graphs are weakly connected. Twin graphs are not required to be: removing one edge
may split the drawing, which is a legitimate picture and keeps tree-like graphs usable.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .graphs import Edge, Graph
from .rng import Rng

Action = Tuple[str, int, int]  # ("add_edge" | "remove_edge", src, dst)
Mod = Tuple[str, int, int]  # ("add" | "remove", src, dst): a change to the *image* graph

ADD = "add_edge"
REMOVE = "remove_edge"


# ----------------------------------------------------------------------------- repair
class RepairSolution:
    """Outcome of exhaustively solving a repair task with a budget of one edit."""

    __slots__ = ("status", "solutions", "min_edits")

    def __init__(self, status: str, solutions: List[Tuple[Action, ...]], min_edits: Optional[int]):
        self.status = status
        self.solutions = solutions
        self.min_edits = min_edits

    def answer_key(self) -> FrozenSet[Tuple]:
        """Set of all fully correct answers, as hashable tuples."""
        if self.status == "edited":
            return frozenset(("edited",) + sol for sol in self.solutions)
        return frozenset([(self.status,)])


def satisfied(g: Graph, must_reach: Sequence[Edge], must_not: Sequence[Edge]) -> bool:
    closure = g.closure()
    for s, t in must_reach:
        if not closure[s] >> t & 1:
            return False
    for s, t in must_not:
        if closure[s] >> t & 1:
            return False
    return True


def apply_action(g: Graph, action: Action) -> Graph:
    op, u, v = action
    if op == ADD:
        return g.with_edge(u, v)
    if op == REMOVE:
        return g.without_edge(u, v)
    raise ValueError("unknown op: %r" % (op,))


def legal_actions(g: Graph, additions: Sequence[Edge]) -> List[Action]:
    """Declared action space: listed additions (if absent) and removal of any existing edge."""
    acts: List[Action] = [(ADD, u, v) for (u, v) in additions if not g.has_edge(u, v)]
    acts += [(REMOVE, u, v) for (u, v) in g.edges]
    return acts


def solve_repair(
    g: Graph, must_reach: Sequence[Edge], must_not: Sequence[Edge], additions: Sequence[Edge]
) -> RepairSolution:
    if satisfied(g, must_reach, must_not):
        return RepairSolution("unchanged", [()], 0)
    sols = [
        (a,)
        for a in legal_actions(g, additions)
        if satisfied(apply_action(g, a), must_reach, must_not)
    ]
    if sols:
        return RepairSolution("edited", sorted(sols), 1)
    return RepairSolution("impossible", [], None)


# ------------------------------------------------------------------------- helpers
def all_pairs(n: int) -> List[Edge]:
    return [(s, t) for s in range(n) for t in range(n) if s != t]


def _has(closure: Sequence[int], p: Edge) -> bool:
    return bool(closure[p[0]] >> p[1] & 1)


def _fixes(closure: Sequence[int], e: Edge, pair: Edge) -> bool:
    """Would adding edge e make pair reachable, given the current closure?"""
    (u, v), (s, t) = e, pair
    return (s == u or _has(closure, (s, u))) and (v == t or _has(closure, (v, t)))


def single_edge_mods(g: Graph) -> List[Mod]:
    """All ways to change the drawn graph by one edge while staying in scope."""
    mods: List[Mod] = [("remove", u, v) for (u, v) in g.edges]
    mods += [("add", u, v) for (u, v) in all_pairs(g.n) if g.can_add(u, v)]
    return mods


def apply_mod(g: Graph, mod: Mod) -> Graph:
    kind, u, v = mod
    return g.with_edge(u, v) if kind == "add" else g.without_edge(u, v)


# ---------------------------------------------------------------- repair proposals
REPAIR_CATEGORIES = ("add", "remove", "unchanged", "impossible")


def repair_category(sol: RepairSolution) -> str:
    if sol.status != "edited":
        return sol.status
    ops = {a[0] for s in sol.solutions for a in s}
    return "add" if ops == {ADD} else "remove" if ops == {REMOVE} else "mixed"


def _plan_add(g: Graph, rng: Rng, closure, reach, unreach, eligible):
    f = rng.choice(eligible)
    c2 = g.with_edge(*f).closure()
    new_pairs = [p for p in unreach if _has(c2, p)]
    if not new_pairs:
        return None
    nondirect = [p for p in new_pairs if p != f]
    main = rng.choice(nondirect) if nondirect and rng.chance(0.8) else rng.choice(new_pairs)
    must_reach, must_not, cands = [main], [], [f]
    distractors = [e for e in eligible if e != f and _fixes(closure, e, main)]
    if distractors and rng.chance(0.7):
        d = rng.choice(distractors)
        cd = g.with_edge(*d).closure()
        bad = [p for p in unreach if _has(cd, p) and not _has(c2, p)]
        if bad:
            must_not.append(rng.choice(bad))
            cands.append(d)
    safe_not = [p for p in unreach if not _has(c2, p) and p not in must_not]
    if not must_not:
        if not safe_not:
            return None
        must_not.append(rng.choice(safe_not))
    if rng.chance(0.5):
        if rng.chance(0.5):
            extra = rng.choice(reach)
            if extra not in must_reach:
                must_reach.append(extra)
        else:
            pool = [p for p in safe_not if p not in must_not]
            if pool:
                must_not.append(rng.choice(pool))
    return must_reach, must_not, cands, {"direct": main == f}


def _plan_remove(g: Graph, rng: Rng, closure, reach, unreach, eligible):
    e = rng.choice(g.edges)
    c2 = g.without_edge(*e).closure()
    lost = [p for p in reach if not _has(c2, p)]
    if not lost:
        return None
    nondirect = [p for p in lost if p != e]
    main = rng.choice(nondirect) if nondirect and rng.chance(0.8) else rng.choice(lost)
    must_reach, must_not = [], [main]
    other_cuts = [x for x in g.edges if x != e and not g.without_edge(*x).reaches(*main)]
    if other_cuts and rng.chance(0.7):
        x = rng.choice(other_cuts)
        c3 = g.without_edge(*x).closure()
        bad = [p for p in reach if _has(c2, p) and not _has(c3, p)]
        if bad:
            must_reach.append(rng.choice(bad))
    kept = [p for p in reach if _has(c2, p) and p not in must_reach]
    if not must_reach:
        if not kept:
            return None
        must_reach.append(rng.choice(kept))
    if rng.chance(0.5):
        if rng.chance(0.5):
            extra = rng.choice(unreach)
            if extra not in must_not:
                must_not.append(extra)
        else:
            pool = [p for p in kept if p not in must_reach]
            if pool:
                must_reach.append(rng.choice(pool))
    return must_reach, must_not, [], {"direct": main == e}


def _plan_unchanged(g: Graph, rng: Rng, closure, reach, unreach, eligible):
    n_reach, n_not = rng.choice([(1, 1), (2, 1), (1, 2)])
    must_reach = rng.sample(reach, min(n_reach, len(reach)))
    must_not = rng.sample(unreach, min(n_not, len(unreach)))
    return must_reach, must_not, [], {}


def _plan_impossible(g: Graph, rng: Rng, closure, reach, unreach, eligible):
    kind = rng.choice(["both_sides", "no_add", "no_cut"])
    if kind == "both_sides":
        must_reach, must_not = [rng.choice(unreach)], [rng.choice(reach)]
        if rng.chance(0.5):
            if rng.chance(0.5):
                extra = rng.choice(reach)
                if extra not in must_not and extra not in must_reach:
                    must_reach.append(extra)
            else:
                extra = rng.choice(unreach)
                if extra not in must_reach and extra not in must_not:
                    must_not.append(extra)
        return must_reach, must_not, [], {}
    if kind == "no_add":
        main = rng.choice(unreach)
        fixers = [e for e in eligible if _fixes(closure, e, main)]
        non_fixers = [e for e in eligible if not _fixes(closure, e, main)]
        cands: List[Edge] = []
        must_not: List[Edge] = []
        if fixers and rng.chance(0.6):
            d = rng.choice(fixers)
            cd = g.with_edge(*d).closure()
            bad = [p for p in unreach if _has(cd, p) and p != main]
            if bad:
                must_not.append(rng.choice(bad))
                cands.append(d)
        if not must_not:
            pool = [p for p in unreach if p != main]
            if not pool:
                return None
            must_not.append(rng.choice(pool))
        return [main], must_not, cands, {"fill_from": non_fixers}
    # no_cut: a forbidden reachability that no single removal can break without
    # destroying something that must be kept.
    main = rng.choice(reach)
    cuts = [x for x in g.edges if not g.without_edge(*x).reaches(*main)]
    if len(cuts) > 2:
        return None
    must_reach: List[Edge] = []
    for x in cuts:
        cx = g.without_edge(*x).closure()
        lost = [p for p in reach if not _has(cx, p) and p != main and p not in must_reach]
        if not lost:
            return None
        must_reach.append(rng.choice(lost))
    if not must_reach:
        pool = [p for p in reach if p != main]
        if not pool:
            return None
        must_reach.append(rng.choice(pool))
    return must_reach, [main], [], {}


_PLANNERS = {
    "add": _plan_add,
    "remove": _plan_remove,
    "unchanged": _plan_unchanged,
    "impossible": _plan_impossible,
}


def propose_repair(g: Graph, rng: Rng, category: str) -> Optional[Dict]:
    """One attempt at a repair task of the requested category; None if it did not work out.

    The returned spec is always re-checked by the exhaustive solver, so a planner bug
    cannot produce a wrong label; it can only lower the acceptance rate.
    """
    closure = g.closure()
    pairs = all_pairs(g.n)
    reach = [p for p in pairs if _has(closure, p)]
    unreach = [p for p in pairs if not _has(closure, p)]
    eligible = [p for p in pairs if g.can_add(*p)]
    if len(eligible) < 2 or not reach or not unreach:
        return None
    plan = _PLANNERS[category](g, rng, closure, reach, unreach, eligible)
    if plan is None:
        return None
    must_reach, must_not, cands, meta = plan
    if not must_reach or not must_not or len(must_reach) + len(must_not) > 3:
        return None
    if set(must_reach) & set(must_not):
        return None
    k_add = rng.randint(2, min(4, len(eligible)))
    fill_pool = meta.pop("fill_from", eligible)
    for e in rng.shuffled(fill_pool):
        if len(cands) >= k_add:
            break
        if e not in cands:
            cands.append(e)
    if len(cands) < 2:
        return None
    additions = sorted(set(cands))
    sol = solve_repair(g, must_reach, must_not, additions)
    if repair_category(sol) != category:
        return None
    violated_reach = any(not _has(closure, p) for p in must_reach)
    violated_not = any(_has(closure, p) for p in must_not)
    if violated_reach and violated_not:
        sides = "both"
    else:
        sides = "reach" if violated_reach else "not" if violated_not else "none"
    info = {
        "category": category,
        "direct": bool(meta.get("direct", False)),
        "violated_sides": sides,
    }
    return {
        "must_reach": sorted(must_reach),
        "must_not": sorted(must_not),
        "additions": additions,
        "solution": sol,
        "info": info,
    }


def repair_twin_mods(g: Graph, spec: Dict) -> List[Tuple[Mod, RepairSolution]]:
    """Single-edge image changes that keep the instruction valid but change every correct answer."""
    base_key = spec["solution"].answer_key()
    additions = spec["additions"]
    listed = set(additions) | {(v, u) for (u, v) in additions}
    out = []
    for mod in single_edge_mods(g):
        kind, u, v = mod
        if kind == "add" and (u, v) in listed:
            continue  # a listed addition must stay absent (and addable) in the twin image
        g2 = apply_mod(g, mod)
        sol2 = solve_repair(g2, spec["must_reach"], spec["must_not"], additions)
        if base_key.isdisjoint(sol2.answer_key()):
            out.append((mod, sol2))
    return out


# ------------------------------------------------------------ non-edit QA proposals
def _pick_by_length(g: Graph, rng: Rng, reach: List[Edge]) -> Edge:
    by_len: Dict[int, List[Edge]] = {}
    for p in reach:
        by_len.setdefault(g.shortest_path_len(*p) or 0, []).append(p)
    length = rng.choice(sorted(by_len))
    return rng.choice(by_len[length])


def propose_reachability(g: Graph, rng: Rng, base_answer: bool) -> Optional[Dict]:
    closure = g.closure()
    pairs = all_pairs(g.n)
    reach = [p for p in pairs if _has(closure, p)]
    unreach = [p for p in pairs if not _has(closure, p)]
    if base_answer:
        if not reach:
            return None
        s, t = _pick_by_length(g, rng, reach)
        twins = [
            ("remove", u, v)
            for (u, v) in g.edges
            if not g.without_edge(u, v).reaches(s, t)
        ]
        info = {"path_len": g.shortest_path_len(s, t), "reverse_reachable": _has(closure, (t, s))}
    else:
        if not unreach:
            return None
        hard = [p for p in unreach if _has(closure, (p[1], p[0]))]
        s, t = rng.choice(hard) if hard and rng.chance(0.6) else rng.choice(unreach)
        fixers = [e for e in pairs if g.can_add(*e) and _fixes(closure, e, (s, t))]
        nondirect = [e for e in fixers if e != (s, t)]
        direct = [e for e in fixers if e == (s, t)]
        ordered = rng.shuffled(nondirect) + direct if rng.chance(0.8) else rng.shuffled(fixers)
        twins = [("add", u, v) for (u, v) in ordered]
        info = {"path_len": None, "reverse_reachable": _has(closure, (t, s))}
    if base_answer:
        twins = rng.shuffled(twins)
    return {"src": s, "dst": t, "answer": base_answer, "twin_mods": twins, "info": info}


def critical_answer(g: Graph, s: int, t: int, edge: Edge) -> bool:
    """True iff removing `edge` makes t unreachable from s. Requires s->t reachable and edge present."""
    if not g.reaches(s, t) or not g.has_edge(*edge):
        raise ValueError("critical-edge question is ill-posed on this graph")
    return not g.without_edge(*edge).reaches(s, t)


def propose_critical_edge(g: Graph, rng: Rng, base_answer: bool) -> Optional[Dict]:
    closure = g.closure()
    reach = [p for p in all_pairs(g.n) if _has(closure, p)]
    if not reach:
        return None
    options: List[Tuple[Edge, Edge]] = []  # (pair, edge)
    for e in g.edges:
        ce = g.without_edge(*e).closure()
        d, x = e
        for p in reach:
            s, t = p
            on_walk = (s == d or _has(closure, (s, d))) and (x == t or _has(closure, (x, t)))
            if not on_walk:
                continue
            lost = not _has(ce, p)
            if lost == base_answer:
                options.append((p, e))
    if not options:
        return None
    nondirect = [o for o in options if o[0] != o[1]]
    (s, t), e = rng.choice(nondirect) if nondirect and rng.chance(0.85) else rng.choice(options)
    twins: List[Mod] = []
    if base_answer:
        # Twin: add a bypass so that the edge stops being critical.
        for (u, v) in all_pairs(g.n):
            if not g.can_add(u, v):
                continue
            g2 = g.with_edge(u, v)
            if g2.without_edge(*e).reaches(s, t):
                twins.append(("add", u, v))
    else:
        # Twin: remove an edge of the alternative route so that the edge becomes critical.
        for f in g.edges:
            if f == e:
                continue
            g2 = g.without_edge(*f)
            if not g2.reaches(s, t):
                continue
            if not g2.without_edge(*e).reaches(s, t):
                twins.append(("remove",) + f)
    info = {"path_len": g.shortest_path_len(s, t), "direct": (s, t) == e}
    return {
        "src": s,
        "dst": t,
        "edge": e,
        "answer": base_answer,
        "twin_mods": rng.shuffled(twins),
        "info": info,
    }


def propose_edge_exists(g: Graph, rng: Rng) -> Dict:
    edges = g.edges
    answer = rng.chance(0.5)
    if answer:
        s, t = rng.choice(edges)
        kind = "edge"
    else:
        reversed_edges = [(v, u) for (u, v) in edges]
        non_adjacent = [p for p in all_pairs(g.n) if not g.has_edge(*p) and not g.has_edge(p[1], p[0])]
        if non_adjacent and rng.chance(0.5):
            s, t = rng.choice(non_adjacent)
            kind = "non_adjacent"
        else:
            s, t = rng.choice(reversed_edges)
            kind = "reversed"
    return {"src": s, "dst": t, "answer": answer, "info": {"kind": kind}}


def propose_shortest_path(g: Graph, rng: Rng) -> Dict:
    closure = g.closure()
    pairs = all_pairs(g.n)
    reach = [p for p in pairs if _has(closure, p)]
    unreach = [p for p in pairs if not _has(closure, p)]
    if reach and (not unreach or rng.chance(0.8)):
        s, t = _pick_by_length(g, rng, reach)
    else:
        s, t = rng.choice(unreach)
    return {"src": s, "dst": t, "answer": g.shortest_path_len(s, t), "info": {}}
