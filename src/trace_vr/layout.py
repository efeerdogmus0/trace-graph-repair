"""Node placement, edge routing and the geometric readability checks.

A drawing is only accepted if it passes every check below, so that the picture states the
graph unambiguously:

* nodes keep a minimum distance from each other and from the canvas border;
* no edge passes under or close to a node it does not belong to;
* every edge is long enough for its shaft and arrowhead to be visible;
* two edges at the same node leave it in clearly different directions;
* edges cross at most once, at a clear angle, away from arrowheads and tails;
* edges that do not cross do not run so close that they look connected.

Each edge's route depends only on the node positions and on that edge, never on other
edges. A twin image (one edge added or removed) therefore keeps every other edge pixel
for pixel in place.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import geometry as geo
from .graphs import Edge, Graph, bits
from .rng import Rng
from .styles import Style

Point = Tuple[float, float]

NODE_GAP = 14.0  # free space between two node shapes
BORDER_GAP = 8.0  # free space between a node shape and the canvas border
EDGE_NODE_CLEAR = 9.0  # free space between an edge and a foreign node shape
MIN_SHAFT = 14.0  # visible shaft length in addition to the arrowhead
MAX_CROSSINGS = 8  # drawings with more edge crossings than this are rejected
# Minimum angle between two edges at one node, as cos^2 of the angle. Two arrowheads need
# the most room, two plain tails the least.
COS2_HEAD_HEAD = 0.834565  # 24 deg
COS2_HEAD_TAIL = 0.904508  # 18 deg
COS2_TAIL_TAIL = 0.949397  # 13 deg
SIN2_CROSS_ANGLE = 0.178606  # sin^2(25 deg): minimum crossing angle
CROSS_HEAD_CLEAR = 8.0  # a crossing stays this far (plus arrow length) from arrow tips
CROSS_TAIL_CLEAR = 8.0
MIN_EDGE_GAP = 7.0  # non-crossing edges of different nodes stay at least this far apart
CURVE_SEGMENTS = 12

FAIL_REASONS = (
    "positions",
    "route",
    "node_angle",
    "adjacent_cross",
    "multi_cross",
    "shallow_cross",
    "cross_near_head",
    "cross_near_tail",
    "edges_too_close",
    "too_many_crossings",
)


class Layout:
    __slots__ = ("pos", "routes", "crossings", "layout_name")

    def __init__(self, pos, routes, crossings, layout_name):
        self.pos: List[Tuple[int, int]] = pos
        self.routes: Dict[Edge, List[Point]] = routes
        self.crossings: int = crossings
        self.layout_name: str = layout_name


# ------------------------------------------------------------------ node positions
def _min_sep(style: Style) -> float:
    return 2 * style.extent + NODE_GAP


def _margin(style: Style) -> float:
    return style.extent + BORDER_GAP


def _min_linked_sep(style: Style) -> float:
    """Connected nodes must be far enough apart for a visible shaft and arrowhead."""
    return max(_min_sep(style), 2 * style.extent + style.arrow_len + MIN_SHAFT + 5.0)


def _linked_pairs(g: Graph) -> set:
    return {(min(u, v), max(u, v)) for u, v in g.edges}


def _spread(points: Sequence[Point], size: int, style: Style, g: Graph, rounds: int = 80) -> List[Point]:
    """Push nodes that are too close apart (bounded number of rounds, deterministic)."""
    lo, hi = _margin(style), size - _margin(style)
    linked = _linked_pairs(g)
    plain, wide = _min_sep(style) + 1.5, _min_linked_sep(style) + 1.5
    pts = [[min(max(x, lo), hi), min(max(y, lo), hi)] for x, y in points]
    for _ in range(rounds):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx, dy = pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]
                d = math.sqrt(dx * dx + dy * dy)
                want = wide if (i, j) in linked else plain
                if d >= want:
                    continue
                if d < 1e-6:
                    dx, dy, d = 1.0 + i, 1.0 + j, math.sqrt((1.0 + i) ** 2 + (1.0 + j) ** 2)
                push = (want - d) / 2 + 0.25
                for k, sign in ((i, -1.0), (j, 1.0)):
                    pts[k][0] = min(max(pts[k][0] + sign * dx / d * push, lo), hi)
                    pts[k][1] = min(max(pts[k][1] + sign * dy / d * push, lo), hi)
                moved = True
        if not moved:
            break
    return [(p[0], p[1]) for p in pts]


def _snap(points: Sequence[Point], size: int, style: Style, g: Graph) -> Optional[List[Tuple[int, int]]]:
    lo, hi = _margin(style), size - _margin(style)
    out = []
    for x, y in _spread(points, size, style, g):
        out.append((int(round(min(max(x, lo), hi))), int(round(min(max(y, lo), hi)))))
    sep2 = _min_sep(style) ** 2
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if geo.dist2(out[i], out[j]) < sep2:
                return None
    return out


def _grid(g: Graph, rng: Rng, size: int, style: Style) -> List[Point]:
    n = g.n
    c = int(math.ceil(math.sqrt(n)))
    options = [(c, -(-n // c)), (c + 1, -(-n // (c + 1)))]
    if c > 2:
        options.append((c - 1, -(-n // (c - 1))))
    cols, rows = rng.choice(options)
    if rng.chance(0.5):
        cols, rows = rows, cols
    m = _margin(style)
    cw, ch = (size - 2 * m) / cols, (size - 2 * m) / rows
    jx = max(0.0, (cw - _min_sep(style)) / 2)
    jy = max(0.0, (ch - _min_sep(style)) / 2)
    cells = rng.sample([(i, j) for i in range(cols) for j in range(rows)], n)
    return [
        (m + (i + 0.5) * cw + rng.uniform(-jx, jx), m + (j + 0.5) * ch + rng.uniform(-jy, jy))
        for i, j in cells
    ]


def _circle(g: Graph, rng: Rng, size: int, style: Style) -> List[Point]:
    n = g.n
    order = rng.shuffled(range(n))
    phase = rng.random()
    radius = size / 2 - _margin(style)
    out: List[Point] = [(0.0, 0.0)] * n
    for k, node in enumerate(order):
        ang = 2 * math.pi * (k + phase) / n
        r = radius * rng.uniform(0.84, 1.0)
        out[node] = (size / 2 + r * math.cos(ang), size / 2 + r * math.sin(ang))
    return out


def layer_assignment(g: Graph, rng: Rng) -> List[List[int]]:
    """Longest-path layering after dropping DFS back edges (so cycles are allowed)."""
    n = g.n
    succ = [rng.shuffled(bits(g.adj[u])) for u in range(n)]
    state = [0] * n
    back = set()
    for root in rng.shuffled(range(n)):
        if state[root]:
            continue
        state[root] = 1
        stack = [(root, 0)]
        while stack:
            u, i = stack[-1]
            if i < len(succ[u]):
                stack[-1] = (u, i + 1)
                v = succ[u][i]
                if state[v] == 0:
                    state[v] = 1
                    stack.append((v, 0))
                elif state[v] == 1:
                    back.add((u, v))
            else:
                state[u] = 2
                stack.pop()
    dag = [e for e in g.edges if e not in back]
    indeg = [0] * n
    out: List[List[int]] = [[] for _ in range(n)]
    for u, v in dag:
        indeg[v] += 1
        out[u].append(v)
    layer = [0] * n
    queue = [u for u in range(n) if indeg[u] == 0]
    while queue:
        u = queue.pop()
        for v in out[u]:
            layer[v] = max(layer[v], layer[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return [[u for u in range(n) if layer[u] == k] for k in range(max(layer) + 1)]


def _layered(g: Graph, rng: Rng, size: int, style: Style) -> List[Point]:
    layers = layer_assignment(g, rng)
    n_layers = len(layers)
    m = _margin(style)
    span = size - 2 * m
    horizontal = rng.chance(0.5)
    cross: Dict[int, float] = {}
    out: List[Point] = [(0.0, 0.0)] * g.n
    for li, nodes in enumerate(layers):
        ordered = rng.shuffled(nodes)
        if li > 0:
            def bary(u: int) -> float:
                ps = [cross[p] for p in cross if g.has_edge(p, u)]
                return sum(ps) / len(ps) if ps else span / 2

            ordered = sorted(ordered, key=bary)
        k = len(ordered)
        jit = min(14.0, max(0.0, (span / k - _min_sep(style)) / 2))
        main = m + (span * li / (n_layers - 1) if n_layers > 1 else span / 2)
        for i, u in enumerate(ordered):
            c = m + span * (i + 0.5) / k + rng.uniform(-jit, jit)
            cross[u] = c
            a = main + rng.uniform(-8.0, 8.0)
            out[u] = (a, c) if horizontal else (c, a)
    return out


def _force(g: Graph, rng: Rng, size: int, style: Style) -> List[Point]:
    """Fruchterman-Reingold with a fixed iteration count (sqrt and basic arithmetic only)."""
    n = g.n
    pos = [[rng.random(), rng.random()] for _ in range(n)]
    und = set()
    for u, v in g.edges:
        und.add((min(u, v), max(u, v)))
    k = math.sqrt(1.0 / n)
    temp = 0.12
    for _ in range(140):
        disp = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < 1e-4:
                    dx, dy, d = 1e-4 * (i + 1), 1e-4 * (j + 1), 1e-4 * math.sqrt((i + 1) ** 2 + (j + 1) ** 2)
                f = k * k / d
                disp[i][0] += dx / d * f
                disp[i][1] += dy / d * f
                disp[j][0] -= dx / d * f
                disp[j][1] -= dy / d * f
        for i, j in sorted(und):
            dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1e-4:
                continue
            f = d * d / k
            disp[i][0] -= dx / d * f
            disp[i][1] -= dy / d * f
            disp[j][0] += dx / d * f
            disp[j][1] += dy / d * f
        for i in range(n):
            dx, dy = disp[i]
            d = math.sqrt(dx * dx + dy * dy)
            if d > 0:
                step = min(d, temp)
                pos[i][0] += dx / d * step
                pos[i][1] += dy / d * step
        temp *= 0.97
    xs, ys = [p[0] for p in pos], [p[1] for p in pos]
    m = _margin(style)
    span = size - 2 * m
    wx, wy = max(xs) - min(xs) or 1.0, max(ys) - min(ys) or 1.0
    return [(m + (p[0] - min(xs)) / wx * span, m + (p[1] - min(ys)) / wy * span) for p in pos]


_PROPOSERS = {"grid": _grid, "circle": _circle, "layered": _layered, "force": _force}


def propose_positions(g: Graph, name: str, rng: Rng, size: int, style: Style):
    return _snap(_PROPOSERS[name](g, rng, size, style), size, style, g)


# ------------------------------------------------------------------- edge routing
def boundary_point(center: Point, toward: Point, style: Style, gap: float) -> Point:
    """Point just outside the node shape at `center`, in the direction of `toward`."""
    ux, uy = geo.unit(toward[0] - center[0], toward[1] - center[1])
    if style.node_shape == "circle":
        r = style.node_radius + gap
    else:
        r = (style.node_radius + gap) / max(abs(ux), abs(uy))
    return (center[0] + ux * r, center[1] + uy * r)


def _bend_order(u: int, v: int, style: Style) -> List[float]:
    if style.curved:
        s = 1.0 if (u * 7 + v * 13) % 2 == 0 else -1.0
        return [0.16 * s, -0.16 * s, 0.28 * s, -0.28 * s, 0.40 * s, -0.40 * s, 0.0]
    return [0.0, 0.14, -0.14, 0.24, -0.24, 0.34, -0.34, 0.44, -0.44]


def _make_route(cu: Point, cv: Point, bend: float, style: Style) -> Optional[List[Point]]:
    dx, dy = cv[0] - cu[0], cv[1] - cu[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return None
    mx, my = (cu[0] + cv[0]) / 2, (cu[1] + cv[1]) / 2
    ctrl = (mx - dy / length * bend * length, my + dx / length * bend * length)
    start = boundary_point(cu, ctrl, style, 1.0)
    end = boundary_point(cv, ctrl, style, 2.0)
    if geo.dist(start, end) < style.arrow_len + MIN_SHAFT:
        return None
    if bend == 0.0:
        return [start, end]
    pts = []
    for i in range(CURVE_SEGMENTS + 1):
        t = i / CURVE_SEGMENTS
        a, b, c = (1 - t) * (1 - t), 2 * t * (1 - t), t * t
        pts.append((a * start[0] + b * ctrl[0] + c * end[0], a * start[1] + b * ctrl[1] + c * end[1]))
    return pts


def route_edge(pos: Sequence[Tuple[int, int]], u: int, v: int, style: Style, size: int) -> Optional[List[Point]]:
    """First route (straight, then increasingly bent) that keeps clear of all other nodes."""
    clear2 = (style.extent + EDGE_NODE_CLEAR) ** 2
    for bend in _bend_order(u, v, style):
        route = _make_route(pos[u], pos[v], bend, style)
        if route is None:
            continue
        x0, y0, x1, y1 = geo.bbox(route)
        if x0 < 3 or y0 < 3 or x1 > size - 3 or y1 > size - 3:
            continue
        if all(geo.poly_point_dist2(route, pos[k]) >= clear2 for k in range(len(pos)) if k != u and k != v):
            return route
    return None


def route_all(g: Graph, pos: Sequence[Tuple[int, int]], style: Style, size: int) -> Optional[Dict[Edge, List[Point]]]:
    routes: Dict[Edge, List[Point]] = {}
    for u, v in g.edges:
        r = route_edge(pos, u, v, style, size)
        if r is None:
            return None
        routes[(u, v)] = r
    return routes


# --------------------------------------------------------------- readability checks
def _departure(route: Sequence[Point], at_start: bool) -> Point:
    if at_start:
        return (route[1][0] - route[0][0], route[1][1] - route[0][1])
    return (route[-2][0] - route[-1][0], route[-2][1] - route[-1][1])


def check_routes(routes: Dict[Edge, List[Point]], style: Style) -> Tuple[bool, int, str]:
    """Returns (ok, number of edge crossings, reason for failure)."""
    edges = sorted(routes)
    # Two edges at the same node must leave it in clearly different directions.
    at_node: Dict[int, List[Tuple[Point, bool]]] = {}
    for (u, v) in edges:
        at_node.setdefault(u, []).append((_departure(routes[(u, v)], True), False))
        at_node.setdefault(v, []).append((_departure(routes[(u, v)], False), True))
    for ends in at_node.values():
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                (a, head_a), (b, head_b) = ends[i], ends[j]
                if head_a and head_b:
                    limit = COS2_HEAD_HEAD
                elif head_a or head_b:
                    limit = COS2_HEAD_TAIL
                else:
                    limit = COS2_TAIL_TAIL
                dot = a[0] * b[0] + a[1] * b[1]
                if dot > 0 and dot * dot > limit * (a[0] ** 2 + a[1] ** 2) * (b[0] ** 2 + b[1] ** 2):
                    return False, 0, "node_angle"
    boxes = {e: geo.bbox(routes[e]) for e in edges}
    head_clear2 = (style.arrow_len + CROSS_HEAD_CLEAR) ** 2
    crossings = 0
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e1, e2 = edges[i], edges[j]
            if geo.bbox_gap2(boxes[e1], boxes[e2]) > MIN_EDGE_GAP ** 2:
                continue
            r1, r2 = routes[e1], routes[e2]
            hits = geo.poly_intersections(r1, r2)
            if set(e1) & set(e2):
                if hits:
                    return False, 0, "adjacent_cross"
                continue
            if len(hits) > 1:
                return False, 0, "multi_cross"
            if not hits:
                if geo.poly_min_dist2(r1, r2) < MIN_EDGE_GAP ** 2:
                    return False, 0, "edges_too_close"
                continue
            pt, si, sj = hits[0]
            a = (r1[si + 1][0] - r1[si][0], r1[si + 1][1] - r1[si][1])
            b = (r2[sj + 1][0] - r2[sj][0], r2[sj + 1][1] - r2[sj][1])
            cross = a[0] * b[1] - a[1] * b[0]
            if cross * cross < SIN2_CROSS_ANGLE * (a[0] ** 2 + a[1] ** 2) * (b[0] ** 2 + b[1] ** 2):
                return False, 0, "shallow_cross"
            if geo.dist2(pt, r1[-1]) < head_clear2 or geo.dist2(pt, r2[-1]) < head_clear2:
                return False, 0, "cross_near_head"
            if geo.dist2(pt, r1[0]) < CROSS_TAIL_CLEAR ** 2 or geo.dist2(pt, r2[0]) < CROSS_TAIL_CLEAR ** 2:
                return False, 0, "cross_near_tail"
            crossings += 1
    return True, crossings, ""


def check_positions(pos: Sequence[Tuple[int, int]], style: Style, size: int) -> bool:
    lo, hi = _margin(style) - 0.51, size - _margin(style) + 0.51
    if any(not (lo <= x <= hi and lo <= y <= hi) for x, y in pos):
        return False
    sep2 = _min_sep(style) ** 2
    return all(
        geo.dist2(pos[i], pos[j]) >= sep2 for i in range(len(pos)) for j in range(i + 1, len(pos))
    )


def layout_for_positions(g: Graph, pos, style: Style, size: int, layout_name: str) -> Tuple[Optional[Layout], str]:
    """Route and check a graph on fixed node positions (used for twin images)."""
    routes = route_all(g, pos, style, size)
    if routes is None:
        return None, "route"
    ok, crossings, reason = check_routes(routes, style)
    if not ok:
        return None, reason
    if crossings > MAX_CROSSINGS:
        return None, "too_many_crossings"
    return Layout(list(pos), routes, crossings, layout_name), ""


def find_layout(
    g: Graph, style: Style, rng: Rng, size: int, budget: int = 180, keep_best_of: int = 3
) -> Tuple[Optional[Layout], Dict[str, int]]:
    """Pick a layout algorithm at random (by style weights) and try proposals from it.

    Among the first few proposals that pass every check, the one with the fewest crossings
    is used. Only if an algorithm produces nothing acceptable is the next one tried, so the
    mix of layouts follows the style weights wherever the graph allows it.
    """
    per_algorithm = max(45, budget // len(style.layouts))
    remaining = [(n, w) for n, w in style.layouts]
    order: List[str] = []
    pick = rng.child("algorithm-order")
    while remaining:
        name = pick.weighted_choice([n for n, _ in remaining], [w for _, w in remaining])
        order.append(name)
        remaining = [(n, w) for n, w in remaining if n != name]
    fails: Dict[str, int] = {}
    for name in order:
        best: Optional[Layout] = None
        passed = 0
        for i in range(per_algorithm):
            prng = rng.child("proposal", name, i)
            pos = propose_positions(g, name, prng, size, style)
            if pos is None:
                fails["positions"] = fails.get("positions", 0) + 1
                continue
            layout, reason = layout_for_positions(g, pos, style, size, name)
            if layout is None:
                fails[reason] = fails.get(reason, 0) + 1
                continue
            passed += 1
            if best is None or layout.crossings < best.crossings:
                best = layout
            if passed >= keep_best_of or best.crossings == 0:
                break
        if best is not None:
            return best, fails
    return None, fails
