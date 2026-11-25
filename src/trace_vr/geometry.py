"""Plain 2D geometry on float tuples.

Decisions (accept/reject a layout) only use +, -, *, / and comparisons on squared
quantities, plus sqrt where a length is needed. These are exactly specified by IEEE 754,
so the same inputs give the same decisions on every platform.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Polyline = Sequence[Point]


def dist2(a: Point, b: Point) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy


def dist(a: Point, b: Point) -> float:
    return math.sqrt(dist2(a, b))


def point_seg_dist2(p: Point, a: Point, b: Point) -> float:
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    den = abx * abx + aby * aby
    if den == 0:
        return apx * apx + apy * apy
    t = (apx * abx + apy * aby) / den
    if t < 0:
        t = 0.0
    elif t > 1:
        t = 1.0
    dx, dy = apx - t * abx, apy - t * aby
    return dx * dx + dy * dy


def seg_intersection(a: Point, b: Point, c: Point, d: Point) -> Optional[Point]:
    """Intersection point of segments ab and cd, or None. Parallel segments return None;
    overlapping collinear segments are caught separately by the minimum-distance check."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    den = rx * sy - ry * sx
    if den == 0:
        return None
    qx, qy = c[0] - a[0], c[1] - a[1]
    t = (qx * sy - qy * sx) / den
    u = (qx * ry - qy * rx) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (a[0] + t * rx, a[1] + t * ry)
    return None


def seg_seg_dist2(a: Point, b: Point, c: Point, d: Point) -> float:
    if seg_intersection(a, b, c, d) is not None:
        return 0.0
    return min(
        point_seg_dist2(a, c, d),
        point_seg_dist2(b, c, d),
        point_seg_dist2(c, a, b),
        point_seg_dist2(d, a, b),
    )


def bbox(poly: Polyline) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_gap2(b1, b2) -> float:
    """Squared distance between two axis-aligned boxes (0 if they overlap)."""
    dx = max(b1[0] - b2[2], b2[0] - b1[2], 0.0)
    dy = max(b1[1] - b2[3], b2[1] - b1[3], 0.0)
    return dx * dx + dy * dy


def poly_point_dist2(poly: Polyline, p: Point) -> float:
    return min(point_seg_dist2(p, poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def poly_min_dist2(p1: Polyline, p2: Polyline) -> float:
    best = float("inf")
    for i in range(len(p1) - 1):
        for j in range(len(p2) - 1):
            d = seg_seg_dist2(p1[i], p1[i + 1], p2[j], p2[j + 1])
            if d < best:
                best = d
    return best


def poly_intersections(p1: Polyline, p2: Polyline) -> List[Tuple[Point, int, int]]:
    """All (point, segment index in p1, segment index in p2) where the polylines meet."""
    out: List[Tuple[Point, int, int]] = []
    for i in range(len(p1) - 1):
        for j in range(len(p2) - 1):
            pt = seg_intersection(p1[i], p1[i + 1], p2[j], p2[j + 1])
            if pt is not None:
                # A crossing exactly at a shared vertex of consecutive segments is reported once.
                if out and dist2(out[-1][0], pt) < 1e-9:
                    continue
                out.append((pt, i, j))
    return out


def poly_length(poly: Polyline) -> float:
    return sum(dist(poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def point_along(poly: Polyline, frac: float) -> Point:
    """Point at the given fraction (0..1) of the polyline's arc length."""
    total = poly_length(poly)
    target = total * frac
    acc = 0.0
    for i in range(len(poly) - 1):
        seg = dist(poly[i], poly[i + 1])
        if acc + seg >= target and seg > 0:
            t = (target - acc) / seg
            return (poly[i][0] + t * (poly[i + 1][0] - poly[i][0]), poly[i][1] + t * (poly[i + 1][1] - poly[i][1]))
        acc += seg
    return poly[-1]


def trim_end(poly: Polyline, d: float) -> List[Point]:
    """Polyline shortened by arc length d from its end (at least the start point remains)."""
    pts = list(poly)
    remaining = d
    while len(pts) >= 2:
        seg = dist(pts[-2], pts[-1])
        if seg > remaining:
            t = (seg - remaining) / seg
            a, b = pts[-2], pts[-1]
            pts[-1] = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            return pts
        remaining -= seg
        pts.pop()
    return pts


def unit(dx: float, dy: float) -> Point:
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return (1.0, 0.0)
    return (dx / length, dy / length)
