"""Drawing with Pillow, plus pixel-level checks that read the finished image back.

The pixel checks are deliberately separate from the geometric checks in `layout.py`:
they look at the actual PNG, so a drawing bug (edge not painted, arrowhead on the wrong
end, missing label) is caught even if the geometry was accepted.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from . import geometry as geo
from .styles import Style

Point = Tuple[float, float]


def assets_dir() -> Path:
    override = os.environ.get("TRACE_VR_ASSETS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "assets"


_FONTS: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def load_font(font_file: str, size: int) -> ImageFont.FreeTypeFont:
    key = (font_file, size)
    if key not in _FONTS:
        path = assets_dir() / "fonts" / font_file
        if not path.exists():
            raise FileNotFoundError("font not found: %s (set TRACE_VR_ASSETS?)" % path)
        _FONTS[key] = ImageFont.truetype(str(path), size)
    return _FONTS[key]


def _scaled(points: Sequence[Point], s: int) -> List[Point]:
    return [(x * s, y * s) for x, y in points]


def draw_diagram(
    pos: Sequence[Tuple[int, int]],
    labels: Sequence[str],
    routes: Dict[Tuple[int, int], Sequence[Point]],
    style: Style,
    size: int,
    supersample: int = 3,
) -> Image.Image:
    """Render one graph image. Coordinates are in final pixels; drawing is supersampled."""
    s = supersample
    img = Image.new("RGB", (size * s, size * s), style.bg)
    d = ImageDraw.Draw(img)
    edge_w = max(1, int(round(style.edge_width * s)))

    for edge in sorted(routes):
        pts = list(routes[edge])
        tip = pts[-1]
        ux, uy = geo.unit(tip[0] - pts[-2][0], tip[1] - pts[-2][1])
        base = (tip[0] - ux * style.arrow_len, tip[1] - uy * style.arrow_len)
        left = (base[0] - uy * style.arrow_half_width, base[1] + ux * style.arrow_half_width)
        right = (base[0] + uy * style.arrow_half_width, base[1] - ux * style.arrow_half_width)
        if style.arrow_kind == "filled":
            shaft = geo.trim_end(pts, style.arrow_len * 0.8)
            d.line(_scaled(shaft, s), fill=style.stroke, width=edge_w, joint="curve")
            d.polygon(_scaled([tip, left, right], s), fill=style.stroke)
        else:  # "vee": open arrowhead, shaft runs to the tip
            d.line(_scaled(pts, s), fill=style.stroke, width=edge_w, joint="curve")
            d.line(_scaled([left, tip, right], s), fill=style.stroke, width=edge_w, joint="curve")

    font = load_font(style.font_file, style.font_size * s)
    r = style.node_radius
    outline_w = max(1, int(round(style.outline_width * s)))
    for (x, y), label in zip(pos, labels):
        box = [(x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s]
        if style.node_shape == "circle":
            d.ellipse(box, fill=style.node_fill, outline=style.stroke, width=outline_w)
        elif style.node_shape == "rounded":
            d.rounded_rectangle(box, radius=int(r * 0.35 * s), fill=style.node_fill, outline=style.stroke, width=outline_w)
        else:
            d.rectangle(box, fill=style.node_fill, outline=style.stroke, width=outline_w)
        d.text((x * s, y * s), label, font=font, fill=style.text, anchor="mm")

    return img.resize((size, size), Image.LANCZOS)


# ------------------------------------------------------------------ pixel checks
def _l1(a: Sequence[int], b: Sequence[int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


class _Reader:
    def __init__(self, img: Image.Image, style: Style) -> None:
        self.img = img.convert("RGB")
        self.px = self.img.load()
        self.w, self.h = self.img.size
        self.style = style
        self.scale = float(_l1(style.stroke, style.bg)) or 1.0

    def ink(self, x: float, y: float) -> float:
        """0 = background colour, 1 = stroke colour (clamped), at the nearest pixel."""
        xi = min(max(int(round(x)), 0), self.w - 1)
        yi = min(max(int(round(y)), 0), self.h - 1)
        return min(1.0, _l1(self.px[xi, yi], self.style.bg) / self.scale)

    def max_ink(self, x: float, y: float, radius: int = 1) -> float:
        return max(
            self.ink(x + dx, y + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
        )

    def ink_mass(self, x: float, y: float, radius: int) -> float:
        total = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    total += self.ink(x + dx, y + dy)
        return total


def image_checks(
    img: Image.Image,
    style: Style,
    nodes: Dict[str, Sequence[float]],
    edges: Sequence[Dict],
) -> List[str]:
    """Read the rendered image back and compare it with the intended drawing.

    `nodes` maps label -> [x, y]; each edge is {"src", "dst", "points"}. Returns a list of
    problems (empty when the image agrees with the geometry).
    """
    problems: List[str] = []
    rd = _Reader(img, style)
    routes = [[tuple(p) for p in e["points"]] for e in edges]

    for e, route in zip(edges, routes):
        name = "%s->%s" % (e["src"], e["dst"])
        for frac in (0.3, 0.5, 0.7):
            p = geo.point_along(route, frac)
            if rd.max_ink(p[0], p[1], 1) < 0.5:
                problems.append("edge %s not painted at %.0f%% of its length" % (name, frac * 100))
        # Arrowhead: across the edge, three quarters of an arrowhead length before the tip,
        # there must be clearly more ink than a bare shaft would leave. Other edges cannot
        # reach this spot (minimum angles at nodes, crossings kept away from arrow tips).
        tip = route[-1]
        ux, uy = geo.unit(tip[0] - route[-2][0], tip[1] - route[-2][1])
        cx, cy = tip[0] - ux * 0.75 * style.arrow_len, tip[1] - uy * 0.75 * style.arrow_len
        reach = int(round(style.arrow_half_width + 1))
        mass = sum(rd.ink(cx - uy * k, cy + ux * k) for k in range(-reach, reach + 1))
        if mass < style.edge_width + 2.0:
            problems.append("edge %s: no visible arrowhead at its target end" % name)

    r = style.node_radius
    half = max(3, int(r * 0.5))
    for label, (x, y) in nodes.items():
        text_like = 0
        count = 0
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                c = rd.px[min(max(int(x) + dx, 0), rd.w - 1), min(max(int(y) + dy, 0), rd.h - 1)]
                count += 1
                if _l1(c, style.text) < _l1(c, style.node_fill):
                    text_like += 1
        if text_like < 0.04 * count:
            problems.append("node %s: label not visible" % label)

    # Where nothing should be drawn, the image must show background.
    present = {(e["src"], e["dst"]) for e in edges}
    labels = sorted(nodes)
    clear2 = (style.extent + 6.0) ** 2
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if (a, b) in present or (b, a) in present:
                continue
            mid = ((nodes[a][0] + nodes[b][0]) / 2.0, (nodes[a][1] + nodes[b][1]) / 2.0)
            if any(geo.dist2(mid, tuple(nodes[c])) < clear2 for c in labels):
                continue
            if any(geo.poly_point_dist2(route, mid) < (style.arrow_half_width + 5.0) ** 2 for route in routes):
                continue
            if rd.max_ink(mid[0], mid[1], 1) > 0.15:
                problems.append("unexpected ink between unconnected nodes %s and %s" % (a, b))
    return problems
