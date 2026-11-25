"""Drawing styles.

A style never depends on the answer, the task or the graph. Which styles a split may use
is set in the dataset config; styles x and y are reserved for the style-OOD test.

Held-out factors in x and y: colour scheme, node shape or size, font family and weight,
arrowhead proportions and (x only) curved edges. The layout algorithm is NOT held out:
force-directed placement is used by every style because it is the only proposer that
draws larger graphs reliably.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class Style:
    style_id: str
    bg: RGB
    stroke: RGB
    node_fill: RGB
    text: RGB
    node_shape: str  # "circle" | "square" | "rounded"
    node_radius: int  # circle radius, or half the side of a square
    outline_width: float
    edge_width: float
    arrow_len: float
    arrow_half_width: float
    arrow_kind: str  # "filled" | "vee"
    font_file: str
    font_size: int
    curved: bool
    layouts: Tuple[Tuple[str, float], ...]

    @property
    def extent(self) -> float:
        """Radius of the smallest circle around the node centre that contains the node shape."""
        return float(self.node_radius) if self.node_shape == "circle" else self.node_radius * 1.42


_TRAIN_LAYOUTS = (("grid", 1.0), ("circle", 1.0), ("layered", 1.0), ("force", 2.0))
_OOD_LAYOUTS = (("force", 1.0),)

STYLES: Dict[str, Style] = {
    s.style_id: s
    for s in [
        Style("a", (255, 255, 255), (20, 20, 20), (255, 255, 255), (0, 0, 0), "circle", 22, 2.5, 2.5,
              15.0, 6.0, "filled", "DejaVuSans-Bold.ttf", 22, False, _TRAIN_LAYOUTS),
        Style("b", (247, 244, 234), (28, 49, 94), (221, 232, 247), (15, 25, 60), "rounded", 20, 2.5, 3.0,
              16.0, 6.5, "filled", "DejaVuSerif-Bold.ttf", 22, False, _TRAIN_LAYOUTS),
        Style("c", (238, 240, 242), (60, 60, 60), (255, 246, 200), (30, 30, 30), "circle", 21, 2.0, 2.0,
              15.0, 6.5, "vee", "DejaVuSansMono-Bold.ttf", 22, False, _TRAIN_LAYOUTS),
        Style("x", (28, 30, 38), (230, 230, 235), (58, 62, 80), (255, 255, 255), "square", 19, 2.5, 2.5,
              16.0, 6.0, "filled", "DejaVuSerif.ttf", 22, True, _OOD_LAYOUTS),
        Style("y", (255, 255, 255), (0, 110, 100), (255, 255, 255), (120, 20, 20), "circle", 23, 3.5, 2.0,
              18.0, 5.0, "filled", "DejaVuSansMono.ttf", 21, False, _OOD_LAYOUTS),
    ]
}


def get_style(style_id: str) -> Style:
    if style_id not in STYLES:
        raise KeyError("unknown style: %s" % style_id)
    return STYLES[style_id]
