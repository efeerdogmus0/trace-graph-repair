"""Risk 5 (visual pipeline): the picture must state the graph, and the checks must notice when it does not."""
from trace_vr import geometry as geo, layout, render
from trace_vr.styles import STYLES

A = STYLES["a"]
SIZE = 512


def test_edge_bends_around_a_node_in_its_way():
    pos = [(100, 256), (400, 256), (250, 256)]
    route = layout.route_edge(pos, 0, 1, A, SIZE)
    assert route is not None and len(route) > 2
    assert geo.poly_point_dist2(route, pos[2]) >= (A.extent + layout.EDGE_NODE_CLEAR) ** 2
    assert len(layout.route_edge(pos[:2], 0, 1, A, SIZE)) == 2  # nothing in the way: straight


def test_too_short_edge_is_refused():
    assert layout.route_edge([(100, 100), (160, 100)], 0, 1, A, SIZE) is None


def test_two_arrowheads_need_more_room_than_two_tails():
    pos = [(100, 100), (300, 100), (300, 160)]  # seen from node 0 the others are about 17 degrees apart
    heads = {(1, 0): layout.route_edge(pos, 1, 0, A, SIZE), (2, 0): layout.route_edge(pos, 2, 0, A, SIZE)}
    assert layout.check_routes(heads, A) == (False, 0, "node_angle")
    tails = {(0, 1): layout.route_edge(pos, 0, 1, A, SIZE), (0, 2): layout.route_edge(pos, 0, 2, A, SIZE)}
    assert layout.check_routes(tails, A) == (True, 0, "")


def test_clean_crossing_is_counted_and_crossing_at_an_arrowhead_is_refused():
    pos = [(100, 100), (400, 400), (400, 100), (100, 400)]
    routes = {(0, 1): layout.route_edge(pos, 0, 1, A, SIZE), (2, 3): layout.route_edge(pos, 2, 3, A, SIZE)}
    assert layout.check_routes(routes, A) == (True, 1, "")
    pos2 = [(100, 250), (300, 250), (265, 120), (265, 400)]  # second edge cuts across just before the arrow tip
    routes2 = {(0, 1): layout.route_edge(pos2, 0, 1, A, SIZE), (2, 3): layout.route_edge(pos2, 2, 3, A, SIZE)}
    ok, _, reason = layout.check_routes(routes2, A)
    assert not ok and reason == "cross_near_head"


# -- reading the finished image back ---------------------------------------------
POS = [(100, 100), (400, 100), (250, 400)]
LABELS = ["A", "B", "C"]
EDGES = [(0, 1), (1, 2), (2, 0)]


def _routes(edges):
    return {e: layout.route_edge(POS, e[0], e[1], A, SIZE) for e in edges}


def _geometry(routes):
    nodes = {LABELS[i]: list(POS[i]) for i in range(3)}
    edges = [{"src": LABELS[u], "dst": LABELS[v], "points": routes[(u, v)]} for (u, v) in sorted(routes)]
    return nodes, edges


def test_correct_image_passes_pixel_checks_in_every_style():
    for style in STYLES.values():
        routes = {e: layout.route_edge(POS, e[0], e[1], style, SIZE) for e in EDGES}
        img = render.draw_diagram(POS, LABELS, routes, style, SIZE)
        nodes = {LABELS[i]: list(POS[i]) for i in range(3)}
        edges = [{"src": LABELS[u], "dst": LABELS[v], "points": routes[(u, v)]} for (u, v) in sorted(routes)]
        assert render.image_checks(img, style, nodes, edges) == [], style.style_id


def test_missing_edge_in_the_picture_is_detected():
    img = render.draw_diagram(POS, LABELS, _routes(EDGES[:2]), A, SIZE)  # C->A not drawn
    problems = render.image_checks(img, A, *_geometry(_routes(EDGES)))
    assert any("C->A not painted" in p for p in problems)


def test_arrowhead_on_the_wrong_end_is_detected():
    img = render.draw_diagram(POS, LABELS, _routes(EDGES), A, SIZE)
    nodes, edges = _geometry(_routes(EDGES))
    for e in edges:  # claim the first edge runs B->A although it was drawn A->B
        if (e["src"], e["dst"]) == ("A", "B"):
            e["src"], e["dst"], e["points"] = "B", "A", list(reversed(e["points"]))
    problems = render.image_checks(img, A, nodes, edges)
    assert any("B->A: no visible arrowhead" in p for p in problems)


def test_extra_edge_in_the_picture_is_detected():
    img = render.draw_diagram(POS, LABELS, _routes(EDGES), A, SIZE)
    problems = render.image_checks(img, A, *_geometry(_routes(EDGES[:2])))  # geometry does not know C->A
    assert any("unexpected ink between unconnected nodes A and C" in p for p in problems)


def test_missing_label_is_detected():
    img = render.draw_diagram(POS, ["A", "B", " "], _routes(EDGES), A, SIZE)
    problems = render.image_checks(img, A, *_geometry(_routes(EDGES)))
    assert any("node C: label not visible" in p for p in problems)
