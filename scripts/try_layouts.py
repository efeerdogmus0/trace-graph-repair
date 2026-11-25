"""Ad-hoc: layout acceptance rates, timing and a contact sheet per style."""
import sys, time, collections
from PIL import Image
from trace_vr.rng import Rng
from trace_vr import families, layout, render
from trace_vr.styles import STYLES
from trace_vr.tasks import LABEL_ALPHABET

out = sys.argv[1]
SIZE = 512
combos = [(f, (5, 10)) for f in ("layered_dag", "tree_plus", "sparse_random")] + [(f, (11, 16)) for f in ("ring_chords", "two_cluster")]
for sid, style in STYLES.items():
    tiles = []
    for fam, (lo, hi) in combos:
        ok = 0; fails = collections.Counter(); t0 = time.time(); cross = []; names = collections.Counter(); problems = 0
        N = 20
        for i in range(N):
            rng = Rng("lay", sid, fam, i)
            n = rng.randint(lo, hi)
            g = families.generate(fam, n, rng.child("g"))
            lay, f = layout.find_layout(g, style, rng.child("layout"), SIZE)
            fails.update(f)
            if lay is None:
                continue
            ok += 1; cross.append(lay.crossings); names[lay.layout_name] += 1
            labels = rng.child("labels").sample(LABEL_ALPHABET, n)
            img = render.draw_diagram(lay.pos, labels, lay.routes, style, SIZE)
            nodes = {labels[k]: list(lay.pos[k]) for k in range(n)}
            edges = [{"src": labels[u], "dst": labels[v], "points": lay.routes[(u, v)]} for (u, v) in sorted(lay.routes)]
            pr = render.image_checks(img, style, nodes, edges)
            if pr:
                problems += 1
                if problems <= 2: print("   pixel problems:", pr[:3])
            if len([t for t in tiles if t[0] == fam]) < 2:
                tiles.append((fam, img))
        dt = (time.time() - t0) / N
        print("style %s %-13s ok %2d/%d  %.2fs/graph  crossings avg %.1f max %s  layouts %s  pixel-problem imgs %d  fails %s" % (
            sid, fam, ok, N, dt, (sum(cross) / len(cross)) if cross else -1, max(cross) if cross else "-", dict(names), problems, dict(fails)))
    sheet = Image.new("RGB", (SIZE * 5, SIZE * 2), (128, 128, 128))
    col = collections.Counter()
    for fam, img in tiles:
        c = [f for f, _ in combos].index(fam)
        sheet.paste(img, (c * SIZE, col[fam] * SIZE)); col[fam] += 1
    sheet = sheet.resize((SIZE * 5 // 2, SIZE))
    sheet.save("%s/layout_style_%s.png" % (out, sid))
