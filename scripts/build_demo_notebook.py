"""Write notebooks/demo_verifier_cpu.ipynb, the CPU demo notebook (no model, no GPU, no download).

The notebook only calls demo/walkthrough.py; its cells live here so the file is generated, not
hand-edited, and tests/test_demo_notebook.py can run them without Jupyter.

  python scripts/build_demo_notebook.py            write the notebook
  python scripts/build_demo_notebook.py --check    exit 1 if the tracked notebook differs
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "demo_verifier_cpu.ipynb"

CELLS = [
    ("markdown", """# TRACE: what the verifier accepts and rejects (CPU, no model)

This notebook takes one root graph of the benchmark and shows, for its reachability question, its critical-edge
question and its repair task, how the exact scorer judges a handful of responses, and what happens to the same
response on the **twin image** (same instruction, a graph that differs by one edge).

- **No model runs here.** Every candidate response is written by `demo/walkthrough.py` to illustrate a case. They are
  not model outputs and nothing in this notebook measures a model.
- The verdicts come from `trace_vr.metrics.score_sample`, the function all evaluations in the repository use. It
  compares a response with the generator's true graph, which exists only for generated images.
- Needs Python with Pillow and PyYAML (`requirements-lock.txt`); run it from a checkout of the repository."""),
    ("code", """import json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "src" / "trace_vr").is_dir():
    if ROOT.parent == ROOT:
        raise SystemExit("run this notebook from inside the repository")
    ROOT = ROOT.parent
for p in (ROOT / "src", ROOT / "demo"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import walkthrough

MANIFEST = Path(os.environ.get("TRACE_DEMO_MANIFEST", ROOT / "manifests" / "smoke.json"))
SPLIT = os.environ.get("TRACE_DEMO_SPLIT", "dev")  # locked test splits are refused
tracked = json.loads(MANIFEST.read_text())
if not all((MANIFEST.parent / tracked["data_dir"] / i["samples_file"]).exists() for i in tracked["splits"].values()):
    # data/ is not tracked: rebuild the smoke set into a scratch folder and compare it with the tracked manifest
    scratch = Path(tempfile.mkdtemp(prefix="trace_demo_"))
    rebuilt = scratch / "smoke.json"
    subprocess.run([sys.executable, "-m", "trace_vr", "generate", "--config", str(ROOT / "configs" / "data_smoke.yaml"),
                    "--out", str(scratch / "data"), "--manifest", str(rebuilt)], check=True, cwd=ROOT,
                   env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    same = json.loads(rebuilt.read_text())["content_digest"] == tracked["content_digest"]
    print("rebuilt the smoke set; content digest equals the tracked manifest:", same)
    MANIFEST = rebuilt
manifest, data_dir, samples = walkthrough.load(str(MANIFEST), SPLIT)
sections = walkthrough.walkthrough(samples, int(os.environ.get("TRACE_DEMO_ROOT", "0")))
print("dataset %s, split %s, content digest %s" % (manifest["name"], SPLIT, manifest["content_digest"][:16]))
print("root graph picked by position only:", sections[0]["base"]["image"])"""),
    ("markdown", """## Reachability

Left: the root image. Right: its twin, where one edge was added or removed so that the true answer flips.
A model that answers from the question text alone, or always says the same thing, cannot get both right."""),
    ("code", """print(walkthrough.render(sections[0]))
walkthrough.side_by_side(data_dir, sections[0])"""),
    ("markdown", """## Critical edge

"Would the connection be lost if this edge were removed?" The twin adds or removes a bypass, so again the answer flips."""),
    ("code", """print(walkthrough.render(sections[1]))
walkthrough.side_by_side(data_dir, sections[1])"""),
    ("markdown", """## Minimal repair

The response is a list of edits. The verifier applies it to the true graph and reports three things separately:
**legal** (allowed additions only, edges that exist, within the edit budget), **goal met**, and **minimal** (no
solution with fewer edits exists; every minimal solution is accepted, not just the stored one). Only a minimal
solution counts as correct. On the twin image the same edit is unnecessary, impossible or no longer enough."""),
    ("code", """print(walkthrough.render(sections[2]))
walkthrough.side_by_side(data_dir, sections[2])"""),
    ("markdown", """## Scoring a real model

Export the prompts, run any model, and score its raw outputs with the same function (README, "Scoring your own model"):

```bash
python scripts/benchmark_io.py export manifests/smoke.json dev --out prompts.jsonl
python scripts/benchmark_io.py score manifests/smoke.json dev predictions.jsonl
```

The measured results of the study's own arms are in `reports/` and the paper, not in this notebook."""),
]


def notebook() -> str:
    cells = []
    for kind, text in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = notebook()
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            raise SystemExit("%s differs from scripts/build_demo_notebook.py; rerun the script" % OUT.name)
        print("ok")
        return
    OUT.write_text(text, encoding="utf-8")
    print("wrote %s (%d cells)" % (OUT, len(CELLS)))


if __name__ == "__main__":
    main()
