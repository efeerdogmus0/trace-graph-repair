# TRACE

Code and data for *Does Learning to Repair a Drawn Graph Teach a Small VLM to Read It? A verified
benchmark with answer-flipping twins* (Yiğit Efe Erdoğmuş, Yiğit Çerçi; preprint in preparation;
the paper source is in `paper/`). Code MIT (`LICENSE`); data, images, reports and paper text
CC BY 4.0 (`LICENSE-DATA.md`).

**Research question.** If a small vision-language model is fine-tuned to *repair* a drawn
directed graph with a minimal edit (add or remove one edge so that reachability goals
hold), does it get better at questions that involve *no editing* (is B reachable from A?
is this edge critical for that connection?) on graphs it has never seen, compared with
ordinary QA fine-tuning and structural fine-tuning of the same base model?

**Status (2026-11-23): locked test and external check done; pilot scale.** The data core is built and validated;
the model pipeline (Qwen3.5-2B + LoRA) runs on free T4 GPUs (Colab, then Kaggle). On the locked
test (200 root graphs per split, three splits, pooled over three seeds), the pre-registered primary
statistic (mean of reachability and critical-edge accuracy over the three splits) is B 0.762,
D 0.657, E 0.827, C 0.538. Repair training (D) did **not** beat QA training (B): D - B = -0.105
[-0.117, -0.093], reading `negative` (H1 not supported; reachability about equal, critical-edge
much lower). Parse-the-graph-then-solve (E) is the strongest arm: E - B = +0.065 [+0.048, +0.082],
reading `positive_clear`. Structural training (C) is lowest: C - B = -0.224. Source:
`reports/final/final_primary.md` (`scripts/final_primary.py`, rules fixed before the test was
opened). On the pre-registered external VGCure subset (other
authors' drawings, edge existence and reachability only, adapted prompts, no decision rule) D is ahead
of B, +0.056 [+0.038, +0.075] at every seed, and E falls behind B (its parser fails on the foreign
style): `reports/external/vgcure_final_seeds.md`. Per-adapter cards: `docs/model_cards/`. See also
`RELEASE_CHECKLIST.md`, `docs/protocol.md` and the benchmark datasheet `docs/datasheet.md`.

## What exists

| Piece | Where | Checked how |
|---|---|---|
| Graph families, reachability, isomorphism | `src/trace_vr/graphs.py`, `families.py`, `iso.py` | against networkx and a second independent implementation |
| Exact repair solver and task proposals | `src/trace_vr/semantics.py` | against a brute-force reference; hand-checked cases |
| Instructions, strict JSON parser, scorer | `src/trace_vr/tasks.py` | hand-checked cases; scorer agrees with the full solution set on generated tasks |
| Layout, edge routing, readability rules | `src/trace_vr/layout.py`, `geometry.py` | geometric unit tests; contact sheets inspected by eye |
| Renderer and pixel read-back checks | `src/trace_vr/render.py` | tests that damage the picture on purpose |
| Dataset builder, manifest, model view | `src/trace_vr/dataset.py` | rebuild gives identical digests; private fields cannot reach the model view |
| Validator and text-only shortcut probes | `src/trace_vr/validate.py`, `shortcuts.py` | tests that corrupt a dataset on purpose |
| Metrics from raw responses | `src/trace_vr/metrics.py` | hand-checked prediction file; oracle and constant responders |

## Quick start (CPU only)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pillow pyyaml pytest networkx
export PYTHONPATH=src

.venv/bin/python -m trace_vr doctor
.venv/bin/python -m trace_vr generate --config configs/data_smoke.yaml
.venv/bin/python -m trace_vr validate-data --manifest manifests/smoke.json --report reports/smoke/validation.json
.venv/bin/python -m trace_vr render-gallery --manifest manifests/smoke.json
.venv/bin/python -m pytest
```

These are the commands that were actually run for the smoke set. `generate` writes
`data/<name>/` (not tracked in git) and `manifests/<name>.json` (tracked). The gallery
ends up in `reports/<name>/gallery/`; open `index.html` there to browse every image with
its questions and answers.

### Scoring your own model

The benchmark does not depend on this repository's model code. Export the prompts of a split,
run any model on them, and score the raw outputs:

```bash
.venv/bin/python scripts/benchmark_io.py export manifests/smoke.json dev --out prompts.jsonl
# prompts.jsonl: {"sample_id", "task", "image", "prompt"} per line; write your model's raw
# output for each as {"sample_id": ..., "text": ...} into predictions.jsonl
.venv/bin/python scripts/benchmark_io.py score manifests/smoke.json dev predictions.jsonl --flags mine.json
```

The scorer prints strict and lenient accuracy per task, the non-edit macro with a root-graph
bootstrap interval, and how many answer-flipping twin pairs had both members right. Two
`--flags` files give a paired comparison with `scripts/bootstrap_ci.py <manifest> a.json b.json`.
Locked test splits are refused; develop on `dev`. Checked by `tests/test_benchmark_io.py`.

### Demo notebook (CPU, no model)

`notebooks/demo_verifier_cpu.ipynb` takes one root graph and shows how the exact scorer judges a few
hand-written responses to its reachability, critical-edge and repair tasks (legal / goal met / minimal),
and how the same response fares on the one-edge-different twin image. No model runs in it and it measures
nothing; if `data/smoke` is missing it rebuilds the smoke set in a scratch folder (about 20 s) and compares
the content digest with the tracked manifest. The same text without Jupyter:
`PYTHONPATH=src .venv/bin/python demo/walkthrough.py manifests/smoke.json dev`. The notebook is generated by
`scripts/build_demo_notebook.py` (`--check`); `tests/test_demo_notebook.py` runs its cells.

### Dev gallery (recorded model outputs)

`reports/pilot/gallery_dev.md` shows what the frozen base (A) and the trained arms (B, C, D, E; seed 3407) actually
answered on seven dev images, next to the verifier's verdict: a reachability success and failure, a critical-edge
twin pair, and three repair cases (one edit needed, impossible goal, a failure). The responses are read from the
recorded predictions files (sha256 checked against the flags the reported numbers come from); the selection rule
is fixed in `scripts/dev_gallery.py` and uses a hash order, so the page is not a set of picked successes. It
illustrates behaviour and measures nothing. Style-change pairs are missing: no dev response was recorded for them.
On the repair cards every edit named by the generator's stored answer or by a response is marked on a copy of the image
(orange route with a cross = remove, blue dashed arrow = add; the colour is the operation, not the verdict; an added
arrow is a straight marker, not a laid-out edge). An edit that names an edge or node that is not in the image is
listed with that reason and not drawn.

### Dev figures

`python scripts/dev_figures.py` redraws `reports/pilot/figures/` from the recorded development metrics: `arms_dev`
(accuracy per task and arm, pooled over three seeds, with the 95% root-graph interval and the single seeds) and
`cost_dev` (each run's non-edit macro accuracy against its measured T4 time) and `errors_dev` (per run and task: right,
parsed but wrong, format failure; for arm E the critical-edge questions its solver could not answer). SVG and PNG, plus
PDF copies of the first and third for the paper; `figures_dev.json` lists every plotted value and the hash of its source, and `--check` says
whether it is stale. Accuracy axes run from 0 to 1. Development split only; no locked-test value is drawn.

### Locked-test reports

Generated once the locked test was opened (2026-11-22), each by its own script from the collected final kernels;
none is edited by hand. `scripts/final_reports.py --out-dir reports/final` writes the tables:

- `reports/final/final_primary.md`: the pre-registered primary statistic and its readings (the numbers in Status).
- `reports/final/final_table.md`: every run and split, pooled over seeds with root-graph intervals.
- `reports/final/final_metrics.md`, `final_slices.md`, `final_pairs.md`, `final_invariance.md`, `final_density.md`,
  `final_floor.md`: secondary tables (per-task metrics, slices, twin pairs, style invariance, graph density, frozen-base floor).
- `reports/final/figures/README.md`: non-edit accuracy on the six locked splits (`scripts/final_figures.py`).
- `reports/final/style_cards.md`: one reachability and one critical-edge question per split on a root image and its
  restyled copy, with every final run's recorded answer (`scripts/final_style_cards.py`; an illustration, not a measurement).
- `reports/final/kernel_env.json`: the GPU environment the final kernels recorded.

Training and evaluation run on a GPU (`scripts/train_arm.py`, `scripts/eval_arm.py`; Colab launcher `scripts/colab/job.sh`); see `RUNBOOK.md`.

## Dataset design in brief

- **Graphs.** Directed, 5-10 nodes in training (11-16 in the structure-OOD split), single
  uppercase letter labels, no self-loops, no pair of opposite edges.
- **One image, many questions.** Each root graph gets a repair task, a reachability
  question, a critical-edge question and a few secondary questions.
- **Twins.** For each main question there is a second image that differs from the first by
  exactly one edge, keeps the instruction text identical and changes the correct answer.
  Any predictor that ignores the image gets at most one of the two right; the validator
  reports this "text-only ceiling"; it is 0.50 on the smoke and pilot sets.
- **Repair answers.** `edited` (one add or one remove), `unchanged`, or `impossible`.
  All minimal solutions are stored; a model answer is applied to the true graph and
  judged for legality, goal satisfaction and minimality. Text match is never used.
- **Leakage control.** The split unit is the root graph together with everything derived
  from it. In addition no two graphs from different groups are isomorphic, checked by an
  invariant fingerprint followed by exact matching.
- **Splits.** `train`, `dev`, `test_id` (same families and styles), `test_style_ood`
  (unseen drawing styles) and `test_struct_ood` (unseen families, larger graphs).
- **What a model sees.** Only the image and the instruction text
  (`dataset.to_model_example`). Ground truth lives under `private_truth`.

## Known limitations

- Drawings are synthetic and tidy. Nothing here shows performance on real diagrams.
- The layout algorithm is not a held-out factor of the style-OOD split (see `styles.py`).
- `edge_exists` and `shortest_path_length` are secondary tasks without twins.
- Images are byte-identical across rebuilds on the same machine. Across platforms the
  sample content is designed to be identical (own RNG, no platform-dependent maths in
  decisions), and the pilot content digest was identical on macOS and Colab Linux; PNG
  bytes differ between machines (font rasterisation).
- On root images the critical-edge answer is "true" about twice as often as "false"
  (the overall balance is exactly 50/50 through the twins).

## Compute used

No cash was spent. All GPU work ran on free or already-owned allocations (Colab and Kaggle T4
sessions). `reports/budget.md` sums the logged training and evaluation seconds by purpose and by
platform and lists the failed runs and the runs whose time was lost; it is a lower bound, because
setup time on the GPU machine is not logged. Regenerate it with `python scripts/budget_report.py`.

## Repository layout

```
configs/      dataset configs (smoke, pilot draft)
docs/         protocol (draft), environment, related work
manifests/    dataset manifests: hashes, statistics, rejection counts
reports/      validation reports and galleries
scripts/      ad-hoc development checks
src/trace_vr/ the package
tests/        pytest suite
assets/fonts/ DejaVu fonts (see LICENSE_DEJAVU)
```

`TRACE_ASTRA_MASTER_PROMPT.md` is the original project brief (Turkish) and is kept
unchanged. `DECISIONS.md` records where and why the implementation deviates from it.

## Third-party material

DejaVu fonts are redistributed under their own licence (`assets/fonts/LICENSE_DEJAVU`); they are the
only third-party files in the repository. `THIRD_PARTY.md` (generated by `scripts/third_party.py`,
`--check`) lists every component used, its pinned version, the licence it declares and what that
statement rests on: the base model (Apache-2.0 per its model card and licence file), the VGCure dataset
and code (no licence stated, so nothing of it is copied here) and the Python packages.
## Licence

Code (`src/`, `scripts/`, `tests/`, `notebooks/`, `demo/`): MIT, see `LICENSE`. Generated benchmark data,
images, manifests, reports, figures, paper text and released adapter weights: CC BY 4.0, see
`LICENSE-DATA.md`. The base model, the DejaVu fonts and VGCure keep their own terms (above).
