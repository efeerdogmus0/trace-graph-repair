# Datasheet: TRACE drawn-graph benchmark (draft, not released)

Structure follows "Datasheets for Datasets" (Gebru et al.). Statistics are generated, not
typed: `docs/datasheet_pilot_stats.md` (rebuild with
`python scripts/datasheet.py manifests/pilot.json reports/pilot/validation.json docs/datasheet_pilot_stats.md`;
`tests/test_datasheet.py` fails if the committed file is out of date).

Status: the `pilot` build (train 512 / dev 200 root graphs) exists and validates. The locked
test build `pilot_test` (`configs/data_pilot_test.yaml`: test_id 200, test_style_ood 200,
test_struct_ood 200, test_combined_ood 100 root graphs; 7,006 samples, 3,078 images) was built
2026-11-12 with no graph isomorphic to any pilot train/dev graph and validates with 0 errors
(`reports/pilot_test/validation.json`). Its labels are locked: no statistics beyond counts are
reported until the final evaluation. A second locked build `pilot_diag`
(`configs/data_pilot_diag.yaml`: test_family_ood and test_size_ood, 100 root graphs each; 1,978
samples, 869 images) separates the family and size shifts that test_struct_ood combines; it is
isomorphism-disjoint from pilot and pilot_test and validates with 0 errors
(`reports/pilot_diag/validation.json`). Nothing has been published.

## Motivation

- Purpose: measure whether a vision-language model reads the *structure* of a drawn directed
  graph (which edge goes where, which path depends on which edge), and whether training it
  to *repair* a drawing with a minimal edit transfers to non-edit questions.
- Gap it fills: closest public sets (VGCure/MCDGraph, GraphVerse; see `docs/related_work.md`)
  have no edit-action output, no answer-flipping twin images and no reachability repair.
- Created by: the TRACE project (Yiğit Efe Erdoğmuş); no external funding; zero cash spent.

## Composition

- Instances: PNG drawings (512 x 512) of directed graphs with 5-10 nodes, each with several
  questions. One sample = one image + one instruction + private ground truth.
- Tasks: `minimal_graph_repair` (add/remove at most one edge so that reachability goals hold;
  answers `edited`, `unchanged`, `impossible`), `reachability`, `critical_edge` (main);
  `edge_exists`, `shortest_path_length`, `graph_extraction` (secondary).
- Twins: every main-task instruction appears with two images that differ by exactly one edge
  and have opposite answers. A predictor that ignores the image is capped at 0.50 on main
  tasks (validator "text-only ceiling", 0.50 on every main task and split of the pilot).
- Invariance pairs: about a quarter of root graphs also get a relabelled and a restyled copy
  with the same answer.
- Labels: computed by an exact solver and checked against an independent brute-force
  reference; the repair scorer accepts every optimal answer (all minimal solutions are stored).
- Sampling: synthetic. Three families (`layered_dag`, `sparse_random`, `tree_plus`), four
  layouts, three drawing styles. Split unit is the root graph and everything derived from it;
  no two graphs in different groups are isomorphic.
- Not a sample of real diagrams. No personal data, no human subjects, no scraped content.

## Known skews and limits (read before using the numbers)

- Secondary tasks have no twins. Their text-only ceiling is high (dev `edge_exists` 0.895)
  because most instructions occur once, so a per-instruction lookup "fits" them; it is not
  an answer leak, but these tasks cannot separate image use from guessing.
- On root images the critical-edge answer is "true" more often (dev 118 / 82); overall
  balance is near 50/50 only through the twins. Report twin-pair accuracy alongside accuracy.
- Repair answers on root images are not uniform (dev: add 43, impossible 52, remove 51,
  unchanged 54); the "direct" share (goal names an addable edge) is small (dev 23 of 200).
- Small topologies run out: most rejections are duplicate topologies, concentrated at 5 nodes,
  which shifts the node-count distribution upward (a rejected root is redrawn with a new node
  count). Measured capacity: about 100 five-node layered_dag classes, 255 tree_plus, 422
  sparse_random (`docs/topology_capacity.md`). A model can see a large share of all five-node
  topologies in training; the main build starts at 7 nodes (protocol item 7).
- Layout is not a held-out factor of the style-OOD split.
- Drawings are clean and synthetic; results do not show performance on real diagrams.

## Collection / generation process

- Fully procedural: `src/trace_vr/` with a project RNG (no platform-dependent maths in
  decisions). Config: `configs/data_pilot.yaml`; seed in the manifest.
- Every image passes geometric readability rules (node spacing, edges clear of nodes,
  recorded crossings) and a pixel read-back against the intended drawing.
- Reproducibility: the content digest (graphs, labels, instructions) was identical on
  macOS/Python 3.9, macOS/Python 3.12 (smoke) and Colab Linux/Python 3.13 (pilot); PNG bytes
  differ between machines (font rasterisation), so `images_digest` is machine-specific.

## Uses

- Intended: controlled evaluation and fine-tuning studies of diagram reading in VLMs;
  perception-vs-reasoning diagnostics (oracle edge list as text, parse-and-solve).
- Not intended: claims about real-world diagram understanding, safety-critical use, or
  ranking general model ability from this set alone.

## Distribution and maintenance (planned, not done)

- Release form: generator code + configs + manifests (images are rebuilt; digests verify the
  content). Code under MIT (`LICENSE`), generated data and images under CC BY 4.0 (`LICENSE-DATA.md`); DejaVu fonts keep their own licence.
- Release requires the owner's confirmation right before upload (RUNBOOK).
