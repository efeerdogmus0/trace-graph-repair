# External evaluation (M7): pre-selected VGCure subset

Fixed 2026-11-13, before any model output on VGCure was produced or looked at.

## Source and access

- Dataset: Hugging Face `AAAndyZ/VGCure`, revision `06d8cf2c10ffc6fb6d4c518eda1275b6eee2f56c`
  (last modified 2025-05-15), not gated. Paper: Zhu et al., ACL 2025 (`aclanthology.org/2025.acl-long.1482`).
  Code: `github.com/AAAndy-Zhu/VGCure` (scoring script `scripts/metrics_for_qwenvl_minigptv2_monkey_internvl.py`).
- Licence: none stated on the dataset card or in the GitHub repository (API `license: null`, checked 2026-11-13).
  Consequence: we evaluate on it and cite it; we do not redistribute its files or question text. Only ids,
  file hashes and the subset digest are committed (`reports/external/vgcure_subset.json`); the raw files are
  downloaded into `external/` (gitignored) or on the Kaggle side. Nothing from VGCure enters any training pool.
- Format observed: JSONL per source and directed/undirected variant, one image per graph (`*_akg_images.zip`,
  ~0.1-0.5 GB each, 6.2 GB for all 20 zips; directed zips ~3.1 GB). Records hold question/claim strings and
  answers, not the edge list. Images draw entity nodes `E1..` and relation labels `R1..` on the edges (style OOD
  for our models). Directed test sets: ba/er/pubchemqc/sbm/sfn/star 400 graphs, dblp/dbpedia 200,
  social_network 300, openflights 100; 4-19 nodes.

## Pre-registered subset (adapted protocol)

Selection rule, fixed in `src/trace_vr/vgcure.py` and tested in `tests/test_vgcure.py`:
all 10 directed sources; per source the first 50 graphs in numeric image-id order that carry all three items:

| item | from | our task | answer balance (500 graphs) |
|---|---|---|---|
| edge true | `neighbor_query.fc_sup` | `edge_exists` | 500 true |
| edge false | `neighbor_query.fc_ref` | `edge_exists` | 500 false (485 same-direction non-neighbours, 15 reversed existing edges) |
| path | `connective_path_query.qa` (Yes/No part only) | `reachability` | 409 yes / 91 no (ba, pubchemqc, sbm, sfn, star all yes; openflights all no) |

1,500 samples, subset sha256 `ac4796f87a6d5311a0cb7bed1bdd2875c9b5500cc667fcdf8a7b7d7264563888`.
Each neighbour claim is checked against the record's own neighbour list (positive listed, same-direction negative
not listed). Instructions are our own templates (`tasks.render_instruction`), answers strict JSON, so this is an
**adapted** protocol on VGCure images and must be reported as "pre-selected VGCure subset, adapted prompts", not as
VGCure's official benchmark numbers.

## Scoring (fixed now)

- `edge_exists`: accuracy over the 1,000 balanced items (per source and pooled).
- `reachability`: balanced accuracy (mean of yes-accuracy and no-accuracy, pooled over sources) because the
  file's classes are 82/18 and five sources are single-class; raw accuracy reported as secondary.
- External macro = mean of the two. Arm comparisons: paired bootstrap over graphs (`root_id` = one VGCure graph),
  same as dev. E (parse-and-solve) is scored with the same items; its parser was trained on our drawing style, so a
  transcription failure on relation-labelled images is an expected style-OOD effect, reported as such.
- Arms: the final adapters of B, C, D and E at the seed used for the locked test. Evaluated once, after the locked
  test, so no decision on dev depends on VGCure.

## Evaluation path (built 2026-11-13, not run)

- `scripts/vgcure_prepare.py <work_dir>` (on Kaggle): downloads the 20 JSONL files and 10 directed zips at the pinned
  revision, checks each zip against its LFS sha256 (`vgcure.ZIP_SHA256`), rebuilds the subset and refuses unless the
  digest and source-file hashes equal `reports/external/vgcure_subset.json`, extracts only the 500 selected images.
- **Image preprocessing (fixed now, before any output):** every VGCure image is letterboxed to the dev/test canvas,
  512x512 (longest side scaled to 512, aspect kept, LANCZOS, centred on white). Our adapters only ever saw 512x512
  inputs; VGCure's native resolution would change the number of visual tokens and the T4 cost. Downscaling can make
  small relation labels hard to read; that is part of the style-OOD effect and is reported with the original size range
  (`VGCURE-PREPARED` line).
- `scripts/eval_vgcure.py <work_dir> <run_dir> <adapter|none> [--parse-solve]`: direct answers (max_new 96, as on dev) or
  arm E (one transcription per image, max_new 512, exact solver). Writes `metrics_vgcure.json`, prints EVAL-SUMMARY and
  FLAGS-SUMMARY (bit strings in subset order; `vgcure.flags`). Scoring = `vgcure.evaluate` (the rules above).
- `scripts/kaggle/make_vgcure_kernel.py <job> --after-final-evaluation <source_job>:<run> ... [<job>:pilot3_E:parse=<parser>] [base]`:
  one kernel for all arms; adapters attached via `kernel_sources` and sha256-checked against `runs/kaggle_artifacts.json`;
  refuses without the flag. Cost estimate: ~1,500 short generations per direct arm (dev: ~1,250 s for ~1,900 samples) +
  500 long ones for E, plus the 3.1 GB download on Kaggle's side.

## Not done yet

- Running the kernel (after the locked test).
- VGCure's official-format prompts (their question text, their scorer) as a secondary row; our adapters were never
  trained on free-form answers, so this row mostly measures format compliance.
