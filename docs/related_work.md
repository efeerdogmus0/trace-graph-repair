# Related work and source check

Access date: 2026-11-09. Collected by a search subagent that opened the pages listed
below. Re-checked directly by the project lead on 2026-11-10: both base-model repositories
exist on the Hub with the pipeline tag, licence and creation dates given here, and their
current revisions are `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` and
`Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`. Everything else
is "reported by subagent" until spot-checked. The search for overlapping work was about
eight queries and is not an exhaustive negative.

## Base model candidates (both exist)

| | `Qwen/Qwen3.5-2B` | `Qwen/Qwen3-VL-2B-Instruct` |
|---|---|---|
| Pipeline tag | image-text-to-text | image-text-to-text |
| License | apache-2.0 | apache-2.0 |
| Parameters | 2,274,069,824 | 2,127,532,032 |
| Weights | 4,548,221,488 bytes, one safetensors file | 4,255,140,312 bytes, one file |
| Created | 2026-02-28 | 2025-10-19 |
| Class | `Qwen3_5ForConditionalGeneration` | `Qwen3VLForConditionalGeneration` |
| transformers | card says install from main; config records 4.57.0.dev0 | same |
| `trust_remote_code` | not in card examples, no `auto_map` | same |

Unsloth notebook `nb/Qwen3_5_(2B)_Vision.ipynb` exists. It loads `unsloth/Qwen3.5-2B`
(mirror of the Qwen repo, apache-2.0) with `load_in_4bit=False`, pins
`transformers==5.2.0`, LoRA r=16, batch 2 x grad-accum 4, 30 steps on a Tesla T4; saved
outputs show 5.146 GB peak reserved and 9.32 minutes. These are the notebook's records on
its own OCR data, not our measurements. Only the first ~40% of the notebook JSON was read.

Decision (2026-11-10, `DECISIONS.md`): Qwen3.5-2B is the base, Qwen3-VL-2B-Instruct the fallback.

## Closest work

| Work | What it does | Difference from TRACE |
|---|---|---|
| VGCure / MCDGraph (ACL 2025, aclanthology 2025.acl-long.1482; arXiv 2412.13540) | Structural SFT on graph images: masked graph infilling (10k), contrastive graph discrimination (5k, two images, same graph or not), graph description (5k). Bases: Qwen2-VL-7B, InternVL2-8B with LoRA | No constraint-driven edit actions; no edit-vs-QA transfer comparison. No 2B checkpoint. Repo and dataset state no license |
| GraphVerse (arXiv 2608.06769, v2 2026-10-09) | 11k-sample benchmark, 9 single-image and 2 paired-image tasks; "editing" means topology-preserving input perturbations. Trains Qwen3-VL-8B/2B | No add/remove-edge action output; reachability is not a task. Repo is README only ("will release soon") |
| MED (arXiv 2506.07227) | 50K+ minimally edited natural-image pairs, fine-tuning with consistency loss | Natural images, no graph actions, no exact verifier |
| CF-VLM (arXiv 2506.17267) | Counterfactual minimal edits as contrastive signal | Natural images, no graph repair |
| GeoWeave (arXiv 2609.12606) | Diagram editing as one SFT stage for geometry | No isolated edit-to-non-edit transfer reported in the abstract |

Other names from the planning notes: arXiv 2603.02865 exists but v2 (2026-11-06) was
retitled "Signals Are Present, Predictions Fail..." and now says node and edge
information is decodable even where VQA fails (v1 said edges were not linearly
separable). R-Qwen (2608.01534) and TRON (2606.01599) exist. RECAP and LoRe are NeurIPS
2026 posters unrelated to graphs. "LoopLook" was not found; nearest name is LoopVL
(arXiv 2609.38426).

Correct term for the MCDGraph task: "Contrastive Graph Discrimination", not "graph comparison".

## Free compute

- Colab FAQ: free GPU access "is heavily restricted", limits are not published and
  fluctuate, notebooks run at most 12 hours; T4 is not named.
- Kaggle GPU docs: could not be opened (connection resets). Nothing confirmed.

## Not readable in this pass

Kaggle docs; the ACL PDF (arXiv HTML used instead); second half of the Unsloth notebook;
last part of the GraphVerse HTML; a Stanford CS224R report on GRPO with graph images.

## Overlap search 2 (2026-11-15, arXiv API only)

`scripts/overlap_search.py --out docs/sources/overlap_2026-11-15`: 14 fixed arXiv API queries
(listed in the script and in `hits.json`), raw Atom XML kept, 100 unique hits. Titles of all 100
were read; abstracts of the nine plausible ones. No web search engine. Claims below are from
arXiv abstracts only (papers not read).

| Work | From the abstract | Bearing on TRACE |
|---|---|---|
| VGCompiler, arXiv 2609.22327 (2026-09-16) | Qwen3-VL-8B + RL recovers an explicit graph representation, then compiles the query into an executable graph operation; evaluated on GVLQA, VisionGraph, VGCure | Closest precedent for arm E (parse-and-solve). Not an edit-training-transfer study. Cited |
| VisionGraph, arXiv 2405.04950 (2024) | Benchmark of 8 graph-theory tasks on images; description-program-reasoning chain; perception of structure limits all models, also after SFT | Prior art for describe-then-solve and for the perception bottleneck. Cited |
| Knossos/Ariadne, arXiv 2610.04721 (2026-10-03) | 19,200 synthetic diagrams with exact topology; training small VLMs on complete extraction; node inventory then source-conditioned edges | Related to arm C description task and E's parser. Cited |
| MapReason-OSM, arXiv 2606.22597 (2026-06) | Rendered OSM maps with hidden street graph and exact oracles; structured decisions scored for validity/optimality | Shares the verifiable-structured-output design; no training comparison. Cited |
| Edit, Don't Imitate, arXiv 2610.30231 (2026-10-21) | "Minimal edit" = teacher scoring of token-level corrections in on-policy distillation for visual math | Same words, different meaning (no graph edits); not cited, note to avoid confusion |
| VNA benchmark 2405.06634; Graph-to-Vision 2503.21435 | Zero-shot benchmarks of drawn-graph analysis; multi-graph benchmark with fine-tuning | Background; not needed for the claims |
| Decodable Is Not Grounded, 2606.31257 | Blank-image control separates grounded perception from priors | Related to our twin bound (image-blind 0.50); possible citation later |

Outcome: no hit trains a VLM to output a goal-satisfying graph edit and tests transfer to non-edit
questions against QA and structural fine-tuning; the novelty statement in `paper/sections/related.tex`
stands, but arm E is now framed as a known method (reference arm), not a contribution.
