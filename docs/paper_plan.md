# Paper plan (owner's goal, 2026-11-11: a contribution useful to the AI literature, published)

Status: plan. No claim below is a result until the run that supports it exists in `runs/`.
Numbers quoted are pilot numbers (one seed, dev split) and only motivate the plan.

## What we can contribute, in order of how safe it is

1. **A verified benchmark and generator for reading directed-graph drawings** (safe: exists).
   - Every label comes from an exact solver, checked by an independent brute-force
     reference; every drawing passes geometric readability checks and a pixel read-back.
   - **Answer-flipping twins**: each question comes with a second image that differs by one
     edge and has the opposite answer, so a text-only or image-blind model scores at most
     0.50, and "both twins right" measures whether the model actually uses the edge.
   - Invariance pairs (relabel, restyle), group split with no isomorphic graph across
     splits, held-out styles, cross-platform deterministic content digest.
   - A one-edit *repair* task (goal + image → add/remove one edge, or "unchanged" /
     "impossible") with a verifier that accepts every optimal answer.
   - Useful even if every model result is negative: closest public work (VGCure/MCDGraph,
     GraphVerse) has no edit-action output, no exact twin controls, no reachability repair.
2. **A controlled transfer study** (dev: three seeds done 2026-11-14; locked test running): does training a 2B VLM only on graph
   repair improve non-edit questions (reachability, edge criticality) compared with
   training on those questions directly (QA-SFT), with equal samples, images and tokens?
   Dev, three seeds pooled (`reports/pilot/seeds_dev_interim.md`): non-edit macro D - B
   -0.131 [-0.153, -0.110]. Per task: reachability D - B -0.014 [-0.037, +0.008] (interval
   contains zero), critical edge -0.248 [-0.287, -0.211]; D gets both critical-edge twins
   right on 66 of 600 pairs, B on 325 (`reports/pilot/pairs_seeds_dev.md`). Dev-split shape,
   to be confirmed or not on the locked test: *learning to edit for reachability goals
   teaches reading paths, not reasoning about which edge a path depends on*.
3. **Where the failure is: perception or reasoning** (dev done; locked test running).
   Oracle-graph text conditions (graph given as text, with or without the image;
   `reports/pilot/oracle.md`) and parse-and-solve (E): dev non-edit macro 0.870
   [0.844, 0.894], +0.072 [+0.042, +0.102] vs B at three seeds. Solver-only sensitivity
   shows how fast exact solving degrades with one or two misread edges
   (`reports/parse_solve/sensitivity_dev.json`).
4. **Negative and engineering results** (`docs/negative_results.md`), incl. that the frozen
   base cannot be scored fairly under a strict JSON contract.

## Minimum experiments before writing results

| # | Experiment | Compute (T4) | State |
|---|---|---|---|
| 1 | B and D, 3 epochs, 200 dev graphs | ~1.1 h per arm | done (`reports/pilot/pilot3.md`; three seeds in row 5) |
| 2 | Oracle text conditions for B and D | ~0.5 h per arm | D done (job9, `reports/pilot/oracle.md`); B done on Kaggle (job11, 100 roots, edges mode) |
| 3 | Arm E (model parses graph, solver answers) | ~0.5 h | done on dev, three seeds (`reports/pilot/e.md`, row 5) |
| 4 | Arm C (MCDGraph-style structural SFT), same budget | ~1.1 h | done on dev, three seeds (`reports/pilot/c.md`, row 5) |
| 5 | Three seeds for B, C, D | ~10 h | done on dev 2026-11-14 (B, C, D, E; `reports/pilot/seeds_dev_interim.md`) |
| 6 | Locked test split + style-OOD (x, y) + size-OOD, once, at the end | ~37.5 h Kaggle session time (estimate, `reports/pilot/final_plan.md`) | queue of 12 kernels fixed; jobF01, jobF02 collected 2026-11-14 (numbers unread), jobF03 running; 2 of 12 collected |
| 7 | Frozen base with a fair prompt (decided on dev) | ~1-2.5 h (estimate, 3 variants x 20 roots at 768 tokens) | chosen on dev 2026-11-14 (jobA1, `reports/pilot/base_prompt.md`); locked-test run jobF13 not pushed |

About 15 T4-hours in total: 4–5 days at the current ~3.5 h/day, at zero cost.
Keep adapters: save each adapter to the repo (about 65 MB each) before the VM can be
recycled, so oracle and OOD evals do not need retraining.

## Paper outline (workshop length, 4–8 pages)

1. Introduction: VLMs read diagrams poorly; does learning to *edit* a diagram teach reading it?
2. Benchmark: generator, verifier, twins, splits, controls (figure: one root image, its
   twin, its repair).
3. Setup: Qwen3.5-2B + LoRA r16, arms A–E with matched samples/images/tokens.
4. Results: main table with root-graph bootstrap CIs over 3 seeds; twin-pair accuracy;
   oracle and parse-and-solve decomposition.
5. Analysis: answer bias on critical edge, family/size/style slices.
   External check on VGCure images (adapted prompts; exploratory subset now, pre-registered subset once at the end).
6. What did not work (from `docs/negative_results.md`).
7. Limitations: synthetic drawings, one model size, single-edit budget.

## Where to publish (all free)

- **arXiv** preprint (cs.CV or cs.LG). First-time submitters usually need an
  *endorsement* from an existing arXiv author in that category: the owner must arrange
  this (a mentor, a professor, or a contact from YC Startup School / Boğaziçi). To check
  closer to submission.
- A **workshop** at a major venue (NeurIPS/ICLR/ICML/CVPR workshops on reasoning,
  multimodal, or negative results), or **TMLR** (rolling, no deadline).
- Code + generator on GitHub, dataset + adapters on Hugging Face (owner's accounts).
- Informal version: a blog post / X thread with the twin figure.

Every upload is an outward action: prepare it fully, then confirm with the owner right
before it happens (RUNBOOK).
