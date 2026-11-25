# What did not work (running log)

Every entry must point to evidence in this repository. Strength labels:

- **solid**: enough runs to support the statement as written;
- **preliminary**: one run or a small sample; do not state it as a finding yet;
- **engineering**: a practical lesson about tooling, not a research result.

Ideas that were discussed but never run are listed separately at the end and must never be
described as "tried".

## Research results

### 1. Repair-only training did not beat QA training on non-edit questions — solid on the dev split and on the locked test (three seeds, pilot scale)
- Pilot, one seed, one epoch, 512 training graphs, 60 dev graphs: non-edit macro accuracy
  D 0.650 vs B 0.767 (`reports/pilot/first_pilot.md`, `runs/pilot_D/summary.json`,
  `runs/pilot_B/summary.json`).
- Open alternative explanations: both arms under-trained (D repair accuracy only 0.467);
  60 graphs give wide intervals; one seed. The 3-epoch rerun on 200 dev graphs (pilot3)
  is meant to address the first two.
- pilot3, D only so far (3 epochs, 200 dev graphs): macro 0.671, 95% interval 0.648–0.694
  over root graphs; repair accuracy still 0.545 (`reports/pilot/pilot3.md`,
  `runs/pilot3_D/`).
- pilot3 B (3 epochs, same 512 graphs, same 200 dev graphs, same images digest): macro
  0.8075 (0.784–0.833). Paired difference D - B over root graphs: -0.136, 95% interval
  -0.163 to -0.109, no bootstrap replicate above zero
  (`runs/pilot3_B/bootstrap_D_vs_B_dev.json`). One seed; B is trained directly on the two
  evaluated question types, so this is the in-distribution QA baseline the hypothesis must beat.
- Three seeds (2027, 3407, 9173; all on Kaggle T4, 200 dev roots, 2026-11-13): pooled non-edit macro D 0.667
  [0.647, 0.686] vs B 0.798 [0.776, 0.822]; D - B -0.131 [-0.153, -0.110], no replicate above zero; seed-matched
  gaps -0.136 / -0.134 / -0.124 (`reports/pilot/seeds_dev_interim.json`, `reports/pilot/seeds_dev_interim.md`). The interval is over
  root graphs, not seeds. Scope of "solid": this model, this recipe (512 training roots, 3 epochs), dev split.
  The locked test and OOD splits were not used for this dev statement.
- Locked test (2026-11-22, opened once after all rules were fixed; `reports/final/final_primary.md`, `scripts/final_primary.py`):
  pre-registered six-cell primary statistic pooled over seeds 2027/3407/9173, 200 root graphs per split: D 0.657 [0.646, 0.667]
  vs B 0.762 [0.750, 0.773]; D - B -0.105 [-0.117, -0.093], seed-matched -0.105 / -0.115 / -0.096, same sign on test_id,
  test_style_ood and test_struct_ood; reading `negative`. H1 is not supported at this scale.

### 2. Transfer appeared for reachability but not for edge criticality — solid on the dev split and on the locked test (three seeds, pilot scale)
- D never saw a reachability label yet reached 0.825 (B 0.850), both twins right on 65%
  of pairs; frozen base 0.567 lenient on 30 graphs.
- On critical-edge questions D was at chance (0.475) and got both twins right on 1.7% of
  pairs, i.e. it answered the same way regardless of the one-edge change. B: 0.683 / 36.7%.
- pilot3 D (3 epochs, 200 dev graphs) repeats the shape: reachability 0.840, critical edge
  0.503 with "true" answered 75% of the time and the same answer for both twins in most
  pairs (both right 19 of 200) (`runs/pilot3_D/breakdown_dev.json`). Still one seed.
- pilot3 B: reachability 0.845 (D 0.840, paired difference -0.005, interval -0.035 to
  +0.025), critical edge 0.770 (D 0.503, difference -0.268, -0.313 to -0.223). So with three
  epochs repair-only training matches direct QA training on reachability and the whole gap
  is in edge criticality. One seed.
- Three seeds, pooled (`reports/pilot/seeds_dev_interim.json`): reachability D 0.814 vs B 0.828, D - B -0.014 [-0.037, +0.008]
  (interval contains zero; 8.9% of replicates above zero); critical edge D 0.519 vs B 0.767, D - B -0.248
  [-0.287, -0.211]. D's critical-edge accuracy per seed: 0.5175 / 0.5125 / 0.5275.
- Twin pairs and answer bias at all three seeds (2026-11-14, `scripts/seed_pairs.py`, `reports/pilot/pairs_seeds_dev.md`;
  descriptive counts, no interval; seed 3407 here is the Kaggle retrain, the "19 of 200" above is the Colab run):
  critical-edge pairs both right D 22 / 25 / 19 of 200 (seeds 3407 / 2027 / 9173; summed 66 of 600 = 0.110), B 106 / 113 /
  106 (325 of 600 = 0.542); D says "true" to 0.7525 / 0.7325 / 0.8825 of critical-edge questions (summed 0.789; true is right
  for 0.500). Reachability pairs both right D 377 of 600 (0.628), B 394 of 600 (0.657). C: critical-edge pairs both right
  8 of 600, says "true" 0.198 (0.075 at seed 2027). E: 438 of 600 critical-edge pairs both right (0.730).
- Locked test (`reports/final/final_primary.json`, pooled over three seeds and three splits): reachability D - B -0.005
  [-0.020, +0.011] (interval contains zero), critical edge D - B -0.206 [-0.225, -0.186]. Per split the critical-edge gap is
  -0.214 (test_id), -0.226 (test_style_ood), -0.178 (test_struct_ood); the reachability gap's interval contains zero on all three.
- It survived the locked test (more data is still untested); it is the most interesting shape of the result:
  learning to edit graphs to satisfy reachability goals transfers to reading reachability
  but not to reasoning about which edge a path depends on.

### 4. Giving D the true graph as text does not fix edge criticality — preliminary
- job9, 2026-11-11: D retrained (same recipe), evaluated on 100 dev roots with the exact edge
  list as text instead of the image: macro 0.6925 vs 0.6825 with the image, paired difference
  +0.010 [-0.0325, +0.050] (corrected 2026-11-12; the earlier [-0.028, +0.053] was a transcription error); critical edge 0.55 vs 0.505 (`runs/pilot3_oracle/`,
  `reports/pilot/oracle/bootstrap_D_edges_vs_D_image_dev100.json`).
- So D's gap to B is not a perception gap. Caveats: one seed, retrained adapter, text-only input
  was never seen in training. B's oracle eval (same job) decides whether B also stays put.

### 5. Revision 1 (D2: repair + edit-outcome auxiliary) did not improve transfer — preliminary
- job10, 2026-11-12, Kaggle T4, one seed, 200 dev roots: non-edit macro 0.6325 vs D 0.671
  (paired −0.039 [−0.063, −0.016]) and B 0.8075 (paired −0.175 [−0.200, −0.149]).
  Critical edge 0.495: the model answers "false" to 99% of critical-edge questions
  (`reports/pilot/d2.md`, `runs/pilot3_D2/`).
- The closer auxiliary signal (outcome of a single edit) did not teach edge dependence; it
  moved the answer prior. Caveats: one seed; D and B were trained on Colab (different torch),
  D2 on Kaggle; outcome labels were imbalanced (196 true / 316 false) and not rebalanced.

### 6. Structural SFT (arm C) is below repair and QA training (solid on the dev split, three seeds); the repair gap is reasoning, not reading (preliminary, one seed)
- job12, 2026-11-12, Kaggle T4, one seed, 200 dev roots: C non-edit macro 0.5375 [0.5125, 0.564],
  C - B_kaggle -0.259 [-0.294, -0.224]; critical edge 0.375 (answers "false" 73.5%)
  (`reports/pilot/c.md`, `reports/pilot/results_dev.json`).
- Perception split (pre-registered 20:40Z, job13 E extraction): on the 135 roots whose drawings E
  transcribes exactly, D - B is still -0.148 [-0.190, -0.105]; D2 and C also read reasoning_gap
  (`reports/pilot/perception_split_dev.md`). Readability is judged by E's parser (proxy).
- Three seeds, pooled (`reports/pilot/seeds_dev_interim.json`): C non-edit macro 0.549 [0.530, 0.569], C - B -0.249 [-0.277, -0.222],
  seed-matched -0.241 / -0.259 / -0.248; reachability -0.139 [-0.172, -0.106], critical edge -0.359 [-0.404, -0.317].
- Caveats: dev split, pilot scale. The perception split was run for seed 3407 only.

### 3. The untrained base model cannot be scored fairly with a strict JSON contract — solid for this model and prompt
- 64 of 80 (cell output, 2026-11-10) and 99 of 180 (pilot A) responses ran out of tokens
  while reasoning in prose, despite "answer with a single JSON object and nothing else"
  (`runs/a0_smoke_t4/metrics.json`, `runs/pilot_A/summary.json`).
- Strict 0.10 vs lenient 0.40 macro on the same responses. Any comparison with a frozen
  base needs a protocol decision (token limit or no-reasoning prompt), made on dev only.

## Engineering lessons

### 4. Simple layouts could not draw larger graphs readably — engineering
- With grid/circle/layered placement and the readability rules, training styles drew
  0–5 of 20 `two_cluster` and 7–15 of 20 `ring_chords` graphs; force-directed placement
  fixed it (19–20 of 20). Cost: layout is no longer a held-out style factor (`DECISIONS.md`).

### 5. Pixels are not reproducible across machines, content is — engineering
- Same seed and code: content digest identical on macOS/Python 3.9, macOS/Python 3.12 and
  Colab Linux/Python 3.13; image digest differed every time, even with the same Pillow
  version (`DECISIONS.md`, 2026-11-10).

### 6. One epoch over 16 samples was not enough to overfit; 320 steps was — engineering
- 96 steps: 8 of 16 right (no edited sample learned). 320 steps: 15 of 16
  (`runs/overfit_t4/metrics.json`, `runs/overfit_t4_320/metrics.json`).

### 7. Unattended Colab jobs are lost if not read back in time — engineering
- job5 (3-epoch pilot, about 3 T4-hours) disappeared with its runtime (`runs/budget.jsonl`).
- It happened again with job6: D was saved, but B (started 15:52Z) was gone by 16:48Z, when the
  notebook was reconnecting and Colab had no GPU backend to give. Hourly polling is too slow;
  a run that cannot be read back within ~30 min of a summary line can be lost.

### 8. Opening the notebook can starve a running Colab job of memory — engineering
- 2026-11-11T15:39Z, job6 evaluating: `free -m` showed 12,022 of 12,975 MB used, 953 MB
  available. Four `pyright-langserver` processes (Colab's editor helper) held 1.6–1.8 GB
  each; the evaluation itself 3.1 GB. Two of the four had started 85 s earlier, right
  after the status cell was edited. After `pkill` of the language servers: 3,845 MB used,
  job still running (cell output read at 15:40Z; not saved as a file).
- Likely cause, not verified: the notebook holds three cells with 150 kB base64 payloads
  and every open/edit starts an analyser on them. The status cell now kills the language
  servers first; keep the notebook tab closed between polls.

## Larger layout search does not remove the density bias of large drawings (engineering)

- 2026-11-12, `docs/layout_budget.md`: on the same seeds, raising `find_layout`'s budget from 180
  to 720 proposals raised 18-node acceptance from 0.40 to 0.47 (sparse_random) and 0.61 to 0.71
  (layered_dag), but accepted graphs stayed 3.7-4.9 edges sparser than rejected ones, at 2-7x
  the layout time. 120 graphs per cell, three styles; one generator seed stream.
- Consequence: size-OOD capped at 16 nodes; density reported per split.

## Discussed but never run (do not call these "tried")

- Recurrent visual working memory / latent loop modules (original idea, dropped before
  implementation because of closely related literature; brief section 3.3).
- Qwen3-VL-2B-Instruct as base (kept as fallback only).
- (Arm C, structural SFT, and arm E, parse-and-solve, were run on 2026-11-12: entry 6, `reports/pilot/c.md`, `reports/pilot/e.md`.)

## Kaggle launch, 2026-11-12
- job10 v1 on Kaggle failed after 42 s: the payload has no `assets/fonts`, so data generation stopped
  (font not found). Embedding the fonts made the script ~2.2 MB and Kaggle refused the push (HTTP 400).
  Fix: the kernel links matplotlib's bundled DejaVu files after checking each against the laptop's
  sha256 (`scripts/kaggle/make_kernel.py`); job.sh still asserts the data content digest.
