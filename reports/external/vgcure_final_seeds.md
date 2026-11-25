# VGCure, pre-registered subset, three training seeds (jobV2 + jobV3 + jobV4)

Pre-registered subset `reports/external/vgcure_subset.json` (500 graphs, 1,500 items, sha256 ac4796f8…), fixed
2026-11-13 before any model output on VGCure (`docs/external_eval.md`); evaluated once, after all locked-test kernels
were collected. Adapted protocol: our instruction templates and strict JSON scoring, images letterboxed to 512x512;
not official VGCure numbers. Arms B, C, D, E at the locked-test seeds 3407 / 2027 / 9173 (jobV2 / jobV3 / jobV4 on
Kaggle T4, every run 1,500 of 1,500 responses, prediction hashes equal to the kernel's FLAGS-SUMMARY lines).
Pooled = mean of the three per-seed scores. Intervals: paired bootstrap over graphs, 10,000 resamples, the same draw
applied to every seed and arm (`scripts/vgcure_seed_pool.py`, records `vgcure_final_seeds_vs_{B,C}.json`); they
cover the choice of graphs, not the choice of training seed. No decision rule was pre-registered for this subset.

| arm | edge acc. | reach. balanced acc. | macro (pooled) | macro per seed (3407 / 2027 / 9173) |
|---|---|---|---|---|
| B QA-SFT | 0.818 | 0.814 | 0.816 | 0.816 / 0.816 / 0.816 |
| C structural | 0.817 | 0.622 | 0.720 | 0.733 / 0.694 / 0.733 |
| D repair | 0.867 | 0.877 | 0.872 | 0.874 / 0.862 / 0.880 |
| E parse-and-solve | 0.734 | 0.715 | 0.725 | 0.730 / 0.711 / 0.733 |

| difference | macro | edge acc. | reach. balanced acc. |
|---|---|---|---|
| D − B | +0.056 [+0.038, +0.075] | +0.049 [+0.033, +0.065] | +0.063 [+0.029, +0.097] |
| C − B | −0.096 [−0.119, −0.073] | −0.001 [−0.025, +0.023] | −0.191 [−0.232, −0.150] |
| E − B | −0.092 [−0.121, −0.061] | −0.084 [−0.112, −0.057] | −0.099 [−0.145, −0.051] |
| D − C | +0.152 [+0.135, +0.170] | +0.050 [+0.030, +0.071] | +0.254 [+0.226, +0.282] |
| E − C | +0.005 [−0.020, +0.029] | | |

Per seed, D − B macro is +0.057 [+0.033, +0.082], +0.046 [+0.023, +0.069], +0.064 [+0.044, +0.085]: every interval
above zero. On the exploratory VGCure-dev subset the same comparison was +0.022 [+0.002, +0.044].

E's parser (trained on our drawings) gave no usable graph for 27, 36 and 46 of the 500 images at the three seeds
(invalid JSON 12 / 12 / 26 images, malformed edge entry 15 / 24 / 20; each failed image costs its three items), the
style-OOD transcription failure `docs/external_eval.md` named in advance.

Reading (no rule, so stated as an observation). The locked test decides the hypothesis (D − B −0.105, negative); there
D is level with B on reachability (0.784 vs 0.789) and its deficit is in critical edge, a task VGCure does not have. On
VGCure's drawings (other authors, relation labels, unseen layout) D is ahead of B on both tasks VGCure has, at every
seed. This does not contradict the locked-test reading; it is one external dataset with adapted prompts, reported as
an observation for a dedicated test, not as a transfer result.
