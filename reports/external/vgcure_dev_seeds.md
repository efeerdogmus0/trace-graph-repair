# VGCure-dev, three training seeds (jobV0 + jobV1) — exploratory

Exploratory subset `reports/external/vgcure_dev_subset.json` (400 graphs, 1,200 items, sha256 6ce6782d…, no graph
shared with the pre-registered subset). VGCure images letterboxed to 512x512, adapted prompts; not official VGCure
numbers. Arms trained only on our own generated data, seeds 3407 / 2027 / 9173. Pooled = mean of the three per-seed
scores. Intervals: paired bootstrap over graphs, 10,000 resamples, the same draw applied to every seed and arm
(`scripts/vgcure_seed_pool.py --dev`); they cover the choice of graphs, not the choice of training seed.

| arm | edge acc. | reach. balanced acc. | macro (pooled) | macro per seed |
|---|---|---|---|---|
| B QA-SFT | 0.821 | 0.843 | 0.832 | 0.837 / 0.834 / 0.826 |
| C structural | 0.800 | 0.629 | 0.715 | 0.729 / 0.691 / 0.724 |
| D repair | 0.860 | 0.848 | 0.854 | 0.854 / 0.853 / 0.856 |

| difference | macro | edge acc. | reach. balanced acc. |
|---|---|---|---|
| D − C | +0.140 [+0.115, +0.163] | +0.061 [+0.036, +0.085] | +0.219 [+0.176, +0.258] |
| B − C | +0.118 [+0.091, +0.143] | +0.021 [−0.008, +0.050] | +0.214 [+0.168, +0.256] |
| D − B | +0.022 [+0.002, +0.044] | +0.040 [+0.021, +0.058] | +0.005 [−0.032, +0.043] |

Per seed, D − C macro is +0.126, +0.162, +0.132 (every interval above zero). D − B macro is +0.018 [−0.012, +0.048],
+0.019 [−0.004, +0.042], +0.030 [+0.004, +0.056]: positive at every seed, above zero alone at one.

Notes. D's lead over B comes from the edge-existence items; on reachability the two are level. C answers "no" to most
reachability items at every seed (yes-accuracy 0.307 / 0.227 / 0.334, no-accuracy 0.985 / 0.969 / 0.954). Only 65 of
the 400 reachability items have a "no" answer. C is our adaptation of MCDGraph-style structural training on our data,
not the MCDGraph authors' model. This subset was looked at; a claim needs the single pre-registered run.
