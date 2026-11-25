# VGCure-dev, seed 3407 (jobV0) — exploratory

Exploratory subset `reports/external/vgcure_dev_subset.json` (400 graphs, 1,200 items, sha256 6ce6782d…, no graph
shared with the pre-registered subset). VGCure images letterboxed to 512x512, adapted prompts; not official VGCure
numbers. One training seed (3407). Kaggle T4, kernel `trace-jobv0`. Intervals: paired bootstrap over graphs,
10,000 resamples (`scripts/vgcure_compare.py --dev`).

| arm | edge acc. | reach. balanced acc. | macro | format ok |
|---|---|---|---|---|
| A base (no adapter) | 0.115 | 0.063 | 0.089 | 0.113 |
| B QA-SFT | 0.833 | 0.841 | 0.837 | 1.000 |
| C structural | 0.811 | 0.646 | 0.729 | 1.000 |
| D repair | 0.855 | 0.854 | 0.854 | 1.000 |
| E parse-and-solve | 0.745 | 0.702 | 0.723 | 0.945 |

Macro differences: D − C +0.126 [+0.098, +0.151]; B − C +0.108 [+0.076, +0.138]; E − C −0.005 [−0.040, +0.027];
D − B +0.018 [−0.012, +0.048]; E − B −0.113 [−0.151, −0.076].

Notes. The base model's score is a format failure (498 of 1,200 generations hit the 96-token limit, 11% parse), not a
measure of what it can read. C answers "no" to most reachability items (yes-accuracy 0.307, no-accuracy 0.985). Only 65
of the 400 reachability items have a "no" answer, so the balanced accuracy rests on a small class. E fails to produce a
usable transcription for 66 items (36 invalid JSON, 30 bad edge) and trails B here, the reverse of our own dev set.
Seeds 2027 and 9173 of B, C, D run in jobV1.
