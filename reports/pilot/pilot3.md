# Pilot 3 (single seed, 3 epochs, 200 dev graphs): D versus B

2026-11-11, Colab T4 (job6), `Qwen/Qwen3.5-2B` + LoRA r=16, three epochs over 1,024
samples from 512 training root graphs, seed 3407. Evaluation on all 200 dev root graphs
(1,200 samples), strict scoring, 96 new tokens. **Pilot only: one seed, dev split.** D ran
as Colab job6, B as job7 on a fresh VM (job6's B was lost); both builds have the same
content digest `ab499a73...` and images digest `a5410cad...`, so samples are paired.

Where the numbers come from: the `EVAL-SUMMARY` line of the job log and a per-sample bit
string (`runs/pilot3_D/flags_dev.json`, checked against a hash computed on the Colab
side). Intervals: 2,000 bootstrap resamples of root graphs, 95% percentile
(`runs/pilot3_D/bootstrap_dev.json`). Raw predictions stayed on the Colab VM
(sha256 `7604ebcf...`).

## Arm D (repair only)

| | value | 95% interval |
|---|---:|---:|
| Reachability accuracy (n=400) | 0.840 | 0.808 – 0.873 |
| Critical-edge accuracy (n=400) | 0.503 | 0.473 – 0.535 |
| **Non-edit macro accuracy** | **0.671** | 0.648 – 0.694 |
| Repair accuracy (n=400) | 0.545 | 0.495 – 0.595 |
| Repair: minimal edit right when an edit is needed (n=200) | 0.395 | not computed |
| Repair status accuracy | 0.623 | not computed |
| Reachability pairs, both right (n=200) | 0.680 | not computed |
| Critical-edge pairs, both right (n=200) | 0.095 | not computed |
| Format failures / answers cut off | 0 / 0 | |
| Training: 2,196 s, loss (mean of last 100 samples) 0.035, peak reserved 14.2 GiB | | |
| Evaluation: 1,539 s | | |

## What the per-sample flags show (`runs/pilot3_D/breakdown_dev.json`)

- Critical edge is at chance and biased: D answered "true" on 75.2% of questions (true is
  right for 50%). Accuracy 0.755 when the answer is true, 0.250 when it is false. Of 200
  twin pairs: both right 19, exactly one right 163, both wrong 18. So D mostly gives the
  same answer to both twins: it does not use the one-edge difference for this question.
- Reachability is not biased (said "true" 46.0%); 136 of 200 pairs both right, none both
  wrong. Edited images are harder than root images (0.800 vs 0.880).
- Repair is still not learned after three epochs: 0.395 when an edit is needed, 0.600 for
  "impossible", 0.790 for "unchanged". Training loss fell from 0.177 to 0.035, so part of
  the gap between training loss and dev accuracy may be memorisation of 512 graphs; not
  tested (no accuracy on training graphs was measured).
- No slice by family, size, layout or style stands out beyond what 40–270 samples per
  slice can show.

## Compared with the one-epoch pilot

One epoch, first 60 dev graphs: macro 0.650, reachability 0.825, critical edge 0.475,
repair 0.467. Three epochs, 200 dev graphs: 0.671, 0.840, 0.503, 0.545. Different numbers
of graphs, so only the direction is informative: two more epochs changed the non-edit
scores very little.

## Arm B (QA-SFT) and the paired comparison

B: Colab job7, training 1,848 s, loss (last 100) 0.058, peak reserved 8.0 GiB; evaluation
1,371 s, 4 format failures of 1,200, none cut off (`runs/pilot3_B/summary.json`,
`runs/pilot3_B/flags_dev.json`, predictions sha256 `db7e57b7...`, checked on the Colab side).
Intervals: 2,000 paired bootstrap resamples of the 200 root graphs
(`runs/pilot3_B/bootstrap_D_vs_B_dev.json`).

| | D | B | D - B (95% interval) |
|---|---:|---:|---:|
| Reachability | 0.840 | 0.845 | -0.005 (-0.035 to +0.025) |
| Critical edge | 0.503 | 0.770 | -0.268 (-0.313 to -0.223) |
| **Non-edit macro** | 0.671 | **0.8075** | **-0.136 (-0.163 to -0.109)** |
| Repair | 0.545 | 0.295 | +0.250 (+0.193 to +0.308) |

- **The pilot does not support H1.** B is ahead on the non-edit macro in every bootstrap
  replicate. B is trained on the very question types that are evaluated, so this is the
  strong in-distribution baseline; the brief's success criterion needs D to beat it.
- Reachability: D (never trained on it) equals B (trained on it). This is the one positive
  transfer signal and it held over two pilots.
- Critical edge: B is not biased (said "true" 48%) and gets both twins right on 108 of 200
  pairs; D answered "true" 75% and got both twins right on 19. The gap is all here.
- B on repair (not trained on it): 0.900 on "impossible", 0.250 "unchanged", 0.015 when an
  edit is needed, i.e. it mostly answers "impossible".
- Slices for B (`runs/pilot3_B/breakdown_dev.json`): reachability drops on 9–12-node graphs
  (0.776 vs 0.880); nothing else stands out at this sample size.

## Still to do

1. Oracle-graph text condition for B (job8, 100 dev roots, `edges` and `image+edges`):
   tells whether B's remaining errors are perception or reasoning. D needs a retrain to get
   the same numbers (its adapter was lost with job6's VM).
2. Arm E (parse + solve) with the real model.
3. Only then spend revision 1 of 2 (section 19/20): the evidence points at edge criticality,
   e.g. a D variant whose repair targets include break-a-path goals (critical-edge-like
   edits), decided on dev only.
