# Arm C (structural SFT, three MCDGraph-inspired tasks), pilot dev

Run: Kaggle kernel `yiitefeerdomu/trace-job12`, Tesla T4, seed 3407, 3 epochs (768 updates), pilot train
(1,024 samples / 512 roots), evaluated in image mode on 200 dev roots (1,200 main samples). Collected with
`scripts/kaggle/collect.py job12 C` (flags re-scored locally from the downloaded predictions and equal to the
log line). Files: `runs/pilot3_C/summary.json`, `runs/pilot3_C/flags_dev.json`.

| | Repair | Reach. | Crit. edge | Non-edit macro |
|---|---|---|---|---|
| C (Kaggle) | 0.255 | 0.700 | 0.375 | 0.5375 [0.512, 0.564] |

- Train: 1,780 s, peak 8.42 GiB allocated / 12.47 reserved, loss (mean of first/last 100 steps) 0.385 -> 0.021,
  target tokens seen 91,920.
- Critical edge 0.375 is below chance: C answers `false` on 294 of 400 critical-edge samples (73.5%) and gets
  both members of a twin pair right on only 2.5% of pairs. Repair: valid edit 2.5%; most of its 0.255 comes
  from the "no edit needed" status.
- Paired on the same 200 roots (`reports/pilot/results_dev.md`, 2,000 root-graph bootstrap replicates):
  C - B_kaggle -0.259 [-0.294, -0.224]; D2_kaggle - C +0.095 [+0.0687, +0.1225]
  (`runs/pilot3_C/bootstrap_D2_vs_C_dev.json`, a = C, b = D2: C - D2 -0.095 [-0.1225, -0.0687]).

Reading (preliminary: one seed, dev, pilot scale): on the non-edit tasks, repair training (D2, and D on
Colab) is above the structural-SFT baseline C and below direct QA training B. C's structural tasks teach the
output format but no usable criticality judgement. The ordering B > D, D2 > C is what the paper's transfer
analysis reports; it does not reopen H1 (which compares D with B).
