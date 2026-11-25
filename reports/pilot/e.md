# Arm E (parse the drawing, solve exactly), pilot dev

Run: Kaggle kernel `yiitefeerdomu/trace-job13`, Tesla T4, seed 3407. The parser is the `C_desc` adapter
(description-only structural SFT, 3 epochs, 768 updates, pilot train 1,024 samples / 512 roots); every dev
question is then answered by the exact solver on the transcribed graph. Evaluated in image mode on 200 dev roots
(1,200 main samples, 763 images). Collected with `scripts/kaggle/collect.py job13 E` (flags re-scored locally
from the downloaded predictions and equal to the log line). Files: `runs/pilot3_E/summary.json`,
`runs/pilot3_E/flags_dev.json`, `runs/pilot3_E/extraction_dev.jsonl`.

| | Repair | Reach. | Crit. edge | Non-edit macro |
|---|---|---|---|---|
| E (Kaggle) | 0.830 | 0.935 | 0.8325 | 0.884 [0.857, 0.909] |

- Paired vs B_kaggle on the same 200 roots (`reports/pilot/results_dev.md`, 2,000 replicates):
  +0.088 [+0.054, +0.121]. E is the best arm on every task.
- Train (C_desc): 1,808 s, peak 8.45 GiB allocated / 14.08 reserved, loss 0.239 -> 0.0042 (first/last 100 steps),
  target tokens 195,996. Eval 4,211 s (long: up to 512 new tokens per image for the edge list), 0 truncated.
- Extraction (strict = lenient): exact graph on 0.4705 of images (359 / 763), edge precision 0.920,
  recall 0.928, direction 0.967, node recall 0.9998. 47 of 1,200 samples (all critical edge, 11.75% of that
  task) could not be answered because the question's premise fails on the transcribed graph; they count as wrong.
- Perception split (pre-registered 2026-11-12T20:40Z, `reports/pilot/perception_split_dev.md`; readable =
  E's transcription exact; 135 readable / 147 unreadable roots): D_colab - B on readable images
  -0.148 [-0.190, -0.105], on unreadable -0.105 [-0.143, -0.063] -> reasoning_gap; D2 and C also
  reasoning_gap. E's 1.000 on readable images holds by construction (exact graph + exact solver).

Reading (preliminary: one seed, dev, pilot scale): the end-to-end arms' deficit is not explained by unreadable
drawings; even where the drawing can be transcribed exactly, repair-trained D answers worse than B. A 2B model
that only learns to transcribe the drawing, plus a solver, beats all end-to-end fine-tunes. Caveat: readability
is measured by E's parser, a proxy for drawing difficulty, not what the other arms perceived; E uses an external
exact solver and so is an upper-bound style pipeline baseline, not a competitor on equal terms.
