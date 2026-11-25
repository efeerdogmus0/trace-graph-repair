# First pilot (single seed, small): D vs B

2026-11-10, Colab T4, `Qwen/Qwen3.5-2B` + LoRA r=16 on the language model, one epoch over
1,024 samples from 512 training root graphs per arm, seed 3407. Evaluation on the first 60
dev root graphs (360 samples), strict scoring. Numbers copied from the job log; raw
predictions stayed on the Colab VM. **Pilot only: one seed, 60 graphs, no confidence
intervals. Not a result about the hypothesis.**

| | D (repair only) | B (QA only) |
|---|---:|---:|
| Reachability accuracy (n=120) | 0.825 | 0.850 |
| Critical-edge accuracy (n=120) | 0.475 | 0.683 |
| **Non-edit macro accuracy** | **0.650** | **0.767** |
| Reachability pairs, both right (n=60) | 0.650 | 0.700 |
| Critical-edge pairs, both right (n=60) | 0.017 | 0.367 |
| Repair accuracy (n=120) | 0.467 | 0.242 |
| Repair: minimal edit right when an edit is needed (n=60) | 0.300 | 0.017 |
| Format failures | 0 | 0 |
| Training time / peak reserved memory | 714 s / 14.2 GiB | 623 s / 8.0 GiB |

Frozen base (A), first 30 dev graphs (180 samples), 160 new tokens allowed:

| | strict | lenient (last JSON object in the text) |
|---|---:|---:|
| Reachability (n=60) | 0.133 | 0.567 |
| Critical edge (n=60) | 0.067 | 0.233 |
| Repair (n=60) | 0.000 | 0.000 |

99 of the 180 base responses were cut off before any final answer, so the lenient column
is a lower bound on what the base model "meant", not a fair floor yet. A and the trained
arms were also evaluated on different numbers of graphs (30 vs 60).

## What this does and does not say

- On this pilot B is ahead of D on the non-edit macro score by about 12 points. The
  hypothesis needs D ahead of B. As it stands the pilot does not support it.
- D was never shown a reachability answer and still gets 0.825 on reachability, close to
  B's 0.850, and gets both twins right on 65% of pairs. The frozen base is at 0.567
  lenient on reachability (about chance, and a lower bound because of cut-off answers).
  So repair training alone moved reachability a long way; how much of that is specific to
  the repair objective is exactly what the B comparison is for, and there B is level.
- D is at chance on critical edge (0.475, pairs 0.017). B learns it only partly (0.683).
- Both arms are far from having learned their own task after one epoch (D repair 0.467,
  training loss still falling), so the comparison is between two under-trained models.
- 60 graphs give wide uncertainty: a difference of a few points between 0.825 and 0.850
  means nothing here.

## Next, in order (brief section 20: diagnose before scaling)

1. Give the frozen base a fair run (higher token limit or a no-reasoning instruction), same 200 dev graphs.
2. Train both arms longer (3 epochs) and evaluate on all 200 dev graphs with bootstrap
   intervals over graphs. One epoch was a cost probe, not a fair test.
3. Look at D's critical-edge errors: does it answer one value for both twins?
4. Only then decide on the first of the two allowed revisions.
