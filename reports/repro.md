# GPU rerun from a clean clone (rule `repro-v1`)

Recorded run `runs/pilot3_D_kaggle`, rerun `runs/repro_D_s3407`; arm D, seed 3407, 1200 dev samples.

| task | recorded | rerun | difference |
|---|---|---|---|
| minimal_graph_repair | 0.5400 | 0.5400 | +0.0000 |
| reachability | 0.8125 | 0.8125 | +0.0000 |
| critical_edge | 0.5125 | 0.5125 | +0.0000 |
| non-edit macro | 0.6625 | 0.6625 | +0.0000 |

Per-sample correct/wrong agreement: 1.0000. Predictions byte-identical: yes.

Verdict: **identical** (tolerance: each task 0.03, macro 0.02).
