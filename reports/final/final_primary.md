Locked test, pre-registered primary statistic: unweighted mean of six accuracies {reachability, critical_edge} x {test_id, test_style_ood, test_struct_ood}, main samples, pooled over three seeds; 95% stratified paired bootstrap over root graphs (2000 replicates, primary-v1). Only this statistic decides the hypothesis.

| Arm | Primary | reachability mean | critical_edge mean | Per seed (primary) |
|---|---|---|---|---|
| B | 0.762 [0.750, 0.773] | 0.789 [0.771, 0.805] | 0.735 [0.717, 0.750] | 2027: 0.756, 3407: 0.769, 9173: 0.760 |
| D | 0.657 [0.646, 0.667] | 0.784 [0.766, 0.802] | 0.529 [0.516, 0.542] | 2027: 0.652, 3407: 0.654, 9173: 0.664 |
| E | 0.827 [0.812, 0.842] | 0.883 [0.869, 0.898] | 0.771 [0.747, 0.794] | 2027: 0.824, 3407: 0.830, 9173: 0.828 |
| C | 0.538 [0.528, 0.549] | 0.657 [0.640, 0.674] | 0.419 [0.407, 0.431] | 2027: 0.545, 3407: 0.531, 9173: 0.537 |

| Difference | Primary | Reading | Seed-matched | Task drop > 0.02 | Splits same sign |
|---|---|---|---|---|---|
| D - B | -0.105 [-0.117, -0.093] | negative | 2027: -0.105, 3407: -0.115, 9173: -0.096 | critical_edge | yes |
| E - B | +0.065 [+0.048, +0.082] | positive_clear | 2027: +0.068, 3407: +0.061, 9173: +0.068 | none | yes |
| C - B | -0.224 [-0.239, -0.208] | negative | 2027: -0.212, 3407: -0.238, 9173: -0.223 | reachability, critical_edge | yes |
| E - D | +0.171 [+0.154, +0.188] | positive_clear | 2027: +0.172, 3407: +0.176, 9173: +0.163 | none | yes |

Rules: `scripts/final_primary.py` docstring, `src/trace_vr/seeds.py`; the E - D row is reported, it does not change any reading against the reference.
