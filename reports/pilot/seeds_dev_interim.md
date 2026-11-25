Dev split `dev`, 200 root graphs, main samples; pooled = mean over seeds on the same resampled roots (2000 replicates, 95%, root sampling only). Rules: `src/trace_vr/seeds.py`.

| Arm | Seeds | Macro per seed | Pooled macro | Pooled - B | Seed-matched diffs | Reading |
|---|---|---|---|---|---|---|
| B | 3 | 2027: 0.805, 3407: 0.796, 9173: 0.792 | 0.798 [0.776, 0.822] | ref | - | - |
| C | 3 | 2027: 0.564, 3407: 0.537, 9173: 0.545 | 0.549 [0.530, 0.569] | -0.249 [-0.277, -0.222] | 2027: -0.241, 3407: -0.259, 9173: -0.248 | negative |
| D | 3 | 2027: 0.669, 3407: 0.662, 9173: 0.669 | 0.667 [0.647, 0.686] | -0.131 [-0.153, -0.110] | 2027: -0.136, 3407: -0.134, 9173: -0.124 | negative |
| E | 3 | 2027: 0.871, 3407: 0.884, 9173: 0.854 | 0.870 [0.844, 0.894] | +0.072 [+0.042, +0.102] | 2027: +0.066, 3407: +0.088, 9173: +0.061 | positive_clear |
