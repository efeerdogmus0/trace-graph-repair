Locked test, final evaluation (run once per trained run; 12 runs, three seeds per arm). Non-edit macro accuracy pooled over seeds, 95% interval over root graphs (2000 replicates). Rules: `src/trace_vr/seeds.py`.

| Split | Roots | B | D | E | C |
|---|---|---|---|---|---|
| test_id | 200 | 0.768 [0.746, 0.791] | 0.659 [0.642, 0.677] | 0.873 [0.848, 0.896] | 0.540 [0.522, 0.557] |
| test_style_ood | 200 | 0.804 [0.785, 0.824] | 0.683 [0.665, 0.701] | 0.883 [0.859, 0.908] | 0.541 [0.521, 0.562] |
| test_struct_ood | 200 | 0.713 [0.693, 0.735] | 0.627 [0.607, 0.648] | 0.725 [0.697, 0.756] | 0.533 [0.517, 0.550] |
| test_combined_ood | 100 | 0.749 [0.721, 0.777] | 0.648 [0.620, 0.677] | 0.751 [0.714, 0.789] | 0.516 [0.491, 0.541] |
| test_family_ood | 100 | 0.809 [0.779, 0.838] | 0.692 [0.662, 0.720] | 0.838 [0.800, 0.876] | 0.526 [0.501, 0.552] |
| test_size_ood | 100 | 0.777 [0.748, 0.806] | 0.665 [0.641, 0.689] | 0.754 [0.714, 0.791] | 0.533 [0.513, 0.553] |

Pooled difference to B (reading by the pre-registered rule):

| Split | D - B | E - B | C - B |
|---|---|---|---|
| test_id | -0.109 [-0.131, -0.088] negative | +0.104 [+0.077, +0.131] positive_clear | -0.229 [-0.257, -0.202] negative |
| test_style_ood | -0.121 [-0.143, -0.100] negative | +0.079 [+0.053, +0.105] positive_clear | -0.263 [-0.290, -0.236] negative |
| test_struct_ood | -0.086 [-0.106, -0.065] negative | +0.013 [-0.022, +0.047] inconclusive | -0.180 [-0.207, -0.152] negative |
| test_combined_ood | -0.101 [-0.129, -0.074] negative | +0.002 [-0.039, +0.045] inconclusive | -0.233 [-0.269, -0.196] negative |
| test_family_ood | -0.117 [-0.145, -0.091] negative | +0.029 [-0.010, +0.069] inconclusive | -0.283 [-0.316, -0.247] negative |
| test_size_ood | -0.113 [-0.142, -0.083] negative | -0.023 [-0.066, +0.020] inconclusive | -0.245 [-0.281, -0.209] negative |
