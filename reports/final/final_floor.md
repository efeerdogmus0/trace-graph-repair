Frozen base (arm A, jobF13): a reported floor, not a compared arm. Prompt variant `system`, 96 new tokens, no adapter, one run. Accuracy with a 95% interval over root graphs (2000 replicates).

| Split | Roots | Non-edit macro | minimal_graph_repair | reachability | critical_edge |
|---|---|---|---|---|---|
| test_id | 200 | 0.610 [0.590, 0.631] | 0.062 [0.040, 0.085] | 0.682 [0.650, 0.715] | 0.537 [0.515, 0.562] |
| test_style_ood | 200 | 0.630 [0.609, 0.651] | 0.062 [0.043, 0.087] | 0.718 [0.685, 0.750] | 0.542 [0.517, 0.568] |
| test_struct_ood | 200 | 0.589 [0.566, 0.611] | 0.065 [0.043, 0.090] | 0.632 [0.600, 0.665] | 0.545 [0.517, 0.573] |
| test_combined_ood | 100 | 0.555 [0.530, 0.580] | 0.060 [0.030, 0.090] | 0.600 [0.560, 0.645] | 0.510 [0.480, 0.540] |
| test_family_ood | 100 | 0.647 [0.613, 0.680] | 0.060 [0.030, 0.095] | 0.710 [0.660, 0.755] | 0.585 [0.540, 0.630] |
| test_size_ood | 100 | 0.620 [0.588, 0.650] | 0.060 [0.030, 0.090] | 0.705 [0.655, 0.750] | 0.535 [0.500, 0.570] |

Per task from the metrics file. Cell: strict / lenient / format-failure rate; responses cut at the limit.

| Split | minimal_graph_repair | reachability | critical_edge | Truncated |
|---|---|---|---|---|
| test_id | 0.062 / 0.060 / 0.062 | 0.682 / 0.000 / 0.682 | 0.537 / 0.000 / 0.537 | 0 |
| test_style_ood | 0.062 / 0.045 / 0.062 | 0.718 / 0.000 / 0.718 | 0.542 / 0.003 / 0.542 | 0 |
| test_struct_ood | 0.065 / 0.030 / 0.065 | 0.632 / 0.000 / 0.632 | 0.545 / 0.000 / 0.545 | 1 |
| test_combined_ood | 0.060 / 0.035 / 0.060 | 0.600 / 0.000 / 0.600 | 0.510 / 0.015 / 0.510 | 3 |
| test_family_ood | 0.060 / 0.020 / 0.060 | 0.710 / 0.000 / 0.710 | 0.585 / 0.000 / 0.585 | 0 |
| test_size_ood | 0.060 / 0.045 / 0.060 | 0.705 / 0.000 / 0.705 | 0.535 / 0.005 / 0.535 | 0 |

Repair detail. Cell: status accuracy / valid-edit rate / minimal-repair accuracy.

| Split | Repair |
|---|---|
| test_id | 0.465 / 0.125 / 0.125 |
| test_style_ood | 0.480 / 0.125 / 0.125 |
| test_struct_ood | 0.485 / 0.130 / 0.130 |
| test_combined_ood | 0.485 / 0.120 / 0.120 |
| test_family_ood | 0.490 / 0.120 / 0.120 |
| test_size_ood | 0.480 / 0.120 / 0.120 |
