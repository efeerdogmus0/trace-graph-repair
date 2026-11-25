Locked test, per-task detail (12 runs, three seeds per arm). Every cell is the mean over the three seeds of the value in that seed's metrics file. Descriptive, no intervals. Cell: strict accuracy / lenient accuracy / format-failure rate.

| Split | Task | Questions | B | D | E | C |
|---|---|---|---|---|---|---|
| test_id | minimal_graph_repair | 400 | 0.265 / 0.026 / 0.265 | 0.521 / 0.000 / 0.521 | 0.844 / 0.000 / 0.844 | 0.253 / 0.000 / 0.253 |
| test_id | reachability | 400 | 0.796 / 0.000 / 0.796 | 0.792 / 0.000 / 0.792 | 0.910 / 0.001 / 0.910 | 0.663 / 0.000 / 0.663 |
| test_id | critical_edge | 400 | 0.741 / 0.000 / 0.741 | 0.527 / 0.000 / 0.527 | 0.835 / 0.096 / 0.835 | 0.416 / 0.000 / 0.416 |
| test_style_ood | minimal_graph_repair | 400 | 0.293 / 0.032 / 0.293 | 0.553 / 0.000 / 0.553 | 0.856 / 0.002 / 0.856 | 0.251 / 0.000 / 0.251 |
| test_style_ood | reachability | 400 | 0.836 / 0.000 / 0.836 | 0.820 / 0.000 / 0.820 | 0.928 / 0.001 / 0.928 | 0.682 / 0.000 / 0.682 |
| test_style_ood | critical_edge | 400 | 0.772 / 0.000 / 0.772 | 0.547 / 0.000 / 0.547 | 0.839 / 0.099 / 0.839 | 0.399 / 0.000 / 0.399 |
| test_struct_ood | minimal_graph_repair | 400 | 0.240 / 0.027 / 0.240 | 0.403 / 0.000 / 0.403 | 0.669 / 0.000 / 0.669 | 0.246 / 0.000 / 0.246 |
| test_struct_ood | reachability | 400 | 0.734 / 0.000 / 0.734 | 0.740 / 0.000 / 0.740 | 0.812 / 0.001 / 0.812 | 0.625 / 0.000 / 0.625 |
| test_struct_ood | critical_edge | 400 | 0.692 / 0.000 / 0.692 | 0.514 / 0.000 / 0.514 | 0.638 / 0.273 / 0.638 | 0.441 / 0.000 / 0.441 |
| test_combined_ood | minimal_graph_repair | 200 | 0.253 / 0.017 / 0.253 | 0.435 / 0.000 / 0.435 | 0.710 / 0.000 / 0.710 | 0.243 / 0.000 / 0.243 |
| test_combined_ood | reachability | 200 | 0.778 / 0.000 / 0.778 | 0.755 / 0.000 / 0.755 | 0.852 / 0.000 / 0.852 | 0.623 / 0.000 / 0.623 |
| test_combined_ood | critical_edge | 200 | 0.720 / 0.000 / 0.720 | 0.542 / 0.000 / 0.542 | 0.650 / 0.252 / 0.650 | 0.408 / 0.000 / 0.408 |
| test_family_ood | minimal_graph_repair | 200 | 0.260 / 0.033 / 0.260 | 0.517 / 0.000 / 0.517 | 0.837 / 0.002 / 0.837 | 0.240 / 0.000 / 0.240 |
| test_family_ood | reachability | 200 | 0.827 / 0.000 / 0.827 | 0.792 / 0.000 / 0.792 | 0.882 / 0.002 / 0.882 | 0.658 / 0.000 / 0.658 |
| test_family_ood | critical_edge | 200 | 0.792 / 0.000 / 0.792 | 0.592 / 0.000 / 0.592 | 0.795 / 0.145 / 0.795 | 0.393 / 0.000 / 0.393 |
| test_size_ood | minimal_graph_repair | 200 | 0.273 / 0.025 / 0.273 | 0.488 / 0.000 / 0.488 | 0.717 / 0.002 / 0.717 | 0.250 / 0.000 / 0.250 |
| test_size_ood | reachability | 200 | 0.807 / 0.000 / 0.807 | 0.813 / 0.000 / 0.813 | 0.852 / 0.002 / 0.852 | 0.655 / 0.000 / 0.655 |
| test_size_ood | critical_edge | 200 | 0.748 / 0.000 / 0.748 | 0.517 / 0.000 / 0.517 | 0.657 / 0.242 / 0.657 | 0.410 / 0.000 / 0.410 |

Repair detail. Cell: status accuracy / valid-edit rate / minimal-repair accuracy (mean over seeds).

| Split | B | D | E | C |
|---|---|---|---|---|
| test_id | 0.346 / 0.033 / 0.033 | 0.580 / 0.383 / 0.383 | 0.857 / 0.820 / 0.820 | 0.271 / 0.008 / 0.008 |
| test_style_ood | 0.361 / 0.033 / 0.033 | 0.617 / 0.448 / 0.448 | 0.865 / 0.828 / 0.828 | 0.269 / 0.012 / 0.012 |
| test_struct_ood | 0.326 / 0.037 / 0.037 | 0.526 / 0.282 / 0.282 | 0.690 / 0.578 / 0.578 | 0.264 / 0.003 / 0.003 |
| test_combined_ood | 0.363 / 0.040 / 0.040 | 0.567 / 0.380 / 0.380 | 0.742 / 0.637 / 0.637 | 0.267 / 0.010 / 0.010 |
| test_family_ood | 0.323 / 0.030 / 0.030 | 0.592 / 0.383 / 0.383 | 0.847 / 0.790 / 0.790 | 0.267 / 0.017 / 0.017 |
| test_size_ood | 0.370 / 0.027 / 0.027 | 0.580 / 0.360 / 0.360 | 0.743 / 0.613 / 0.613 | 0.272 / 0.013 / 0.013 |

Responses cut at the token limit, per seed (2027, 3407, 9173).

| Split | B | D | E | C |
|---|---|---|---|---|
| test_id | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| test_style_ood | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| test_struct_ood | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| test_combined_ood | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| test_family_ood | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| test_size_ood | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |

Extraction quality of arm E (strict parse, mean over seeds).

| Split | exact | format ok | edge precision | edge recall | direction accuracy | node recall |
|---|---|---|---|---|---|---|
| test_id | 0.441 | 1.000 | 0.913 | 0.920 | 0.966 | 1.000 |
| test_style_ood | 0.532 | 0.999 | 0.939 | 0.942 | 0.973 | 1.000 |
| test_struct_ood | 0.079 | 1.000 | 0.856 | 0.853 | 0.931 | 0.998 |
| test_combined_ood | 0.115 | 1.000 | 0.872 | 0.871 | 0.932 | 0.998 |
| test_family_ood | 0.476 | 0.999 | 0.921 | 0.922 | 0.967 | 1.000 |
| test_size_ood | 0.156 | 0.999 | 0.874 | 0.876 | 0.959 | 0.999 |
