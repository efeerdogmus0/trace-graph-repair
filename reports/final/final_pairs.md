Locked test, answer-flipping twin pairs (two images that differ by one edge and have opposite answers). Per arm: pairs with both images right, counts summed over three seeds (12 runs). Descriptive, no intervals.

| Split | Task | Pairs per seed | B | D | E | C |
|---|---|---|---|---|---|---|
| test_id | minimal_graph_repair | 200 | 6 of 600 (0.010) | 142 of 600 (0.237) | 441 of 600 (0.735) | 2 of 600 (0.003) |
| test_id | reachability | 200 | 355 of 600 (0.592) | 352 of 600 (0.587) | 495 of 600 (0.825) | 199 of 600 (0.332) |
| test_id | critical_edge | 200 | 294 of 600 (0.490) | 69 of 600 (0.115) | 436 of 600 (0.727) | 5 of 600 (0.008) |
| test_style_ood | minimal_graph_repair | 200 | 7 of 600 (0.012) | 188 of 600 (0.313) | 449 of 600 (0.748) | 2 of 600 (0.003) |
| test_style_ood | reachability | 200 | 404 of 600 (0.673) | 384 of 600 (0.640) | 514 of 600 (0.857) | 220 of 600 (0.367) |
| test_style_ood | critical_edge | 200 | 329 of 600 (0.548) | 81 of 600 (0.135) | 444 of 600 (0.740) | 4 of 600 (0.007) |
| test_struct_ood | minimal_graph_repair | 200 | 4 of 600 (0.007) | 78 of 600 (0.130) | 271 of 600 (0.452) | 1 of 600 (0.002) |
| test_struct_ood | reachability | 200 | 281 of 600 (0.468) | 291 of 600 (0.485) | 385 of 600 (0.642) | 154 of 600 (0.257) |
| test_struct_ood | critical_edge | 200 | 233 of 600 (0.388) | 78 of 600 (0.130) | 289 of 600 (0.482) | 7 of 600 (0.012) |
| test_combined_ood | minimal_graph_repair | 100 | 4 of 300 (0.013) | 53 of 300 (0.177) | 160 of 300 (0.533) | 0 of 300 (0.000) |
| test_combined_ood | reachability | 100 | 167 of 300 (0.557) | 154 of 300 (0.513) | 211 of 300 (0.703) | 76 of 300 (0.253) |
| test_combined_ood | critical_edge | 100 | 133 of 300 (0.443) | 44 of 300 (0.147) | 137 of 300 (0.457) | 1 of 300 (0.003) |
| test_family_ood | minimal_graph_repair | 100 | 3 of 300 (0.010) | 77 of 300 (0.257) | 210 of 300 (0.700) | 2 of 300 (0.007) |
| test_family_ood | reachability | 100 | 196 of 300 (0.653) | 175 of 300 (0.583) | 229 of 300 (0.763) | 96 of 300 (0.320) |
| test_family_ood | critical_edge | 100 | 175 of 300 (0.583) | 70 of 300 (0.233) | 206 of 300 (0.687) | 1 of 300 (0.003) |
| test_size_ood | minimal_graph_repair | 100 | 2 of 300 (0.007) | 65 of 300 (0.217) | 153 of 300 (0.510) | 2 of 300 (0.007) |
| test_size_ood | reachability | 100 | 184 of 300 (0.613) | 188 of 300 (0.627) | 212 of 300 (0.707) | 94 of 300 (0.313) |
| test_size_ood | critical_edge | 100 | 150 of 300 (0.500) | 29 of 300 (0.097) | 142 of 300 (0.473) | 1 of 300 (0.003) |

Share of answered yes/no questions answered true, summed over seeds (last column: share where true is correct).

| Split | Task | B | D | E | C | True is correct |
|---|---|---|---|---|---|---|
| test_id | reachability | 0.532 | 0.483 | 0.458 | 0.222 | 0.500 |
| test_id | critical_edge | 0.494 | 0.797 | 0.505 | 0.166 | 0.500 |
| test_style_ood | reachability | 0.527 | 0.497 | 0.474 | 0.231 | 0.500 |
| test_style_ood | critical_edge | 0.502 | 0.820 | 0.511 | 0.203 | 0.500 |
| test_struct_ood | reachability | 0.481 | 0.482 | 0.365 | 0.192 | 0.500 |
| test_struct_ood | critical_edge | 0.567 | 0.759 | 0.510 | 0.176 | 0.500 |
| test_combined_ood | reachability | 0.468 | 0.498 | 0.395 | 0.207 | 0.500 |
| test_combined_ood | critical_edge | 0.623 | 0.772 | 0.479 | 0.185 | 0.500 |
| test_family_ood | reachability | 0.443 | 0.445 | 0.412 | 0.195 | 0.500 |
| test_family_ood | critical_edge | 0.482 | 0.695 | 0.509 | 0.153 | 0.500 |
| test_size_ood | reachability | 0.527 | 0.527 | 0.406 | 0.208 | 0.500 |
| test_size_ood | critical_edge | 0.542 | 0.847 | 0.547 | 0.173 | 0.500 |
