Split `dev`, 200 root graphs, main samples. Descriptive counts from the tracked flag files; no intervals. "Both right" = both images of an answer-flipping twin pair answered correctly.

## Twin pairs

| Arm | Task | Pairs per seed | Both right per seed | Summed both / one / none right | Both-right rate |
|---|---|---|---|---|---|
| B | minimal_graph_repair | 200 | 2027: 2, 3407: 2, 9173: 2 | 6 / 350 / 244 | 0.010 |
| B | reachability | 200 | 2027: 133, 3407: 132, 9173: 129 | 394 / 206 / 0 | 0.657 |
| B | critical_edge | 200 | 2027: 113, 3407: 106, 9173: 106 | 325 / 271 / 4 | 0.542 |
| C | minimal_graph_repair | 200 | 2027: 0, 3407: 2, 9173: 1 | 3 / 296 / 301 | 0.005 |
| C | reachability | 200 | 2027: 65, 3407: 80, 9173: 84 | 229 / 369 / 2 | 0.382 |
| C | critical_edge | 200 | 2027: 1, 3407: 5, 9173: 2 | 8 / 474 / 118 | 0.013 |
| D | minimal_graph_repair | 200 | 2027: 60, 3407: 61, 9173: 58 | 179 / 286 / 135 | 0.298 |
| D | reachability | 200 | 2027: 128, 3407: 125, 9173: 124 | 377 / 223 / 0 | 0.628 |
| D | critical_edge | 200 | 2027: 25, 3407: 22, 9173: 19 | 66 / 491 / 43 | 0.110 |
| E | minimal_graph_repair | 200 | 2027: 146, 3407: 141, 9173: 140 | 427 / 152 / 21 | 0.712 |
| E | reachability | 200 | 2027: 169, 3407: 174, 9173: 167 | 510 / 90 / 0 | 0.850 |
| E | critical_edge | 200 | 2027: 146, 3407: 150, 9173: 142 | 438 / 101 / 61 | 0.730 |

## Answer bias (yes/no tasks)

| Arm | Task | Said true per seed | Said true, summed | True was right | No answer (summed) |
|---|---|---|---|---|---|
| B | reachability | 2027: 0.532, 3407: 0.525, 9173: 0.502 | 0.520 | 0.500 | 0 of 1200 |
| B | critical_edge | 2027: 0.412, 3407: 0.468, 9173: 0.487 | 0.456 | 0.500 | 0 of 1200 |
| C | reachability | 2027: 0.182, 3407: 0.245, 9173: 0.265 | 0.231 | 0.500 | 0 of 1200 |
| C | critical_edge | 2027: 0.075, 3407: 0.265, 9173: 0.255 | 0.198 | 0.500 | 0 of 1200 |
| D | reachability | 2027: 0.485, 3407: 0.477, 9173: 0.510 | 0.491 | 0.500 | 0 of 1200 |
| D | critical_edge | 2027: 0.733, 3407: 0.752, 9173: 0.882 | 0.789 | 0.500 | 0 of 1200 |
| E | reachability | 2027: 0.472, 3407: 0.465, 9173: 0.472 | 0.470 | 0.500 | 0 of 1200 |
| E | critical_edge | 2027: 0.517, 3407: 0.493, 9173: 0.506 | 0.505 | 0.500 | 159 of 1200 |
