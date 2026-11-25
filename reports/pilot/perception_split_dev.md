Arm E extraction `runs/pilot3_E/extraction_dev.jsonl`: 359 of 763 images read exactly. Readable subset 572 samples / 135 roots; unreadable 628 / 147.

| Arm | macro readable | macro unreadable | vs B_kaggle readable | vs B_kaggle unreadable | reading |
|---|---|---|---|---|---|
| B_kaggle | 0.846 [0.810, 0.881] | 0.751 [0.714, 0.786] | - | - | ref |
| C_kaggle | 0.540 [0.501, 0.575] | 0.532 [0.495, 0.567] | -0.306 [-0.360, -0.258] | -0.218 [-0.266, -0.170] | reasoning_gap |
| E_kaggle | 1.000 [1.000, 1.000] | 0.782 [0.735, 0.826] | 0.154 [0.119, 0.190] | 0.031 [-0.021, 0.081] | no_gap |
| D_colab | 0.698 [0.665, 0.733] | 0.646 [0.608, 0.680] | -0.148 [-0.190, -0.105] | -0.105 [-0.143, -0.063] | reasoning_gap |
| D2_kaggle | 0.634 [0.603, 0.663] | 0.630 [0.600, 0.661] | -0.212 [-0.249, -0.175] | -0.121 [-0.157, -0.082] | reasoning_gap |
