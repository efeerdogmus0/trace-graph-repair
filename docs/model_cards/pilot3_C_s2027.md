# TRACE adapter `pilot3_C_s2027`

LoRA adapter for `Qwen/Qwen3.5-2B` (revision `15852e8c16360a2fea060d615a32b45270f8a8fc`), arm **C**, seed 2027.

- Training objective: Structural fine-tuning (MCDGraph-inspired): graph description, edge infilling and graph matching.
- Produced by Kaggle kernel `trace-job16`; file `runs/kaggle/job16/runs/pilot3_C_s2027/adapter/adapter_model.safetensors`
- `adapter_model.safetensors`: 67332688 bytes, sha256 `677543c55170f876c283b222fcbdc9e249012f0c7c465dac8747aa7976f377fb`

## Scope

Research artifact from a pilot-scale study on synthetic drawn directed graphs (TRACE benchmark, `manifests/pilot.json`), three seeds per arm. Evaluated once on the locked test splits (section below); not evaluated on any external benchmark yet. Not intended for any use outside reproducing the TRACE experiments; the numbers say nothing about natural images or other graph drawings.

## Training

| Field | Value |
|---|---|
| manifest | `manifests/pilot.json` |
| split | `train` |
| content_digest | `ab499a7351a81aaaaaaaa668b785ae44dbd7d244c1c11eb5d42f2bd189eda5c8` |
| images_digest | `a5410cad199de20de17c5f30174e5e75065ededd64f89e2dda73ff590967b028` |
| root_graphs | `512` |
| samples | `1024` |
| epochs | `3` |
| accum | `4` |
| updates | `768` |
| lr | `0.0002` |
| lora_r | `16` |
| seed | `2027` |
| tokens_in_seen | `1197798` |
| target_tokens_seen | `91920` |
| train_seconds | `1784.8000` |
| mean_loss_first_100 | `0.3908` |
| mean_loss_last_100 | `0.0142` |
| gpu | `Tesla T4` |
| peak_allocated_gib | `8.4200` |
| torch | `2.11.0+cu130` |
| platform | `Linux-6.18.55+-x86_64-with-glibc2.39` |

LoRA target modules (regex): `model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)`

## Dev evaluation

200 dev root graphs, 1200 samples, input `image`, max_new_tokens 96, truncated 0.

| Task | n | accuracy | format failures |
|---|---|---|---|
| minimal_graph_repair | 400 | 0.250 | 0.000 |
| reachability | 400 | 0.657 | 0.000 |
| critical_edge | 400 | 0.470 | 0.000 |

Non-edit macro accuracy: **0.564**.

## Locked-test evaluation

Evaluated once in kernel `trace-jobF08` (arm **C**, seed 2027).

Accuracy of this seed per locked split, copied from `reports/final/final_table.json` (pooled-over-seed intervals and the pre-registered reading: `reports/final/final_primary.md`).

| Split | Roots | minimal_graph_repair | reachability | critical_edge | non_edit_macro_accuracy |
|---|---|---|---|---|---|
| test_id | 200 | 0.250 | 0.630 | 0.472 | 0.551 |
| test_style_ood | 200 | 0.250 | 0.647 | 0.453 | 0.550 |
| test_struct_ood | 200 | 0.250 | 0.590 | 0.475 | 0.532 |
| test_combined_ood | 100 | 0.240 | 0.595 | 0.475 | 0.535 |
| test_family_ood | 100 | 0.240 | 0.620 | 0.440 | 0.530 |
| test_size_ood | 100 | 0.250 | 0.610 | 0.470 | 0.540 |

## Licence

Adapter weights: CC BY 4.0 (`LICENSE-DATA.md`); training and evaluation code: MIT (`LICENSE`). The base model's own licence (Apache-2.0 per its model card) applies to the base weights, which are not included.
