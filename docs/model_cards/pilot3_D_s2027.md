# TRACE adapter `pilot3_D_s2027`

LoRA adapter for `Qwen/Qwen3.5-2B` (revision `15852e8c16360a2fea060d615a32b45270f8a8fc`), arm **D**, seed 2027.

- Training objective: Repair fine-tuning: proposes one minimal edge edit that makes the reachability goals hold.
- Produced by Kaggle kernel `trace-job15`; file `runs/kaggle/job15/runs/pilot3_D_s2027/adapter/adapter_model.safetensors`
- `adapter_model.safetensors`: 67332688 bytes, sha256 `516f7616a53695f02b9439ca79fee12dd9d573b83b7855036f879e4389b9a9dc`

## Scope

Research artifact from a pilot-scale study on synthetic drawn directed graphs (TRACE benchmark, `manifests/pilot.json`), three seeds per arm. Evaluated once on the locked test splits and once on a pre-registered, adapted VGCure subset (sections below). Not intended for any use outside reproducing the TRACE experiments; the numbers say nothing about natural images or other graph drawings.

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
| tokens_in_seen | `1471362` |
| target_tokens_seen | `55296` |
| train_seconds | `2003.9000` |
| mean_loss_first_100 | `0.1890` |
| mean_loss_last_100 | `0.0435` |
| gpu | `Tesla T4` |
| peak_allocated_gib | `9.0100` |
| torch | `2.11.0+cu130` |
| platform | `Linux-6.18.55+-x86_64-with-glibc2.39` |

LoRA target modules (regex): `model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)`

## Dev evaluation

200 dev root graphs, 1200 samples, input `image`, max_new_tokens 96, truncated 0.

| Task | n | accuracy | format failures |
|---|---|---|---|
| minimal_graph_repair | 400 | 0.527 | 0.000 |
| reachability | 400 | 0.820 | 0.000 |
| critical_edge | 400 | 0.517 | 0.000 |

Non-edit macro accuracy: **0.669**.

## Locked-test evaluation

Evaluated once in kernel `trace-jobF06` (arm **D**, seed 2027).

Accuracy of this seed per locked split, copied from `reports/final/final_table.json` (pooled-over-seed intervals and the pre-registered reading: `reports/final/final_primary.md`).

| Split | Roots | minimal_graph_repair | reachability | critical_edge | non_edit_macro_accuracy |
|---|---|---|---|---|---|
| test_id | 200 | 0.535 | 0.782 | 0.532 | 0.657 |
| test_style_ood | 200 | 0.552 | 0.818 | 0.545 | 0.681 |
| test_struct_ood | 200 | 0.405 | 0.738 | 0.495 | 0.616 |
| test_combined_ood | 100 | 0.435 | 0.760 | 0.530 | 0.645 |
| test_family_ood | 100 | 0.550 | 0.790 | 0.605 | 0.698 |
| test_size_ood | 100 | 0.485 | 0.820 | 0.520 | 0.670 |

## External evaluation (pre-registered VGCure subset)

Evaluated once on 500 VGCure graphs (1500 items), adapted protocol (our prompts, images letterboxed to 512x512, strict JSON scoring; not official VGCure numbers). Run `runs/kaggle/jobV3/pilot3_D_s2027`; record `reports/external/vgcure_final_seeds_vs_B.json`.

| Edge-existence accuracy | Reachability balanced accuracy | External macro |
|---|---|---|
| 0.872 | 0.852 | 0.862 |

## Licence

Adapter weights: CC BY 4.0 (`LICENSE-DATA.md`); training and evaluation code: MIT (`LICENSE`). The base model's own licence (Apache-2.0 per its model card) applies to the base weights, which are not included.
