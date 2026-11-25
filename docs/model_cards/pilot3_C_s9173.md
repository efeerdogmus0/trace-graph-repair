# TRACE adapter `pilot3_C_s9173`

LoRA adapter for `Qwen/Qwen3.5-2B` (revision `15852e8c16360a2fea060d615a32b45270f8a8fc`), arm **C**, seed 9173.

- Training objective: Structural fine-tuning (MCDGraph-inspired): graph description, edge infilling and graph matching.
- Produced by Kaggle kernel `trace-job19`; file `runs/kaggle/job19/runs/pilot3_C_s9173/adapter/adapter_model.safetensors`
- `adapter_model.safetensors`: 67332688 bytes, sha256 `4fb11f103feecda25e5f1477cde8c07bcbb6f224a4d74c123f016bb9412f1a72`

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
| seed | `9173` |
| tokens_in_seen | `1197798` |
| target_tokens_seen | `91920` |
| train_seconds | `1848.8000` |
| mean_loss_first_100 | `0.4047` |
| mean_loss_last_100 | `0.0148` |
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
| reachability | 400 | 0.710 | 0.000 |
| critical_edge | 400 | 0.380 | 0.000 |

Non-edit macro accuracy: **0.545**.

## Locked-test evaluation

Evaluated once in kernel `trace-jobF12` (arm **C**, seed 9173).

Accuracy of this seed per locked split, copied from `reports/final/final_table.json` (pooled-over-seed intervals and the pre-registered reading: `reports/final/final_primary.md`).

| Split | Roots | minimal_graph_repair | reachability | critical_edge | non_edit_macro_accuracy |
|---|---|---|---|---|---|
| test_id | 200 | 0.250 | 0.677 | 0.388 | 0.532 |
| test_style_ood | 200 | 0.250 | 0.700 | 0.367 | 0.534 |
| test_struct_ood | 200 | 0.237 | 0.650 | 0.443 | 0.546 |
| test_combined_ood | 100 | 0.240 | 0.660 | 0.390 | 0.525 |
| test_family_ood | 100 | 0.240 | 0.695 | 0.370 | 0.532 |
| test_size_ood | 100 | 0.250 | 0.705 | 0.395 | 0.550 |

## External evaluation (pre-registered VGCure subset)

Evaluated once on 500 VGCure graphs (1500 items), adapted protocol (our prompts, images letterboxed to 512x512, strict JSON scoring; not official VGCure numbers). Run `runs/kaggle/jobV4/pilot3_C_s9173`; record `reports/external/vgcure_final_seeds_vs_B.json`.

| Edge-existence accuracy | Reachability balanced accuracy | External macro |
|---|---|---|
| 0.823 | 0.642 | 0.733 |

## Licence

Adapter weights: CC BY 4.0 (`LICENSE-DATA.md`); training and evaluation code: MIT (`LICENSE`). The base model's own licence (Apache-2.0 per its model card) applies to the base weights, which are not included.
