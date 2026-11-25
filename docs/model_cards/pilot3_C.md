# TRACE adapter `pilot3_C`

LoRA adapter for `Qwen/Qwen3.5-2B` (revision `15852e8c16360a2fea060d615a32b45270f8a8fc`), arm **C**, seed 3407.

- Training objective: Structural fine-tuning (MCDGraph-inspired): graph description, edge infilling and graph matching.
- Produced by Kaggle kernel `trace-job12`; file `runs/kaggle/job12/runs/pilot3_C/adapter/adapter_model.safetensors`
- `adapter_model.safetensors`: 67332688 bytes, sha256 `a9f171f8c092e2adb4559ee68e860946bc9ef3785a4b10db7cfed2a27f9bb6b0`

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
| seed | `3407` |
| tokens_in_seen | `1197798` |
| target_tokens_seen | `91920` |
| train_seconds | `1780.2000` |
| mean_loss_first_100 | `0.3854` |
| mean_loss_last_100 | `0.0213` |
| gpu | `Tesla T4` |
| peak_allocated_gib | `8.4200` |
| torch | `2.11.0+cu130` |
| platform | `Linux-6.18.55+-x86_64-with-glibc2.39` |

LoRA target modules (regex): `model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)`

## Dev evaluation

200 dev root graphs, 1200 samples, input `image`, max_new_tokens 96, truncated 0.

| Task | n | accuracy | format failures |
|---|---|---|---|
| minimal_graph_repair | 400 | 0.255 | 0.000 |
| reachability | 400 | 0.700 | 0.000 |
| critical_edge | 400 | 0.375 | 0.000 |

Non-edit macro accuracy: **0.537**.

## Locked-test evaluation

Evaluated once in kernel `trace-jobF04` (arm **C**, seed 3407).

Accuracy of this seed per locked split, copied from `reports/final/final_table.json` (pooled-over-seed intervals and the pre-registered reading: `reports/final/final_primary.md`).

| Split | Roots | minimal_graph_repair | reachability | critical_edge | non_edit_macro_accuracy |
|---|---|---|---|---|---|
| test_id | 200 | 0.258 | 0.682 | 0.388 | 0.535 |
| test_style_ood | 200 | 0.253 | 0.700 | 0.378 | 0.539 |
| test_struct_ood | 200 | 0.250 | 0.635 | 0.405 | 0.520 |
| test_combined_ood | 100 | 0.250 | 0.615 | 0.360 | 0.487 |
| test_family_ood | 100 | 0.240 | 0.660 | 0.370 | 0.515 |
| test_size_ood | 100 | 0.250 | 0.650 | 0.365 | 0.508 |

## External evaluation (pre-registered VGCure subset)

Evaluated once on 500 VGCure graphs (1500 items), adapted protocol (our prompts, images letterboxed to 512x512, strict JSON scoring; not official VGCure numbers). Run `runs/kaggle/jobV2/pilot3_C`; record `reports/external/vgcure_final_seeds_vs_B.json`.

| Edge-existence accuracy | Reachability balanced accuracy | External macro |
|---|---|---|
| 0.830 | 0.636 | 0.733 |

## Licence

Adapter weights: CC BY 4.0 (`LICENSE-DATA.md`); training and evaluation code: MIT (`LICENSE`). The base model's own licence (Apache-2.0 per its model card) applies to the base weights, which are not included.
