# TRACE adapter `pilot3_D`

LoRA adapter for `Qwen/Qwen3.5-2B` (revision `15852e8c16360a2fea060d615a32b45270f8a8fc`), arm **D**, seed 3407.

- Training objective: Repair fine-tuning: proposes one minimal edge edit that makes the reachability goals hold.
- Produced by Kaggle kernel `trace-job20`; file `runs/kaggle/job20/runs/pilot3_D/adapter/adapter_model.safetensors`
- `adapter_model.safetensors`: 67332688 bytes, sha256 `be7c44378664bc07618ec9483ef1277893f070c68c95706a25618664bb1d9aef`

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
| tokens_in_seen | `1471362` |
| target_tokens_seen | `55296` |
| train_seconds | `2062.7000` |
| mean_loss_first_100 | `0.1767` |
| mean_loss_last_100 | `0.0349` |
| gpu | `Tesla T4` |
| peak_allocated_gib | `9.0100` |
| torch | `2.11.0+cu130` |
| platform | `Linux-6.18.55+-x86_64-with-glibc2.39` |

LoRA target modules (regex): `model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)`

## Dev evaluation

200 dev root graphs, 1200 samples, input `image`, max_new_tokens 96, truncated 0.

| Task | n | accuracy | format failures |
|---|---|---|---|
| minimal_graph_repair | 400 | 0.540 | 0.000 |
| reachability | 400 | 0.812 | 0.000 |
| critical_edge | 400 | 0.512 | 0.000 |

Non-edit macro accuracy: **0.662**.

## Locked-test evaluation

Evaluated once in kernel `trace-jobF02` (arm **D**, seed 3407).

Accuracy of this seed per locked split, copied from `reports/final/final_table.json` (pooled-over-seed intervals and the pre-registered reading: `reports/final/final_primary.md`).

| Split | Roots | minimal_graph_repair | reachability | critical_edge | non_edit_macro_accuracy |
|---|---|---|---|---|---|
| test_id | 200 | 0.500 | 0.802 | 0.490 | 0.646 |
| test_style_ood | 200 | 0.565 | 0.830 | 0.530 | 0.680 |
| test_struct_ood | 200 | 0.398 | 0.745 | 0.525 | 0.635 |
| test_combined_ood | 100 | 0.440 | 0.760 | 0.545 | 0.653 |
| test_family_ood | 100 | 0.525 | 0.795 | 0.570 | 0.682 |
| test_size_ood | 100 | 0.500 | 0.800 | 0.490 | 0.645 |

## External evaluation (pre-registered VGCure subset)

Evaluated once on 500 VGCure graphs (1500 items), adapted protocol (our prompts, images letterboxed to 512x512, strict JSON scoring; not official VGCure numbers). Run `runs/kaggle/jobV2/pilot3_D`; record `reports/external/vgcure_final_seeds_vs_B.json`.

| Edge-existence accuracy | Reachability balanced accuracy | External macro |
|---|---|---|
| 0.869 | 0.878 | 0.874 |

## Licence

Adapter weights: CC BY 4.0 (`LICENSE-DATA.md`); training and evaluation code: MIT (`LICENSE`). The base model's own licence (Apache-2.0 per its model card) applies to the base weights, which are not included.
