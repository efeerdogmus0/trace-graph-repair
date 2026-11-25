# TRACE adapter `pilot3_C_desc`

LoRA adapter for `Qwen/Qwen3.5-2B` (revision `15852e8c16360a2fea060d615a32b45270f8a8fc`), arm **C_desc**, seed 3407.

- Training objective: Graph-description-only fine-tuning; used as the parser of the parse-and-solve arm E.
- Produced by Kaggle kernel `trace-job13`; file `runs/kaggle/job13/runs/pilot3_C_desc/adapter/adapter_model.safetensors`
- `adapter_model.safetensors`: 67332688 bytes, sha256 `c0555ea0b4f3d723e1e70664799470167eb5ec61fc6f0fc896176e1ca5173234`

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
| tokens_in_seen | `1218972` |
| target_tokens_seen | `195996` |
| train_seconds | `1807.6000` |
| mean_loss_first_100 | `0.2387` |
| mean_loss_last_100 | `0.0042` |
| gpu | `Tesla T4` |
| peak_allocated_gib | `8.4500` |
| torch | `2.11.0+cu130` |
| platform | `Linux-6.18.55+-x86_64-with-glibc2.39` |

LoRA target modules (regex): `model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)`

## Dev evaluation

No dev evaluation of this adapter on its own (it is used inside another arm).

## Locked-test evaluation

Used as the graph parser of arm **E** (parse with this adapter, then the exact solver) in kernel `trace-jobF03`, seed 3407. The numbers are those of arm E, not of this adapter answering directly.

Accuracy of this seed per locked split, copied from `reports/final/final_table.json` (pooled-over-seed intervals and the pre-registered reading: `reports/final/final_primary.md`).

| Split | Roots | minimal_graph_repair | reachability | critical_edge | non_edit_macro_accuracy |
|---|---|---|---|---|---|
| test_id | 200 | 0.853 | 0.902 | 0.843 | 0.873 |
| test_style_ood | 200 | 0.870 | 0.943 | 0.853 | 0.897 |
| test_struct_ood | 200 | 0.655 | 0.805 | 0.635 | 0.720 |
| test_combined_ood | 100 | 0.725 | 0.860 | 0.630 | 0.745 |
| test_family_ood | 100 | 0.860 | 0.875 | 0.820 | 0.847 |
| test_size_ood | 100 | 0.700 | 0.840 | 0.655 | 0.748 |

## External evaluation (pre-registered VGCure subset)

Evaluated once as the parser of arm E on 500 VGCure graphs (1500 items), adapted protocol (our prompts, images letterboxed to 512x512, strict JSON scoring; not official VGCure numbers). Run `runs/kaggle/jobV2/pilot3_E`; record `reports/external/vgcure_final_seeds_vs_B.json`.

| Edge-existence accuracy | Reachability balanced accuracy | External macro |
|---|---|---|
| 0.745 | 0.716 | 0.730 |

## Licence

Adapter weights: CC BY 4.0 (`LICENSE-DATA.md`); training and evaluation code: MIT (`LICENSE`). The base model's own licence (Apache-2.0 per its model card) applies to the base weights, which are not included.
