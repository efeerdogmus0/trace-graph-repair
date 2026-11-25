# TRACE adapter `pilot3_C_desc_s2027`

LoRA adapter for `Qwen/Qwen3.5-2B` (revision `15852e8c16360a2fea060d615a32b45270f8a8fc`), arm **C_desc**, seed 2027.

- Training objective: Graph-description-only fine-tuning; used as the parser of the parse-and-solve arm E.
- Produced by Kaggle kernel `trace-job21`; file `runs/kaggle/job21/runs/pilot3_C_desc_s2027/adapter/adapter_model.safetensors`
- `adapter_model.safetensors`: 67332688 bytes, sha256 `2a31a98eb335a00c28f5712118cebaf032cda9cef6e16a3cea84efdadafc3756`

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
| tokens_in_seen | `1218972` |
| target_tokens_seen | `195996` |
| train_seconds | `1807.2000` |
| mean_loss_first_100 | `0.2390` |
| mean_loss_last_100 | `0.0040` |
| gpu | `Tesla T4` |
| peak_allocated_gib | `8.4500` |
| torch | `2.11.0+cu130` |
| platform | `Linux-6.18.55+-x86_64-with-glibc2.39` |

LoRA target modules (regex): `model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)`

## Dev evaluation

No dev evaluation of this adapter on its own (it is used inside another arm).

## Locked-test evaluation

Used as the graph parser of arm **E** (parse with this adapter, then the exact solver) in kernel `trace-jobF07`, seed 2027. The numbers are those of arm E, not of this adapter answering directly.

Accuracy of this seed per locked split, copied from `reports/final/final_table.json` (pooled-over-seed intervals and the pre-registered reading: `reports/final/final_primary.md`).

| Split | Roots | minimal_graph_repair | reachability | critical_edge | non_edit_macro_accuracy |
|---|---|---|---|---|---|
| test_id | 200 | 0.838 | 0.892 | 0.840 | 0.866 |
| test_style_ood | 200 | 0.873 | 0.920 | 0.838 | 0.879 |
| test_struct_ood | 200 | 0.680 | 0.815 | 0.637 | 0.726 |
| test_combined_ood | 100 | 0.710 | 0.840 | 0.655 | 0.748 |
| test_family_ood | 100 | 0.825 | 0.870 | 0.775 | 0.823 |
| test_size_ood | 100 | 0.730 | 0.860 | 0.700 | 0.780 |

## Licence

Adapter weights: CC BY 4.0 (`LICENSE-DATA.md`); training and evaluation code: MIT (`LICENSE`). The base model's own licence (Apache-2.0 per its model card) applies to the base weights, which are not included.
