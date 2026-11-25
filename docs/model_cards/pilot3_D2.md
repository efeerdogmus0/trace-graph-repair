# TRACE adapter `pilot3_D2`

LoRA adapter for `Qwen/Qwen3.5-2B` (revision `15852e8c16360a2fea060d615a32b45270f8a8fc`), arm **D2**, seed 3407.

- Training objective: Repair fine-tuning plus an edit-outcome auxiliary task (revision 1, did not beat D).
- Produced by Kaggle kernel `trace-job10`; file `runs/kaggle/job10/runs/pilot3_D2/adapter/adapter_model.safetensors`
- `adapter_model.safetensors`: 67332688 bytes, sha256 `9c9be759af784fa3c9e2f8d20d05fa5e725904a97f998cfb888080ef35b9fbc2`

## Scope

Research artifact from a one-seed-per-run dev pilot on synthetic drawn directed graphs (TRACE benchmark, `manifests/pilot.json`). Not evaluated on the locked test splits or on any external benchmark yet. Not intended for any use outside reproducing the TRACE experiments; numbers below are dev-split numbers and say nothing about natural images or other graph drawings.

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
| tokens_in_seen | `1313106` |
| target_tokens_seen | `38436` |
| train_seconds | `1863.1000` |
| mean_loss_first_100 | `0.3368` |
| mean_loss_last_100 | `0.0467` |
| gpu | `Tesla T4` |
| peak_allocated_gib | `9.0000` |
| torch | `2.11.0+cu130` |
| platform | `Linux-6.18.55+-x86_64-with-glibc2.39` |

LoRA target modules (regex): `model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj|gate_proj|up_proj|down_proj)`

## Dev evaluation

200 dev root graphs, 1200 samples, input `image`, max_new_tokens 96, truncated 0.

| Task | n | accuracy | format failures |
|---|---|---|---|
| minimal_graph_repair | 400 | 0.500 | 0.000 |
| reachability | 400 | 0.770 | 0.000 |
| critical_edge | 400 | 0.495 | 0.000 |

Non-edit macro accuracy: **0.633**.

## Licence

Adapter weights: CC BY 4.0 (`LICENSE-DATA.md`); training and evaluation code: MIT (`LICENSE`). The base model's own licence (Apache-2.0 per its model card) applies to the base weights, which are not included.
