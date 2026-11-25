# Announcement draft (DRAFT, not posted)

Training on minimal repair edits (arm D) did worse than the reference fine-tuning arm on the pre-registered six-cell statistic; the hypothesis is not supported at this scale.

- Difference D - B (six cells, mean of three seeds, 95% interval): -0.105 [-0.117, -0.093]; reading `negative`.
- Arm D: 0.657 [0.646, 0.667]
- Arm B: 0.762 [0.750, 0.773]
- Per-seed difference: 2027: -0.105, 3407: -0.115, 9173: -0.096
- Task mean more than 0.02 below the reference: critical_edge
- All three splits have the sign of the pooled difference: yes
- Reported E - D: +0.171 [+0.154, +0.188] (does not decide)
- Reported C - B: -0.224 [-0.239, -0.208] (does not decide)

Pilot scale: adapters were trained on 512 synthetic root graphs with one 2B base model (Qwen3.5-2B, LoRA); the locked test uses procedurally drawn graphs only. Nothing here measures real diagrams.

Rule `summary-v1`, source `reports/final/final_primary.json`. Any posting or upload is confirmed by the owner first.
