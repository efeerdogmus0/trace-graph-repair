# Oracle-graph diagnostic (job9, Colab T4, 2026-11-11)

Question: is arm D's gap to arm B on non-edit questions a perception problem (reading the
drawing) or a reasoning / training-signal problem? Brief section 20: "oracle graph good,
image bad" points at perception; "oracle graph also bad" points at the task or the model.

Setup: D and B retrained with the pilot3 recipe (3 epochs, 512 training roots, seed 3407;
the pilot3 adapters were lost with an earlier VM). Each adapter is evaluated on the first 100
dev roots (600 samples) with `input_mode=edges`: a blank image and the exact edge list as text
before the unchanged instruction. Image-mode numbers are the pilot3 flags restricted to the
same 100 roots (`bootstrap_ci.py --roots 100`). All intervals: paired bootstrap over root graphs.

## Arm D (read back 2026-11-11)

| | image (pilot3 adapter) | edges (job9 adapter) | edges - image |
|---|---|---|---|
| non-edit macro | 0.6825 | 0.6925 | +0.010 [-0.032, +0.050] |
| reachability | 0.86 | 0.835 | |
| critical edge | 0.505 | 0.55 | +0.045 [-0.005, +0.095] |
| repair (exact minimal) | 0.535 | 0.505 | |

Sources: `runs/pilot3_oracle/D_summary.json`, `runs/pilot3_oracle/D_edges_flags_dev.json`,
`reports/pilot/oracle/bootstrap_D_edges_vs_D_image_dev100.json`.

Reading: with perfect perception D still answers edge-criticality questions near chance. Its
gap to B (B image 0.8175 macro on the same roots) is not explained by reading the drawing.

## Arm B

Not available. job9 part 2 (B retrain + edges eval) was lost on 2026-11-12 before it was read back
(Colab runtime gone, sign-in required). To be rerun; until then no perception-vs-reasoning claim
about B, and the D result stays preliminary.

## Caveats

- One seed; 100 roots.
- Image and edges numbers come from two trainings of the same recipe, not one adapter.
- Neither adapter saw text-only graph input during training (input shift in edges mode).

## Consequence for the plan

Section 20 row "edit good, non-edit transfer absent" allows one auxiliary task (D2). D2 is
built (`arms.py`, task `edit_outcome`): same images and sample count as D, half the samples ask
whether the repair goal holds after a given legal edit. Because removal edits resemble the
critical-edge question, a D2 gain must be reported as "closer supervision", not pure transfer.
