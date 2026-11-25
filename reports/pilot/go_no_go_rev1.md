# Go/no-go after revision 1 (pre-registered rule applied), 2026-11-12

Rule: `src/trace_vr/go_no_go.py`, committed in ac27048 while job11 was still running (DECISIONS
2026-11-12T16:58Z). Inputs: 200 pilot dev roots, image input, strict scoring, root-graph paired
bootstrap (2000 replicates). Output: `runs/go_no_go_rev1.json`.

| Paired non-edit macro difference | point | 95% CI |
|---|---:|---|
| machine: B(Kaggle) − B(Colab) | −0.0113 | [−0.0275, +0.0062] |
| D2(Kaggle) − B(Kaggle) | −0.1637 | [−0.1900, −0.1375] |
| D(Colab) − B(Colab) | −0.1363 | [−0.1625, −0.1088] |

Decision: **close_H1_not_supported**. The machine effect is small (interval contains 0, |point| ≤ 0.03),
both repair-trained arms are below QA-trained B on their own machine, and the closer gap (−0.136)
exceeds 0.10. Revision 2 is not spent (brief 20.1). One seed, dev split, pilot scale.

B on Kaggle (job11, torch 2.11.0): image 200 roots macro 0.796 (reachability 0.83, critical edge
0.7625, repair 0.2925; critical twin pairs both right 0.53). Colab B was 0.8075.

## B with the graph as text instead of the image (oracle "edges" mode, first 100 dev roots)

| B(Kaggle) | edges | image (same 100 roots) | edges − image [95% CI] |
|---|---:|---:|---|
| Non-edit macro | 0.750 | 0.820 | −0.070 [−0.110, −0.033] |
| Reachability | 0.825 | | −0.035 [−0.085, +0.015] |
| Critical edge | 0.675 | | −0.105 [−0.165, −0.045] |

Reading: for B the true edge list as text is *worse* than the image. B was trained on images only, so
the text format is out of distribution for it; "edges" mode is therefore not a clean upper bound with
perception removed, for B or for D (D: edges 0.6925 vs image 0.6825 on 100 roots, CI contains 0,
`reports/pilot/oracle.md`). What it does show: D's critical-edge failure is not fixed by giving the
true graph, while B's critical-edge skill is tied to the image input. A perception/reasoning split
needs arm E (parse, then exact solver) and a text-trained control, not this oracle alone.
