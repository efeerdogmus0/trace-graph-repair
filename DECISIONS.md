# Decisions

Dated, with evidence, the alternative considered and the reason. Newest last.

## 2026-11-09 — Graph scope for version 1
Directed simple graphs without antiparallel pairs; root graphs weakly connected; twin
graphs may be disconnected.
- Alternative: allow opposite edge pairs drawn as two curves.
- Reason: the brief allows leaving them out; every arrow stays unambiguous. Twins may be
  disconnected because otherwise tree-like graphs could never get a removal twin.

## 2026-11-09 — Own random number generator
All draws come from a SHA-256 counter stream (`rng.py`), not `random.Random` helpers.
- Evidence: Python does not guarantee identical `shuffle`/`sample` sequences across
  releases; the laptop runs 3.9, Colab runs a newer version.
- Reason: the dataset must be regenerable elsewhere from code and seed.
- Not yet verified: an actual rebuild on a second platform.

## 2026-11-09 — Answer-flipping twins for every main sample
Each repair, reachability and critical-edge sample has a twin with identical instruction
text whose image differs by one edge and whose correct answers are disjoint.
- Evidence: on the smoke set the text-only ceiling is exactly 0.50 for all three tasks
  (`reports/smoke/validation.json`).
- Alternative: unpaired samples with balanced labels. Weaker: text features could still
  correlate with the answer.
- Cost: about 4.3 images per root graph instead of 1.

## 2026-11-09 — Repair category mix 25/25/25/25 instead of 35/35/15/15
- Evidence: twins of add/remove samples are almost always `unchanged` or `impossible`
  (`scripts/quick_semantics_check.py`: add->add 12, add->remove 82 against
  add->impossible 1190 and add->unchanged 308 in 600 graphs). With full pairing the
  edited share cannot be much above 50%.
- Alternative: keep 35/35/15/15 with unpaired extra add/remove samples. Rejected because
  unpaired samples are where text-only shortcuts would live.
- The brief explicitly allows adjusting the mix if the real distribution is reported.

## 2026-11-09 — Uniqueness across all groups, not only across splits
No two graphs from different groups (root or twin) are isomorphic anywhere in a dataset.
- Reason: simpler to state and verify than cross-split uniqueness, and implies it.
- Known gap: the graph that results from applying a repair solution is not registered,
  because it is never shown as an image.

## 2026-11-10 — Force-directed layout available to every style
- Evidence (`scripts/try_layouts.py`, 20 graphs per cell): with grid/circle/layered only,
  train styles drew 0-5 of 20 `two_cluster` graphs and 7-15 of 20 `ring_chords` graphs;
  with force added, 19-20 of 20 in every cell.
- Consequence: layout algorithm is not a held-out factor of the style-OOD split.
- Alternative: keep force OOD-only and add a position optimiser for the other layouts.
  Rejected for now as more code for a secondary benefit.

## 2026-11-10 — Readability thresholds
Minimum angle between two edge ends at a node: 24 degrees (two arrowheads), 18 (head and
tail), 13 (two tails). Crossings: at most 8 per image, at least 25 degrees, kept
`arrow_len + 8` px away from arrow tips. Connected nodes far enough apart for a 14 px
shaft plus arrowhead.
- Evidence: contact sheets inspected by eye for all five styles; acceptance rates above.
- These are judgement calls on synthetic drawings; not validated with human readers.

## 2026-11-10 — Root graphs without a twin are discarded (`require_pairs: true`)
- Evidence: 1 of 104 candidate roots in the smoke build was dropped for this reason
  (3 more were dropped as duplicate topologies).
- Alternative: keep them unpaired. Rejected: breaks the 0.50 text-only ceiling.

## 2026-11-10 — Base model: Qwen/Qwen3.5-2B
- Evidence: Hub API shows the repo exists, `image-text-to-text`, Apache-2.0, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc` (checked directly). An Unsloth vision
  notebook for exactly this model exists and records a T4 run (reported by the search
  subagent, see `docs/related_work.md`).
- Alternative: `Qwen/Qwen3-VL-2B-Instruct` (also exists, also Apache-2.0). Kept as the
  fallback; GraphVerse uses it.
- Nothing has been loaded yet. Memory and speed are unknown for our inputs.

## 2026-11-10 — No project licence chosen yet
The repository is local and unpublished. The licence is the owner's choice and is left open.

## 2026-11-10 — Cross-platform rebuild checked; images are shipped or rebuilt per machine
- Evidence: pilot `content_digest` is identical (`ab499a73...`) on macOS arm64 / Python
  3.9.6 / Pillow 11.3.0, on the same Mac with Python 3.12 / Pillow 13.0.0 (smoke set), and
  on Colab Linux / Python 3.13.16 / Pillow 11.3.0. `images_digest` differs in both other
  environments, even with the same Pillow version on Linux.
- Consequence: questions, answers, graphs and geometry are reproducible from code and
  seed anywhere. Pixels are not byte-identical across machines. Every run records the
  images digest it used; all arms of one experiment must use images built on one machine.
- Closes protocol open item 6 for the pilot: rebuild on the training machine.

## 2026-11-10 — Training target text uses the instructed key order
`format_target` writes `status` before `edits` and `op, src, dst`, as the instruction
shows, regardless of how the object was stored. Found when the loss-mask check printed
the supervised text for the first time (keys had been alphabetised by the JSONL writer).

## 2026-11-11 — Per-sample results leave Colab as bit strings; intervals are computed locally
- `scripts/export_flags.py` prints strict correctness and format validity per sample in
  evaluation order, with a hash of the sample-id order and of the predictions file.
  `bootstrap.decode_flags` refuses a string made for a different sample list.
- Evidence: pilot3 D string (1,200 bits) matched a sha256 computed in the Colab page;
  its per-task sums equal the EVAL-SUMMARY accuracies (218, 336, 201 right of 400 each).
- Alternative: download `predictions_dev.jsonl`. Not done: a file download from the
  browser needs the owner, and the text would have to pass through the agent's context.
  Cost: no raw responses here, so wrong repair edits cannot be inspected locally yet.
- The VM's source tree predates `bootstrap.py`; `scripts/build_standalone_exporter.py`
  writes an equivalent single file (output compared byte for byte on synthetic answers).

## 2026-11-11 — Bootstrap details
- Unit: root graph (`graph_group_id`); 2,000 resamples; seed string "bootstrap-v1";
  percentile interval; both arms scored on the same resampled graphs for a difference
  (protocol section 5). Only accuracies and the non-edit macro have intervals so far;
  pair rates and repair sub-metrics do not.

## 2026-11-11 — job6 arm B lost; rerun B alone as job7; keep the Colab tab open while a job runs
- Evidence: at 16:48Z the notebook showed "Connect"; after connecting, `/content/trace/runs/pilot3*`
  and `/content/job6.log` did not exist and the GPU was idle, i.e. a fresh VM. D had been
  read back at ~16:00Z; B (started 15:52Z) was lost.
- Likely cause (not proven): the previous hourly run closed the browser pane, and on a Pro
  (not Pro+) account a runtime without an attached frontend is recycled. job5 was lost the
  same way. Rule from now on: a run that launches a job stays alive and keeps the tab open
  until the job's last summary is saved; jobs must therefore fit in about 70 minutes.
- job7 = the job6 launcher with the two D lines removed (same payload, sha a4e6763e5beb3b3d),
  so B uses the same code, data config and hyper-parameters as D. Images are rebuilt on a new
  Colab VM; both images digests are recorded and compared before pairing.

## 2026-11-11 — Oracle graph-text diagnostic: two evaluation-only input modes
- `image+edges`: drawing plus one line listing the true nodes and edges (sorted);
  `edges`: same line, blank white 512 px image (still 256 image tokens, checked with the
  cached processor). The instruction text is otherwise unchanged, so the only difference is
  the inserted line. `to_model_example` refuses oracle modes when a training target is asked for.
- Reason: separates "cannot read the drawing" from "cannot reason about the graph" before a
  revision is spent (RUNBOOK, prior GPT work). Alternative (text-only prompt with a rewritten
  intro) rejected: it would change two things at once.
- Note: two autopilot runs overlapped on 2026-11-11 ~16:50Z (both saw the lost VM; only one
  job7 launch happened, verified by `ps` on the VM). Duplicate budget lines removed.

## 2026-11-11 — Arm E (parse + solve) built; cleaning rules fixed before any E result
- `src/trace_vr/parse_solve.py`, `scripts/eval_parse_solve.py`: one extraction generation per
  image (same eval samples as the other arms), exact solver on the parsed graph, canonical
  JSON answer, scored against the TRUE graph with `metrics.evaluate`. `predictions_<split>.jsonl`
  has the eval_arm format, so `export_flags.py` and the bootstrap work unchanged.
- Rules: strict extraction parse for the main score (lenient reported separately, as for the
  other arms); node set = listed nodes + edge endpoints + query labels; self-loops dropped,
  duplicates merged, antiparallel pairs kept; a critical-edge question whose premise fails on
  the predicted graph scores as wrong (no fallback guess). Alternative rejected: guessing
  "false" on a failed premise would give E a prior-based answer that is not parse-and-solve.
- Default max_new 512 for extraction (16 nodes / ~20 edges need far more than 96 tokens);
  to be set from dev truncation rates (protocol open item 5).
- Solver-only sensitivity (no model, `reports/parse_solve/sensitivity_dev.json`, 200 dev roots):
  true graph → macro 1.0; one dropped edge per image → 0.8075; two → 0.685; one flipped
  direction → 0.73; two → 0.585. So E beats pilot3 D (0.671) only if extraction misses fewer
  than about two edges per image; E's result will mostly be an extraction-quality number.
- Checked end to end with a stub model (lenient path 1.0, strict path 0.0 on prose-wrapped
  JSON, flags exported); not yet run with the real model.

## 2026-11-11 — pilot3 read-out: B ahead of D; no revision before the diagnostics
- Numbers (one seed, 200 dev roots, paired bootstrap): non-edit macro D 0.671, B 0.8075,
  D - B -0.136 [-0.163, -0.109]; reachability D 0.840 = B 0.845; critical edge D 0.503 vs
  B 0.770. Pilot does not support H1 as stated.
- Decision: do not spend revision 1 yet. First the oracle-graph text condition (job8, B) and
  arm E, as RUNBOOK says, so that a revision targets the measured failure (edge criticality)
  rather than a guess. Alternative rejected: a 3-seed rerun of D/B now; the gap is ~5
  interval half-widths, so more seeds would not change the go/no-go of this configuration.
- How the flags left the VM: the FLAGS-SUMMARY line was rebuilt locally from 12 chunks and
  the format-failure positions, and its sha256 prefix (3013a5ee3414609e) matched the one
  computed in the page; the first hand copy was 3 characters off and was discarded.
- The notebook status cell did not print job8 lines (RUNBOOK said it did); appended a
  read-only job8 grep ending POLL8-END.

## 2026-11-11 — job8 (B oracle evals) lost with the VM
- Evidence: 19:52Z notebook showed "Connect"; saved status output ended with edges-mode eval at 300/600, no EVAL-SUMMARY. The browser pane was closed when this run started, the same pattern as job5/job6.
- Next: rerun as job9 (B 3-epoch retrain + both oracle modes on 100 dev roots) only in a run that can stay alive ~60 min with the tab open. Not launched in this run (usage limit).

## 2026-11-11 — job8 lost while the Mac slept
- The Mac slept ~18:05-19:50Z with the Colab tab open; the VM was recycled and job8 (B oracle
  evals) was lost before any EVAL-SUMMARY. Next attempt: one job (retrain B + edges-mode eval
  only) under ~45 min, and run `caffeinate -i` for its duration so the tab stays alive.

## 2026-11-11 — Arm C (structural SFT) built as graph description on D's images
- C = MCDGraph's "graph description" task (list all nodes and edges) on exactly the 1,024
  images D trains on (root + repair twin of each of the 512 training roots). Same sample
  count and images as B and D. Targets are longer: mean 150 characters vs 51 (D) and 16 (B)
  on the pilot train split, so C sees ~3x D's target tokens; reported, not equalised.
- Not included: MCDGraph's masked infilling and contrastive discrimination (the latter needs
  two images per sample; our pipeline is one image per sample). Description is the part that
  needs no new rendering. `tests/test_arms.py::test_arm_c_matches_d_images_and_labels`.
- Run: `train_arm.py manifests/pilot.json train C runs/pilot3_C 3 4`, not run yet.

## 2026-11-11 — How the oracle (edges) evals of job9 are compared with image mode
- job9 evaluates 100 dev roots in `edges` mode; the image-mode flags of pilot3 cover 200 roots.
  `bootstrap_ci.py --roots 100` re-expresses a 200-root flags file on the first 100 roots
  (`bootstrap.subset_flags`, original order hash checked; test `test_subset_flags_keeps_each_sample_flag`).
- Image-mode baseline on those 100 roots (`reports/pilot/oracle/bootstrap_D_vs_B_image_dev100.json`):
  non-edit macro D 0.6825 [0.6475, 0.7175], B 0.8175 [0.7825, 0.8525], D - B -0.135
  [-0.178, -0.095]; critical edge D 0.505 vs B 0.775; reachability both 0.86. Close to the
  200-root numbers, so the 100-root subset is representative.
- Caveat to report: job9 retrains D and B (same script, data and seed) because the pilot3
  adapters were lost with the VM, so image vs edges compares two trainings, not one adapter
  under two inputs. GPU nondeterminism can move numbers slightly; the image-mode eval of the
  retrained adapters is not repeated (time budget).
- Edges-mode flags leave the VM through the status cell: the predictions file is copied to
  `/content/fx/<run>_edges/predictions_dev.jsonl` and the unchanged standalone exporter runs
  with 100 roots (prints POLL9-END).

## 2026-11-11 — Oracle result for D: not a perception problem
- job9 D (retrained, train_seconds 2143.7, loss 0.177 -> 0.036) evaluated with the true edge
  list as text, 100 dev roots: macro 0.6925, critical edge 0.55, reachability 0.835, repair
  0.505 (minimal 0.39), no truncation. Versus image mode on the same roots (pilot3 D flags):
  diff +0.010 [-0.028, +0.053]; critical edge +0.045 [-0.005, +0.095].
- Reading: removing perception does not fix D. Its failure on edge criticality is in reasoning
  (or in what repair training teaches), so a revision should target the reasoning signal
  (e.g. supervision that ties criticality to path counts), not resolution or rendering.
  Preliminary: one seed, a retrained adapter, and an input format D was not trained on.
- Flags were transcribed and rebuilt from six 100-bit chunks; the rebuilt line's sha256 prefix
  df27dfb37b68d9e3 matched the page (first hand copy was 6 characters off and discarded).

## 2026-11-11 — job9 arm D oracle (edges) read back
- D retrained (3 ep, same script/data/seed; train 2143.7 s, last-100 loss 0.0356) and evaluated on
  100 dev roots with the true edge list as text instead of the image (`input_mode=edges`, 600 samples,
  0 format failures). Flags line hash-checked against the page (df27dfb37b68d9e3).
- Paired bootstrap vs the pilot3 image-mode D flags on the same 100 roots
  (`reports/pilot/oracle/bootstrap_D_edges_vs_D_image_dev100.json`): non-edit macro edges 0.6925
  [0.6625, 0.7225] vs image 0.6825, difference +0.010 [-0.032, +0.050]; critical edge 0.55 vs 0.505
  (+0.045 [-0.005, +0.095]); reachability 0.835 vs 0.86; repair 0.505 vs 0.535.
- Reading (preliminary, one seed): perfect perception does not close D's gap to B (B image 0.8175 on
  the same roots). D's weakness on critical edge looks like reasoning/training-signal, not reading the
  drawing. Caveats: the edges eval uses an adapter trained on images (input shift), and it is a
  retrained adapter, not the one that produced the image flags. B's edges eval (job9 part 2) is needed
  before any perception-vs-reasoning claim.

## 2026-11-11 — Revision 1 of 2 (brief section 20): arm D2 = D + "edit outcome" auxiliary
- Trigger: section 20 row "edit good, non-edit transfer absent" allows one auxiliary task D2.
  Evidence: D near chance on critical edge in pilot3 (0.503) and still 0.55 with the true edge
  list as text (job9 oracle), so the fix should target the training signal, not perception.
- D2 keeps D's 1,024 images and sample count. Per root group one image keeps its repair sample,
  the other gets an `edit_outcome` sample (alternating root/twin): "Goal ... Suppose you
  add/remove the edge X->Y and change nothing else. Would the goal hold afterwards?" The edit is
  a legal action of the repair query; with p=1/2 a goal-satisfying one when one exists.
- Pilot train split (512 outcome samples): answer true 196 / false 316; remove 364 (117 true),
  add 148 (79 true). Imbalanced towards false because many repair goals have no satisfying
  edit; reported, not rebalanced. Repair samples drop from 1,024 to 512.
- Honest scope: removal-edit outcome questions are close relatives of critical-edge questions
  (goal = reach constraints, edit = removal), so a D2 gain is "closer supervision", not pure
  transfer. B remains the reference; D2 must beat B, not only D.
- Test: `tests/test_arms.py::test_arm_d2_edit_outcome_same_images_balanced_and_exact` (labels
  re-derived independently with `Graph.reaches`). Run: `train_arm.py manifests/pilot.json train
  D2 runs/pilot3_D2 3 4`, not run yet.

## 2026-11-12 — job9 part B lost; training made resumable
- The 2026-11-11 21:17Z run left B retraining (EVAL-SUMMARY expected ~21:57Z) and no later run
  polled it. At 09:10Z on 2026-11-12 the browser pane had no tabs and the notebook URL redirected
  to Google sign-in, so nothing could be read; the VM is assumed recycled. Third lost job (job5,
  job8, job9-B). B's oracle (edges) result stays open; `reports/pilot/oracle.md` notes it.
  GPU time of job9-B is not added to `gpu_seconds_total` (unknown).
- Response: `src/trace_vr/resume.py` + `train_arm.py` 8th argument `ckpt_every` (samples, rounded
  up to a multiple of accum). A checkpoint (trainable params, AdamW and scheduler state, samples
  seen, log, elapsed time, token counts) is written only at an update boundary via tmp-dir +
  rename; rerunning with the same arguments resumes. The sample order is replayed by drawing
  every epoch's shuffle and skipping `seen` samples. A config mismatch (arm, manifest, split,
  epochs, accum, seed, sample count) raises instead of silently mixing runs.
- Evidence: `tests/test_resume.py` (toy model, frozen param, accum 2, 3 epochs, killed after
  sample 13 with ckpt at 12): resumed parameters and per-sample losses are bit-identical to an
  uninterrupted run on CPU. Not yet run with the real model or on GPU (CUDA kernels may be
  nondeterministic; resumed runs should be reported as such via `resumed_from_sample`).
- Checkpoints only help on Colab if `ckpt/` leaves the VM (Drive mount or read-back); decide that
  at the next launch. Runbook rule kept: poll every hourly run, no job longer than ~3 h.

## 2026-11-12 — Committed Colab launcher; no Drive mount
- Colab still required Google sign-in at 12:3xZ; CPU work only.
- Earlier jobs (job5-9) were written ad hoc into notebook cells and are not in git, so a lost job
  also lost its exact command. Now `scripts/colab/job.sh <job> <arm> <roots> <mode>` holds the full
  sequence (torchao uninstall, data regeneration with content-digest assert, resumable training with
  ckpt_every 256, eval, flags export), each stage skipped when its output exists.
  `scripts/build_payload.py` packs the code (54 files, ~86 kB gz) deterministically (gzip mtime 0,
  zeroed tar metadata). `tests/test_payload.py` checks that every script/config the job calls is in
  the archive and that two builds 1 s apart are byte-identical (first version was not: gzip header time).
- Drive is not mounted: `drive.mount` needs an interactive OAuth consent, which an unattended run
  must not grant. Checkpoints therefore protect against a crashed process on a live VM only; the
  defence against VM recycling stays "short jobs, poll every hourly run".
- job.sh has only been syntax-checked (`bash -n`) and its file references tested; it has not run on
  a VM yet. First real use will be job10 (D2).

## 2026-11-12 — Datasheet generated from the manifest; README/checklist refreshed
- Colab still signed out (~10:00Z by the Mac clock; note the previous state file said 12:40Z, so
  one of the two clocks/notes was off). CPU work toward contribution 1 of `docs/paper_plan.md`.
- `scripts/datasheet.py` renders all datasheet statistics from `manifests/<name>.json` and the
  validation report, so no number is typed by hand; `tests/test_datasheet.py` checks totals
  against the manifest and that the committed `docs/datasheet_pilot_stats.md` is current.
- Prose datasheet `docs/datasheet.md` (Gebru et al. structure) states the skews: secondary tasks
  have no twins (dev edge_exists text-only ceiling 0.895, from unique instructions, not a leak);
  critical-edge root answers 118 true / 82 false on dev; few "direct" repair goals (23/200 dev).
- README status (was "no model loaded", 2026-11-10) and RELEASE_CHECKLIST (all "not started")
  were stale; both now match PROJECT_STATE. Licence choice left to the owner.

## 2026-11-12 — Locked test set built at pilot scale, disjoint from pilot by isomorphism
- Colab still required Google sign-in (10:02Z); battery power and owner active, so CPU only.
- Sizes: test_id 200, test_style_ood 200 (styles x, y), test_struct_ood 200 (ring_chords,
  two_cluster, 11-16 nodes), test_combined_ood 100. This is the pilot-scale reduction the protocol
  allows (main plan 500/500/500/250); the study now runs at pilot scale (512 train roots), and
  200 roots per split keeps a final eval of one arm near the 200-dev-root cost. Seed 20261112.
- Separate build (`configs/data_pilot_test.yaml`) instead of extra splits in `data_pilot.yaml`:
  adding splits to the pilot config would change the pilot's combined content digest that every
  flags file and `job.sh` check. New config key `exclude_isomorphic_to: [pilot.json]` (paths
  relative to the manifest directory) preloads the topology registry with every pilot image
  graph; the manifest records the excluded manifest's content digest and the validator re-checks
  both the digest and isomorphism against it.
- Evidence: build 700 roots / 3,078 images / 7,006 samples, content digest f7c80f93…;
  duplicate_topology rejections 43 (test_id) and 46 (test_style_ood), 0 in the structure splits
  (larger, different families). `reports/pilot_test/validation.json`: 258,984 checks, 0 errors.
  Tests: exclusion bites and leaves no isomorphic graph (checked with networkx), a changed
  excluded manifest and a cloned excluded graph are both caught.
- Lock: `dataset.check_split_allowed` refuses training on any `test*` split and evaluation unless
  TRACE_UNLOCK_TEST=final-evaluation; wired into `train_arm.py` and `eval_arm.py`. No label
  statistics of the test build are reported until then (only counts and rejection reasons).
- Image bytes are machine-specific (as for pilot); the final job must regenerate on the VM and
  assert the content digest, as `job.sh` does for pilot.

## 2026-11-12 — Exposure accounting per arm (protocol open item 3 settled)
- Colab still redirected to Google sign-in (checked 10:35Z); Mac on battery; CPU only.
- `scripts/exposure.py` computes, from the manifest and the same `arms.training_samples` that
  `train_arm.py` uses, per-epoch samples, root groups, images, image tokens, and instruction /
  target characters and tokens (base tokenizer.json at the pinned revision, local cache, no
  chat template). Output `docs/exposure_pilot_train.md`; `tests/test_exposure.py` checks the
  invariants and that the committed table matches the manifest.
- Pilot train, per epoch: every arm 1,024 samples, 512 root groups, 1,024 distinct images,
  262,144 image tokens. Target tokens: D 16,384 (16.0/sample), B 5,120 (5.0), C 63,284 (61.8),
  D2 10,764 (10.5). Instruction tokens: D 195,542, B 73,216, C 64,512, D2 148,410.
- Policy: samples, groups and images are equalised; text and target tokens are not (they are
  what defines each task) and are reported next to every result. So D vs B is "same images
  budget, D reads ~2.7x more instruction tokens and writes ~3.2x more target tokens". This
  belongs in the paper's setup section and limits any claim that D wins "with equal compute".
- Correction found by the count: B does not see the same images as D. C and D2 use exactly D's
  1,024 images; B uses the same 512 root images but the twins of its non-edit pairs, so it
  shares only 546 of D's images. The arms.py docstring said "same images"; now precise.
  This does not change any reported number; it is a description fix for the paper.

## 2026-11-12 — Final-evaluation launcher; test lock in every label-reading script
- Colab still redirected to Google sign-in (10:55Z); battery; CPU only.
- `scripts/colab/final_eval.sh <job> <run> --final-evaluation` closes the gap noted with the locked
  test build: it regenerates `pilot` (needed by `exclude_isomorphic_to`) and `pilot_test`, asserts
  both content digests against the committed manifests, then for each test split evaluates the
  run's adapter in image mode on all roots (200/200/200/100) and exports flags. It refuses to run
  without the explicit third argument and without `runs/<run>/adapter`; finished splits are skipped
  so a crash can be resumed. Max new tokens 96, as in `job.sh`. Only `bash -n` and the refusal
  paths were exercised; it has not run on a VM.
- Audit found five scripts that read labels for a split argument without the lock
  (export_flags, eval_parse_solve, model_smoke, overfit_smoke [train], parse_solve_sensitivity);
  all now call `dataset.check_split_allowed`. `tests/test_payload.py` fails if a script taking a
  split from argv and loading a dataset omits the check, checks that `final_eval.sh` lists exactly
  the manifest's test splits with all roots, and that the payload holds every file both launchers
  call. The dev-only standalone exporter (for old VM trees without the lock) asserts a non-test split.
- `breakdown.py` / `bootstrap_ci.py` are not locked: they only read flags files, which exist for a
  test split only after the final run.

## 2026-11-12 — Family-only and size-only OOD splits (protocol open item 2 settled); item 6 recorded
- Colab still redirected to Google sign-in (11:15Z); battery power; CPU only.
- `test_struct_ood` changes family (ring_chords, two_cluster) and size (11-16 nodes) together, so
  a drop there cannot be attributed to either. A new locked build `pilot_diag`
  (`configs/data_pilot_diag.yaml`, seed 20261113) adds `test_family_ood` (new families, 5-10
  nodes) and `test_size_ood` (ID families, 11-16 nodes), 100 roots each. With `test_id` and
  `test_struct_ood` this is a 2x2 design. It is a separate build so the pilot and pilot_test
  content digests (checked by every flags file and launcher) do not change; it excludes
  isomorphic copies of every pilot and pilot_test graph.
- Evidence: 200 roots / 869 images / 1,978 samples in 69 s, content digest bafc07ab...;
  rejections: test_family_ood 21 duplicate topologies, 17 no-twin; test_size_ood 12 layout.
  Validator: 72,287 checks, 0 errors (`reports/pilot_diag/validation.json`).
- Reporting: the 2x2 table is a diagnostic beside the primary six-cell statistic, which stays
  as pre-registered. 100 roots per cell gives wide intervals; read as direction, not proof.
- `final_eval.sh` now regenerates pilot, pilot_test and pilot_diag, asserts all three digests and
  evaluates all six locked splits (SPLITS entries are manifest:split:roots).
  `tests/test_payload.py` checks the list against both manifests and the exclusion digests.
- Item 6 written into the protocol as already practised: regenerate, assert content digest,
  image bytes machine-specific.

## 2026-11-12 — Topology capacity per node count; main-build node range (protocol item 7 settled)
- Colab still redirected to Google sign-in (checked 11:35Z); battery power; CPU only.
- `scripts/topology_capacity.py` draws 4,000 graphs per (family, n) from the real generators
  (independent seeds, no data build) and counts directed isomorphism classes; output
  `docs/topology_capacity.md` (59 s), test `tests/test_topology_capacity.py`.
- Evidence: five-node cells saturate. Distinct classes after 4,000 draws: layered_dag 98
  (Chao1 100.6), tree_plus 255, sparse_random 422, ring_chords 132, two_cluster 90. This
  explains the pilot rejections (train: 43 duplicate topologies, mostly n=5) and test_id
  layered_dag n=5 accepting 0 of 13 (the pilot had already used the space).
- Duplicate rate while filling a cell from 360 to 400 distinct roots (the per-cell need of the
  brief's main build: ~6,500 ID roots over 3 families x 6 sizes): n=6 layered 0.62, tree 0.47,
  ring 0.77, two_cluster 0.35; n=7 layered 0.27, tree 0.36, sparse 0.00; n>=9 below 0.20 in all
  ID families. A high rate means the accepted roots are a novelty-biased slice of the
  generator's distribution, not a sample from it.
- Decision: main build ID splits use nodes [7, 12], size-OOD [13, 18], with a fixed quota per
  node count (node count kept when a candidate is rejected). 7 is the smallest size where every
  ID family reaches 400 distinct roots at <=36% duplicate rate. Existing pilot builds unchanged.
- Not yet checked: layout success and readability of 13-18-node drawings at 512 px (pilot
  test_size_ood at 11-16 already had 12 layout rejections of 100 roots), and the quota code path.
  Both before the main build; the quota must be a new config key so existing digests hold.
- Reading for the paper: the five-node cells are small enough that train can cover a large
  part of every five-node topology; any result on n=5 in the pilot builds is closer to
  in-distribution recall than to generalisation. Reported as a datasheet limitation.

## 2026-11-12 — Per-size node quota (config key) and 7-18-node layout acceptance
- Colab still redirected to Google sign-in at the last check; battery power; CPU only.
- New split key `node_quota` in `dataset.py` (`random` default, `equal`): with `equal` root idx gets
  lo + idx mod (hi-lo+1) nodes, independent of the attempt, so a rejected candidate is retried at the
  same size. The default path calls the same `rng.child("n")` draw as before; evidence: smoke rebuild
  content digest c26d05b4... identical to `manifests/smoke.json`; tests `test_node_quota_*`,
  `test_node_count_for_default_matches_the_original_draw`, `test_unknown_node_quota_is_rejected`.
- `scripts/layout_capacity.py` (test `tests/test_layout_capacity.py`) ran 40 graphs per
  (family, n, style) for all five families, n 7-18, styles a/b/c (947 s): `docs/layout_capacity.md`.
  Pooled acceptance at n=18: tree_plus 0.99, ring_chords 0.90, two_cluster 0.68, layered_dag 0.61,
  sparse_random 0.40; at n<=12 every family >=0.92. 0 pixel problems in 6,546 rendered drawings.
- Ad-hoc check on the same seeds: accepted 18-node roots have fewer edges than rejected ones
  (sparse_random 24.35 vs 28.72; layered_dag 22.49 vs 26.02). Layout rejection therefore selects
  sparser graphs at large n; size-OOD would also shift density.
- Decision: keep [13, 18] provisionally. Before the main build, try a larger `find_layout` budget for
  n>=15 (new config key, digests unchanged) and re-measure; if sparse_random at n=18 stays below ~0.6
  or the edge gap stays above ~2 edges, cap size-OOD at 16 or report edges-per-root and a
  density-matched size slice. Either way the datasheet reports edges per root per split.
- Readability for the model (not just pixel checks) is still unverified at 13-18 nodes.

## 2026-11-12 — Arm C = three MCDGraph-inspired structural tasks (protocol item 4 settled)
- Owner was active (idle 366 s) on AC power; CPU only. Colab not re-checked. A layout-budget scan
  started by the previous run (`scripts/layout_budget.py`, uncommitted, 180 vs 720 budget, n 15-18)
  was still running in the background and was left alone; its partial first row: sparse_random
  n=15 acceptance 0.65 (budget 180) vs 0.73 (720), edges acc/rej 21.55/24.83 vs 21.86/25.00.
- The brief (10.2) asks for all three MCDGraph tasks with a roughly balanced mix; the 2026-11-11 C
  had description only. C now cycles description / masked infilling / graph matching over D's
  1,024 image slots (342/341/341 on pilot train). Infilling withholds max(1, |E|//3) edges (mean
  2.93, 1-6 on pilot train). Matching is a one-image stand-in for contrastive discrimination;
  negatives are the twin's edge list (one edit away; 171 true of 341).
- Old C kept as `C_desc` (ablation). Exposure regenerated: C target tokens 28,592 (was 63,284 as
  description only), instruction tokens 92,146; still 1,024 samples / 512 groups / 1,024 images.
- C has not been trained. Tests: 84 passed (`.venv-model`, working tree incl. the uncommitted
  layout-budget test). New tests `test_arm_c_three_structural_tasks_on_d_images`,
  `test_infill_and_match_parsing_is_strict`.

## 2026-11-12 — Layout budget does not fix large-graph selection bias; size-OOD capped at [13, 16]
- Owner active (idle 0 s) on AC power; CPU only. Colab checked 12:56Z: still redirects to Google
  sign-in. The background scan (started before the repo move, ran in the old scratch copy) finished
  in 2,139 s: `docs/layout_budget.md` (script and `scan_cell(..., budget=)` were committed with the move).
  A parallel stale run also committed this result in the old copy (b672814); this entry is the
  record in the moved repo.
- Evidence (120 graphs per cell, styles a/b/c pooled, same seeds): n=18 sparse_random 0.40 -> 0.47,
  edge gap 4.36 -> 4.85; n=18 layered_dag 0.61 -> 0.71, gap 3.53 -> 3.69; n=15 sparse_random
  0.65 -> 0.73, gap 3.28 -> 3.14.
- Against the rule written earlier today (sparse_random n=18 >= ~0.6 and gap <= ~2 edges): both fail.
- Decision: keep budget 180, no new config key (existing digests untouched). Main-build size-OOD
  [13, 16] (pooled acceptance at budget 180 in `docs/layout_capacity.md`: sparse_random n=16 0.59,
  layered_dag 0.71, others >= 0.85). The gap exists from n=15 on, so every split reports edges per
  root and size-OOD is also scored on a slice matched to ID edges-per-node (script not yet written).

## 2026-11-12 — Edges-per-node-matched slice implemented (protocol item 7 follow-up)
- Colab checked ~13:00Z: still redirects to Google sign-in; CPU only (nice 19).
- `analysis.root_density / quantile / density_window / density_matched`, `scripts/density_slice.py`,
  `tests/test_density_slice.py`. Density = edges/nodes of the root drawing (metadata, not labels);
  window = nearest-rank 5-95% quantiles of the reference roots (default pilot train). Count-only mode
  reads no labels; flags mode calls `check_split_allowed` (test splits stay locked).
- Evidence (`reports/density/`): pilot train 512 roots, mean 1.288, window [0.875, 1.667].
  pilot_diag test_size_ood 100 roots, mean 1.209, matched 98 (0 below, 2 above);
  test_family_ood 99/100; pilot_test test_struct_ood 199/200; test_id 187/200; pilot dev 183/200.
  pilot3 dev on the slice: D non-edit macro 0.671 (all) vs 0.673 (matched); B 0.807 vs 0.810.
- Reading: at 11-16 nodes the density shift is small (about -0.08 edges/node) and a window slice
  removes almost nothing, so it cannot correct a within-window shift. Decision: keep the window slice
  as the declared check; if the main build's size-OOD (13-16) mean density differs from ID by more
  than ~0.1 edges/node, add a density-binned reweighting of ID accuracy as a second, declared check.

## 2026-11-12 — Token-limit procedure and preamble rule fixed before measurement (protocol items 1, 5)
- Colab checked ~13:30Z: still redirects to Google sign-in; CPU only (nice 19).
- Item 1: primary score stays strict; prose/reasoning preambles are format failures; the lenient
  last-JSON diagnostic is reported next to it for every arm. No `<think>` stripping: the frozen base's
  MPS smoke outputs (24 dev samples, `runs/a0_smoke_mps/predictions.jsonl`) contain plain prose only.
  Fixed now, before any long-limit base output exists, so the rule cannot be tuned to results.
- Item 5: greedy decoding is deterministic, so one dev run at 768 new tokens with token ids saved
  gives every shorter limit as a prefix. Candidates 96/192/384/768; per arm, smallest with every task's
  truncation <= 2%, else 768 with residual truncation reported. `model.generate` now returns `ids`,
  `eval_arm.py` saves `ids` and `hit_limit`; `token_budget.check_prefix_consistency` compares a real
  shorter run with the prefixes when both exist. Trained pilot3 B and D: 0 truncated at 96 (200 roots),
  so 96 stays for trained repair/QA arms.
- New GPU job (not run): `scripts/colab/base_budget.sh job12 10 768` (frozen base, 10 dev roots,
  then TOKEN-BUDGET + FLAGS-SUMMARY). Expected cost unknown; MPS smoke was 11.7 s/sample at <=192
  tokens and T4 5.73 s/sample at 96, so 10 roots is kept small on purpose.
- Tests: `tests/test_token_budget.py` (3), payload test now covers `base_budget.sh`.

## 2026-11-12 — Paper draft started with the benchmark section only
- Colab checked again this run: still redirects to Google sign-in; CPU only (nice 19).
- Contribution 1 (the benchmark) is the part that exists regardless of model results, so its
  section is drafted now: `paper/sections/benchmark.tex`. Every number is copied from generated files
  (`docs/datasheet_pilot_stats.md`, `reports/pilot*/validation.json`); results/abstract stay empty
  until the main runs exist. Pilot model numbers are deliberately not in the draft.
- Figure 1 is generated, not drawn: `scripts/paper_figure.py manifests/pilot.json dev paper/figures/fig1`
  picks the first dev group with a critical-edge twin and an edited repair on the root, writes the PNG
  and a JSON with the exact question/answers/differing edge. The test recomputes both critical-edge
  answers and the repair goals with `trace_vr.reference` (independent of the label solver) and checks
  the script refuses locked test splits. Dev only; no test labels read.

## 2026-11-12 — Paper Setup section drafted, numbers tested against code
- Colab checked ~14:00Z: still redirects to Google sign-in; CPU only (nice 19).
- `paper/sections/setup.tex` states the protocol as fixed now (H1/H0, Qwen3.5-2B@15852e8c, LoRA r=16
  alpha=16 on language-model projections only, AdamW 2e-4, 3% warm-up + linear decay, clip 1.0, batch 1 x
  accum 4, seeds 3407/2027/9173, arms A-E + D2 caveat, exposure table, strict/lenient scoring, token-limit
  rule, primary statistic and interpretation thresholds). It contains no model result.
- `tests/test_paper_setup.py` fails if a training constant, the base revision, the token-budget candidates,
  the bootstrap resample count or any exposure-table number drifts from the code / generated doc.
- C is called "inspired by MCDGraph (not a reproduction)" in the paper too. The bibliography holds only the
  identifiers verified in `docs/related_work.md`; author lists are left TODO rather than written from memory.

## 2026-11-12 — Paper "What did not work" section; oracle interval corrected
- Colab checked this run: still redirects to Google sign-in; CPU only (nice 19).
- `paper/sections/negative.tex` reports pilot observations (one seed, dev) explicitly as not results;
  `tests/test_paper_negative.py` ties each number to `runs/pilot3_*/bootstrap|breakdown`, the oracle
  bootstrap file, `runs/pilot_A/summary.json` and `docs/layout_budget.md` (half-up rounding).
- Correction: the 2026-11-11 entry above quotes the oracle macro difference as +0.010 [-0.028, +0.053].
  `reports/pilot/oracle/bootstrap_D_edges_vs_D_image_dev100.json` says [-0.0325, +0.050], and a local
  re-run of `scripts/bootstrap_ci.py ... --roots 100` reproduces that file exactly. The earlier text was a
  transcription error; `docs/negative_results.md` and `PROJECT_STATE.md` are corrected, the old entry is
  left as written (history not rewritten). Reading unchanged: no gain from the text oracle.

## 2026-11-12 — Invariance pairs get their own eval run; Analysis-plan section
- Colab checked this run: still redirects to Google sign-in; CPU only (nice 19).
- Found while drafting the analysis plan: `arms.eval_samples` keeps only root/edit images, and every
  eval/flags/bootstrap script uses it, so the relabel/restyle copies were never scored and the
  invariance-pair accuracy promised in protocol section 5 could not be computed.
- Decision: keep `eval_samples` unchanged (existing flags files, order hashes and bootstrap code stay
  valid) and score the copies as a separate sample set (`sample_set=invariance`, file suffix `_inv`),
  paired with the main run afterwards by `analysis.invariance_table` (both right, root right, copy right,
  same correctness). `final_eval.sh` runs it for every locked split: 884 copies vs 5,400 main samples.
  The standalone exporter (old VM trees) refuses the invariance set.
- Not decided: whether dev comparisons also run the invariance set (212 copies on pilot dev, ~18%
  of main eval time); default no for pilot jobs, yes for the final evaluation.
- `paper/sections/analysis.tex` fixes the secondary analyses before any main run; only the primary
  statistic decides H1; secondary quantities are not multiplicity-corrected and carry no claim alone.


## 2026-11-12 — Kaggle results are collected from files, checked against the log
- `scripts/kaggle/collect.py <job> <arm> [--budget]` replaces hand transcription for Kaggle runs:
  summary.json is built from the downloaded `train_summary.json` / `metrics_*.json`, and the flags file is
  the log's FLAGS-SUMMARY only if a local re-score of the downloaded predictions gives the identical line
  (predictions sha256, digests, bits). `tests/test_kaggle_collect.py` covers the match and a tampered log.
- `runs/kaggle/` is untracked (adapters are tens of MB); only the small collected files are committed.
- The account also holds a private kernel `visual-skill-composition-e0` (2026-09-06, CPU, temporal action
  windows from a dataset `visual-skill-e2b-recovery-20260906`). It belongs to an earlier, separate
  project; noted in `docs/prior_gpt_work.md`, not used here.

## 2026-11-12 — Introduction written before results
- `paper/sections/intro.tex` states the question, the twin design and four contributions phrased as what
  the paper provides; the findings paragraph is a TODO until M5/M6. `tests/test_paper_intro.py` forbids any
  number except the 0.50 image-blind bound and result phrasing ("we show", "outperform", ...), so the intro
  cannot silently absorb pilot numbers. job10 (D2) still running on Kaggle during this run.

## 2026-11-12T15:56Z Limitations section written before results
The paper's Limitations section is drafted now from protocol/datasheet facts only, so that it is not
shaped by the results. Its compute paragraph says "single T4 GPUs with no extra spending" rather than
"free quota", because early runs used existing units of the owner's paid Colab Pro account.

## 2026-11-12T16:18Z — Revision 1 (D2) read back: no rescue; next is B on the same machine
- D2 (job10, Kaggle) non-edit macro 0.6325 vs D 0.671 and B 0.8075 on 200 dev roots; critical edge
  collapsed to "false" (said true 1%). Recorded as negative result 5 (preliminary). `reports/pilot/d2.md`.
- D2 was the first arm trained on Kaggle (torch 2.11.0+cu130), D and B on Colab. Before any decision
  about revision 2, job11 retrains B on Kaggle with the same recipe and evaluates 200 dev roots (image)
  and 100 roots (oracle edges, finishing the lost job9-B part). Kaggle B vs Colab B (0.8075) measures
  how large a pure machine change is; D2 vs Kaggle-B is then a same-machine comparison.
- `collect.py` no longer requires the kernel's images digest to equal the Mac's: image bytes differ by
  machine (engineering lesson 5); the content digest, predictions hash and every bit must still match,
  and both images digests are stored in summary.json.
- Kaggle output download: fetch without the `ckpt/` optimizer state (`--file-pattern`), ~70 MB instead
  of ~400 MB; an earlier full download stopped partway.

## 2026-11-12T16:37Z: reproducibility appendix
The paper states the content digest (not image bytes) as the reproducibility contract, matching what the
launchers enforce; digests are shown as 16-hex prefixes with full values in the manifests. Placed as an
appendix so the main text stays within a workshop page limit.

## 2026-11-12T16:58Z: go/no-go after revision 1 pre-registered before job11 is read back
- Committed while job11 (B on Kaggle) was still RUNNING, so the rule cannot be shaped by its result.
- Machine effect = paired B(Kaggle) - B(Colab) non-edit macro on the 200 dev roots. Small iff the 95%
  interval contains 0 and |point| <= 0.03 (about the D2-D gap of 0.039; a machine shift that size would
  make revision 1's verdict uncertain). If not small: retrain D on Kaggle before deciding anything.
- If small: any D arm whose paired interval vs same-machine B is above 0 -> H1 supported on dev.
  Both below with the closer point gap still worse than -0.10 -> record H1 not supported and take the
  negative-result path (brief 20.1); revision 2 is not spent, because one targeted change closing a
  >0.10 macro gap is implausible and spending GPU on it would be result-hunting. Otherwise revision 2,
  with success fixed now as D_rev2 - B (same machine) interval above 0; anything else closes H1.
- Thresholds 0.03 / 0.10 are pinned by `tests/test_go_no_go.py`; changing them later voids this entry.
- A dry run with B(Colab) standing in for B(Kaggle) (machine effect 0 by construction) only checked the
  script; it is not a result.

## 2026-11-12T17:40Z: pre-registered rule applied: H1 closed as not supported (pilot)
- machine B(Kaggle)−B(Colab) −0.011 [−0.028, +0.006] (small); D2−B(Kaggle) −0.164 [−0.190, −0.138];
  D−B(Colab) −0.136 [−0.163, −0.109]. Decision close_H1_not_supported; revision 2 not spent.
  `reports/pilot/go_no_go_rev1.md`, `runs/go_no_go_rev1.json`.
- Next, negative-result path (brief 20.1) with the paper plan's contributions 1, 3, 4 and the specific
  transfer finding (reachability transfers, critical edge does not): strong baselines first (arm C at
  matched exposure on Kaggle, then C_desc + arm E parse-and-solve), then seeds for B/C/D, then the locked
  test/OOD evaluation once. No new base models (brief 20.1: no result-hunting).
- The oracle "edges" mode is out of distribution for image-trained adapters (B loses 0.07 with it);
  it is reported as a diagnostic, not as a perception-free upper bound.

## 2026-11-12T18:50Z: seed runs wired, seeds taken from the protocol only
- `job.sh` takes `TRACE_SEED` (default 3407), `make_kernel.py --seed N`, `collect.py --seed N`. Only the
  seeds pre-registered in `docs/protocol.md` (3407, 2027, 9173) are accepted; anything else exits before
  any work, so a seed cannot be added after seeing results. Seed 3407 keeps the old run name
  (`runs/pilot3_<arm>`, existing results unchanged); others use `runs/pilot3_<arm>_s<seed>`.
- Arm E (job_e.sh) has no seed option: it is a parse-and-solve arm whose trained part is C_desc; seeds for
  it are not in the paper plan.
- `tests/test_seeds.py` executes the seed/run-name lines of job.sh itself and checks the three lists agree.

## 2026-11-12T19:05Z: results table only from flag files
- `scripts/results_table.py` builds the paper's dev results table from `flags_*.json`; every arm must share
  split, roots, sample set and dataset build, and missing responses abort. Paired differences use the same
  bootstrap seed as go_no_go, so the table reproduces `runs/go_no_go_rev1.json` exactly (tested).
- Reference arm for the dev table is B_kaggle, because C, E and D2 run on Kaggle. Cross-machine rows keep the
  machine in the label (D_colab); the machine effect (B_colab - B_kaggle +0.011 [-0.006, +0.028]) is shown
  as its own row so readers can judge it.

## 2026-11-12T19:20Z: arm C read back
- C (three structural tasks, matched samples/roots/images, Kaggle) is the weakest trained arm on the non-edit
  tasks: macro 0.5375, below D2 by 0.095 [0.069, 0.123] on the same machine. Recorded as a baseline result,
  not as a revision; H1 (D vs B) stays closed. budget.jsonl got the C line twice (cause not found; no second
  collect process was running afterwards); the duplicate was removed so the line appears once.

## 2026-11-12T19:16Z: arm E collection reads the parser's train summary
- job_e.sh writes C_desc training to `runs/pilot3_C_desc` and the parse-and-solve eval to `runs/pilot3_E`.
  `collect.py <job> E` now takes train_summary.json from the parser dir (`TRAIN_DIR` map) so E's budget line
  counts C_desc training + E eval; summary.json records `train_dir`. Tested with a fake kernel output.

## 2026-11-12T19:40Z: Kaggle GPU queue
- Remaining dev GPU work is a fixed, committed order (`scripts/kaggle/queue.json`): E, B/D/C at seed 2027,
  B/D/C at 9173, then D at seed 3407 retrained on Kaggle. Order interleaves arms so that if the weekly quota
  runs out, each finished seed still has B and D on the same machine.
- D at seed 3407 is retrained on Kaggle (`runs/pilot3_D_kaggle`) because the existing pilot3_D ran on Colab;
  the three-seed D-B comparison then has no cross-machine term. The Colab D stays as the go/no-go record.
- An entry counts as done only when collect.py has written its summary.json, so a kernel that finished but
  was not read back is not skipped.

## 2026-11-12T19:55Z: seed pooling and seed reading fixed before the second seed
- Pooled arm statistic = mean over its seeds of the per-seed statistic, all seeds on the same resampled root
  graphs (same bootstrap stream as cluster_bootstrap). Pooled A - R needs no cross-arm seed pairing. The
  interval covers root sampling only; seed spread (per-seed points, min/max) is reported beside it.
- Seed-matched differences pair runs with the same training seed. Reading on the non-edit macro:
  positive_clear = pooled interval above 0, every seed-matched point > 0, pooled point >= 0.05;
  positive = same without the 0.05; negative = pooled interval below 0 and every seed-matched point < 0;
  otherwise inconclusive. Any positive reading with fewer than the three pre-registered seeds is
  `incomplete_seeds`. Implements protocol section 5 ("a positive reading must hold in each seed").

## 2026-11-12T20:25Z: Results section holds the pilot dev table
- With H1 closed at pilot scale, the paper's Results section now shows the generated dev table and the
  go/no-go numbers instead of an empty stub, labelled as one-seed development-split pilot numbers. It is to be
  replaced (not extended) when seeds and the locked test exist. Every number is checked by a test against the
  generated JSON, so the section cannot drift from the table.
- The generated LaTeX table must contain no bare underscore (found one in the header; fixed in the generator).

## 2026-11-12T20:40Z: perception split pre-registered before E read-back
- "Readable" image = arm E's strict extraction is exact (node set and directed edge set). Each arm's non-edit macro
  is reported on readable vs unreadable samples, each with its paired difference to the reference arm (B_kaggle).
- Reading for arm X vs ref: reasoning_gap if the readable-subset difference interval is entirely below 0;
  perception_gap if it contains 0 while the unreadable-subset interval is entirely below 0; no_gap if neither is
  below 0; unclear otherwise; too_few if either subset has < 30 root graphs.
- Caveat fixed in the module docstring: readability is measured by E's parser (C_desc adapter), a proxy for how
  hard a drawing is, not what the other arm perceived. Complements the oracle-text diagnostic; does not replace it.

## 2026-11-12T20:58Z: arm E read back; Kaggle artifacts hashed
- E (C_desc parser + exact solver) is the best arm on dev (macro 0.884, +0.088 over B). It is reported as a pipeline
  baseline with an external exact solver, not as a competitor on equal terms with the end-to-end arms.
- The pre-registered perception split reads reasoning_gap for D, D2 and C: the D-B deficit persists on images E
  transcribes exactly. The paper's Results section states this with the proxy caveat.
- runs/kaggle/ stays untracked (adapters ~67 MB each); `runs/kaggle_artifacts.json` records sha256/bytes of every
  downloaded file (ckpt/ skipped) and ties each predictions file to the tracked flags file with the same hash.
  `runs/pilot3_E/extraction_dev.jsonl` (218 KB) is tracked so the perception split can be rerun without Kaggle.
- Seed queue continues with job14 (B seed 2027).

## 2026-11-12T21:17Z: Kaggle quota is checked from session wall time before each push
- Kaggle bills GPU quota by session wall time (setup, pip pins, data generation included). budget.jsonl's gpu_seconds
  (train + eval) undercount it: job10-13 sum to 15,797 s there but 17,164 s of session time in the kernel logs.
- `scripts/kaggle/quota.py` counts sessions started in the trailing 7 days (conservative; the CLI does not expose the
  weekly reset), assumes a 30 h quota (not read from Kaggle) and keeps a 2 h margin. A push happens only on `ok`;
  on `wait` the autopilot does CPU work and records it in PROJECT_STATE.

## 2026-11-12T21:45Z: negative-results log and paper section brought up to date with C, D2, E
- `docs/negative_results.md` and `paper/sections/negative.tex` still listed arms C and E as "never run" after job12/job13.
  Fixed: new log entry 6 (C below B; perception split reasoning_gap), paper paragraphs for revision 1 (D2 vs same-machine
  B_kaggle -0.164) and structural C (-0.259) plus the readable-subset D-B -0.148. D2 is quoted against B_kaggle (same
  machine), not the Colab B (-0.175 in d2.md), to match the pre-registered go/no-go comparison.
- `tests/test_paper_negative.py` now fails if C or E reappear in either never-run list and checks every new number
  against `results_dev.json`, `perception_split_dev.json` and the D2 breakdown.

## 2026-11-12T23:25Z: B seed 2027 collected; job15 pushed before the interim seed table
- B at seed 2027 (0.805) and seed 3407 on Kaggle (0.796) differ by 0.009 macro, inside either seed's interval;
  `reports/pilot/seeds_dev_interim.md` pools them (0.801 [0.778, 0.824]). Interim only: the pre-registered reading
  needs all three seeds per arm; the D/C rows will be filled as job15-20 come back.
- job15 was pushed right after job14 completed (one kernel at a time holds) to keep the GPU queue moving; the
  artifacts download and collection happened while job15 was queued.
- `kaggle kernels output` can time out on kaggleusercontent.com and silently skip files; check that
  train_summary.json exists and re-fetch with `--file-pattern` before running collect.py.

## 2026-11-13T00:05Z: final locked-test evaluation runs on Kaggle from the training kernels' adapters
- Adapters live only in Kaggle kernel outputs (and untracked runs/kaggle/). Re-uploading 67 MB adapters from the Mac
  would cost metered bandwidth; instead the final kernel attaches the source kernel (`kernel_sources`) and copies the
  adapter after checking its sha256 against runs/kaggle_artifacts.json, so the scored adapter is the dev-reported one.
- The builder refuses without `--final-evaluation`; the locked splits are still scored once, after the seed queue.
- Arm E (parse-and-solve) is not covered by final_eval.sh (it evaluates an adapter end-to-end); a final E path is still
  needed before M6.

## 2026-11-13T11:18Z: arm E gets the same locked-test evaluation as the adapter arms
- The 00:05Z entry left E without a final path. `final_eval.sh` now takes `--parse-solve=<parser_run>` as a fourth
  argument: same splits, digests and skip-if-done logic, but `eval_parse_solve.py` with the parser adapter
  `runs/<parser_run>/adapter` and max_new 512 (job_e.sh's dev setting, so test and dev use one recipe).
- E is also scored on the invariance copies (`sample_set invariance`), so the pair-consistency table covers every arm.
  Default file names for the main run are unchanged (run_suffix), so job13's dev files stay valid.
- The Kaggle builder checks the sha256 of the parser adapter (job13 `pilot3_C_desc`), not of `pilot3_E` (which has no
  adapter). Only seed 3407 of E exists; whether E gets seeds 2027/9173 (parser retrain) is open and decided before
  the final evaluation.

## 2026-11-13T11:40Z: arm E is run at all three pre-registered seeds
- The 11:18Z entry left open whether E gets seeds 2027/9173. Decided yes: E is the strongest dev arm (+0.088 over B at
  seed 3407) and the paper's baseline comparison should not rest on one parser seed while B/C/D are seed-pooled. The
  seed changes the C_desc parser's training only; the solver is exact, so this measures parser variance.
- Cost: whole remaining queue ~15.1 h of session time (quota.py), 7.54 h used of an assumed 30 h -> fits this week with
  the 2 h margin. E jobs go last (job21, job22) so the B/C/D seed table completes first.
- Directory rule: seed 3407 keeps `pilot3_C_desc` / `pilot3_E` (job13 files and the parser sha256 in
  runs/kaggle_artifacts.json refer to them); other seeds append `_s<seed>` to both, same rule as job.sh.

## 2026-11-13T12:00Z: B seed set complete
- B now has all three pre-registered seeds (3407 Kaggle 0.796, 2027 0.805, 9173 0.792; pooled 0.798 [0.776, 0.822]).
  The B reference in the interim seed table is final for dev; C/D/E comparisons are taken against this pooled B.

## 2026-11-13T14:20Z: D at seed 9173 reproduces seed 2027
- D seeds 2027 and 9173 both score 0.669 dev macro (per-task differences <= 0.015). The D-B gap is stable across seeds
  (-0.136, -0.124); the H1-negative pilot reading does not depend on the seed. D's third seed (3407) is the Colab run,
  cross-machine, so job20 retrains D at 3407 on Kaggle before the table is final.
- job19 (C seed 9173) pushed right after collection (one kernel at a time holds; quota ok).

## 2026-11-13T14:30Z: VGCure external subset pre-registered (adapted protocol)
- VGCure test records carry question/claim strings and answers but no edge list, so new queries cannot be generated from
  the graphs. The subset uses only items whose answer the file fixes: neighbour-claim pairs -> edge_exists (balanced),
  connective_path Yes/No -> reachability (82% yes; balanced accuracy). Critical-edge and repair cannot be built from it.
- All 10 directed sources, first 50 graphs per source in numeric image-id order; no model output consulted. Our own
  instruction templates are used (the adapters answer JSON), so results are labelled "pre-selected VGCure subset, adapted
  prompts". VGCure's official prompts/scorer stay a possible secondary row.
- No licence is stated for VGCure: only ids, hashes and the subset digest are committed; raw files stay in gitignored
  `external/` or are fetched on Kaggle. Evaluated once, after the locked test, so no dev decision can depend on it.
- 15 of the 500 false edge items are reversed existing edges (VGCure's fc_ref flips direction); kept, reported as a slice.

## 2026-11-13T18:00Z: job19 collected, job20 pushed
- C seed 9173 dev macro 0.545; C now has all three pre-registered seeds: pooled -0.249 [-0.277, -0.222] vs B, every
  seed-matched gap between -0.241 and -0.259. Structural-SFT C is a solid negative on dev (pilot scale).
- job20 (D seed 3407 retrained on Kaggle, out pilot3_D_kaggle) pushed right after collection; quota ok (10.37 h of 30 h).

## 2026-11-13T18:05Z: VGCure evaluation path and image preprocessing
- VGCure images are letterboxed to 512x512 (longest side 512, aspect kept, LANCZOS, white pad) before any model sees
  them. Reason: every adapter was trained and evaluated on 512x512 inputs; native resolution would change the visual
  token count and the T4 cost. Fixed before any VGCure output; the original size range is printed and reported.
- One Kaggle kernel evaluates all chosen arms (plus optional frozen base) on the 1,500 items; prepare refuses unless the
  rebuilt subset digest and source-file hashes equal reports/external/vgcure_subset.json, and every zip matches its LFS
  sha256. The builder refuses without --after-final-evaluation (VGCure runs once, after the locked test).

## 2026-11-13T18:12Z: reference baseline is B (chosen on dev, before the locked test)
- Rule coded before applying it (`seeds.choose_reference`): candidates B and C must each have all three pre-registered
  seeds on dev; the higher pooled non-edit macro point wins; exact tie keeps B. Applied to the seed table once B and C were
  complete (D's missing seed does not enter the choice).
- Result: B 0.798 (0.805 / 0.796 / 0.792) vs C 0.549 (0.564 / 0.537 / 0.545); C - B pooled -0.249 [-0.277, -0.222].
  Reference = B for the locked test (D, E compared against B; C still reported). `reports/pilot/reference_baseline.{md,json}`.

## 2026-11-13T19:32Z: D seed set complete; B/C/D dev seed table final
- job20 retrained D at seed 3407 on Kaggle: 0.662 (Colab run at the same seed 0.671; cross-machine difference 0.009, inside
  the seed spread). The table uses the Kaggle run so all nine B/C/D runs share one machine.
- D pooled 0.667 [0.647, 0.686], D - B -0.131 [-0.153, -0.110]; all three seed-matched gaps -0.124 to -0.136. The H1 transfer
  claim is not supported on dev at pilot scale, across seeds. Only E (job21, job22) remains before the locked test.

## 2026-11-13T19:33Z: seed-pooled table enters the paper
- The paper's results section now carries the B/C/D three-seed table (generated by `seed_table.py --out-tex`, test asserts
  byte equality with the generator). The single-seed table stays: it is the only place D2, E and the cross-machine B
  comparison appear. Section title drops "one seed". E joins the seed table after job22; the paragraph says E seeds are
  still running until then.

## 2026-11-13 19:58Z — Model cards are generated, never hand-written
Cards for released adapters must not drift from the numbers they report. `scripts/model_cards.py` builds them only from
files already hashed in `runs/kaggle_artifacts.json` (it never hashes adapters itself) and the Kaggle train/eval summaries;
a metrics file evaluating a different adapter path stops the build. Cards say dev-only scope and that the licence is open.
Colab-era adapters are not carded (weights lost). Alternative (hand-written cards per release) rejected: error-prone.

## 2026-11-13 21:40Z — Two-seed E numbers stay out of the paper
After job21 the E arm has two seeds (3407, 2027). The pooled E-B gap is computed (`reports/pilot/seeds_dev_interim_E2.*`)
but `seeds_dev_interim.json`, `paper/tables/seeds_dev.tex` and the results paragraph keep B/C/D only until job22 completes,
so the paper never shows an arm with fewer seeds than the pre-registered three. Alternative (add E row now, flagged
incomplete) rejected: one more table edit to undo and a partial number in the draft.

## 2026-11-13T22:10Z: owner asks to focus on VGCure; exploratory VGCure-dev subset added

Owner instruction (2026-11-14 local): focus on VGCure and try to beat the MCDGraph-style method there.
"Trying" means iterating, which the pre-registered external subset (1,500 items, evaluated once after the
locked test) must not be used for. Decision:
- The pre-registered subset and its rule (once, after the locked test) are unchanged and stay guarded.
- New **VGCure-dev** subset for iteration: per directed source the 40 usable graphs after the first 50
  (`vgcure.build_subset(..., 40, skip=50)`), 400 graphs / 1,200 items, no graph shared with the pre-registered
  subset (`reports/external/vgcure_dev_subset.json`, ids and hashes only; test in `tests/test_vgcure.py`).
  Same scoring, same letterboxing, same adapted prompts.
- Everything measured on VGCure-dev is exploratory and is reported as such. Any arm or training change chosen
  with VGCure-dev is confirmed once on the pre-registered subset at the end, together with B, C, D, E.
- Consequence to state openly: once new arms are chosen using VGCure-dev, the sentence "no decision depends on
  VGCure" only holds for the arms frozen before today (B, C, D, E at three seeds and their locked test).
- No VGCure file enters any training pool (unchanged). A VGCure-like drawing style may be added to our own
  generator (our graphs, our labels) as a style bridge; that is training on our data, to be reported.

## 2026-11-13T22:15Z: locked-test queue order and cost, fixed before any test output
- All twelve trained runs (B, C, D, E x seeds 3407/2027/9173) get one final kernel each on the six locked splits
  (main + invariance copies), as `final_eval.sh` already defines. Order in `scripts/kaggle/final_queue.json`: seed rounds
  3407 -> 2027 -> 9173; inside a round B (reference), D, E (the compared arms), then C (reported only). Reason: the
  estimated total (~37.5 h Kaggle session time, `reports/pilot/final_plan.md`) exceeds one assumed 30 h quota week, so the
  evaluation spans two weeks; stopping between rounds always leaves complete seed rounds with B present, never an arm
  compared against a missing reference. Pooled test numbers are only reported once all three rounds are in.
- Estimates scale each run's measured dev eval time by 900/200 roots x (1 + 212/1200 invariance per main sample) plus
  2 x the largest past non-model session time (759 s). These are planning numbers, not measurements.
- Alternative (arm-major order: all B seeds, then D, ...) rejected: a quota stop mid-queue would leave arms without
  their seed-matched reference. Alternative (skip C on test) rejected: C is a reported baseline with three dev seeds.

## 2026-11-13T22:40Z: final (locked-test) outputs are collected all-or-nothing
- `collect_final.py` writes a run's test flags only when all twelve evaluations of its final kernel are present, the log
  has FINAL-END, and every locally re-scored flags line equals the kernel's (predictions sha256, bits, content digest; the
  per-machine images digest is recorded only). Reason: a partly collected final run would invite looking at some test
  splits before the rest exist. Files go next to the run's dev files (queue `dev_dir`), named by split, so dev and test
  flags never share a file name.
- The script sets TRACE_UNLOCK_TEST itself because it only scores an existing predictions file; the lock on model
  evaluation scripts is unchanged. Alternative (reuse collect.py with a split option) rejected: collect.py is built
  around one dev evaluation plus a train summary, and final kernels have neither a train stage nor a single split.

## 2026-11-13T23:00Z: the locked-test table exists only as a complete table
- `scripts/final_table.py` builds the test table per locked split with the same pooling, interval and reading rule as the
  dev seed table, and takes the reference arm from `reports/pilot/reference_baseline.json` (chosen on dev); the reference is
  not a command-line option, so it cannot be changed after test numbers are seen.
- It stops without printing or writing any number while any of the twelve final runs is uncollected or any arm lacks a
  seed. Reason: the 22:15Z decision (pooled test numbers only once all three seed rounds are in); with the queue spanning
  two quota weeks, a partial table would otherwise be one command away. Alternative (per-round partial tables) rejected.
- Invariance-pair scores on test are left out of this table for now (main samples only); they need their own table.

## 2026-11-14T10:20Z: VGCure-dev seed 3407 collected; other seeds pushed

jobV0 on the exploratory VGCure-dev subset: D 0.854, B 0.837, C 0.729, E 0.723 external macro (base 0.089, a format
failure). D − C +0.126 [+0.098, +0.151], D − B +0.018 [−0.012, +0.048] (paired bootstrap over graphs). One seed, so
jobV1 evaluates seeds 2027 and 9173 of B, C, D on the same subset before anything is claimed. Report
`reports/external/vgcure_dev_jobV0.md`. Still exploratory; the pre-registered subset is untouched and stays last.

## 2026-11-14T10:15Z: dev seed queue complete; locked test waits for quota and a free GPU slot
- job22 closes the dev queue: every arm has the three pre-registered seeds. E is +0.072 [+0.042, +0.102] above B
  (positive_clear), D and C stay negative. The reference baseline for the locked test stays B (chosen 2026-11-13 on dev).
- The locked test is not started in this run. Reasons: (a) jobV1 (VGCure-dev, pushed by the interactive session on the
  owner's "focus on VGCure" instruction) is running and the rule is one GPU kernel at a time; (b) `quota.py` had not
  counted jobV0 (3.64 h): real use is 18.53 h of the assumed 30 h, so 9.47 h are left and the 3407 round (B, D, E, C,
  ~12.6 h) no longer fits this week. The 22:15Z plan assumed it did. Starting a round that cannot finish would leave
  partial test output for longer than needed; nothing is lost by waiting (adapters are stored on Kaggle and hashed).
  Alternative (push jobF01 + jobF02 now, 4.3 h) rejected: it competes with the VGCure work the owner prioritised.
- Tooling: quota.py counts kernels without JOB-START / with lower-case log names (start = log download time, which is
  conservative); artifact_manifest.py hashes VGCure predictions without requiring a flags file (`external: vgcure`);
  model_cards.py ignores byte-identical adapter copies carried by evaluation kernels and still refuses different weights.

## 2026-11-14T10:18Z: locked-test invariance pairs are a descriptive table, all-or-nothing
- `scripts/final_invariance.py` reports what analysis.tex fixed before the main runs (both right, same correctness, next to
  root right and copy right) per locked split, arm and pair kind. Counts are summed over reachability and critical edge and
  over the three seeds; every seed scores the same pairs, so the pooled rate equals the mean of the per-seed rates. The
  per-seed range of the both-right rate and the per-task counts stay in the JSON.
- No interval, no reading rule, no reference arm: the pre-registered decision uses the main table only, and the pair counts
  per split are small (about 53 per kind and task on 200 roots). Alternative (bootstrap over roots as in final_table.py)
  rejected for now: it would add an unplanned inferential statistic after the analysis plan was fixed.
- Same refusal as final_table.py (nothing printed or written until all twelve final runs have their main and `_inv` flags),
  and each flags file must name the expected split, roots, sample set and dataset build.

## 2026-11-14T10:40Z: the artifact manifest refuses a half-collected locked-test kernel
- `artifact_manifest.py` reads `final_queue.json`: for every jobF with downloaded output it requires the twelve predictions
  files, each matching the flags file collect_final.py writes into that entry's dev_dir, and an evaluated adapter (parser
  for arm E) with the sha256 recorded for the source training kernel. Jobs not yet downloaded are not a problem.
- Reason: the generic check only asked for "some tracked flags file with this hash"; it would not notice a missing
  evaluation or an adapter copy that differs from the trained one. The kernel checks the adapter hash itself; this repeats
  the check on the downloaded files, which are what a release would ship.
- Alternative (a separate script) rejected: the manifest is already the step run after every collect.

## 2026-11-14T11:00Z: frozen-base prompt variants and their selection rule, fixed before any variant is run
- Background: with the common prompt the base reasons in prose (pilot_A: 99 of 180 cut off at 160 tokens, strict 0.10,
  lenient 0.40; on VGCure-dev the base scored 0.089, a format failure). analysis.tex already says the base prompt is
  chosen on dev before the locked splits; until now there was no variant and no rule.
- Variants (`base_prompt.py`): `common`; `direct` = common + "Do not explain your reasoning. Reply with the JSON object
  only." after the question; `system` = the same sentence as a system message. Three only, so the choice is not a search.
- Rule: one dev run per variant at 768 new tokens on the same roots; each variant's limit by the item-5 rule; pick the
  highest strict non-edit macro at that limit; exact tie -> common, direct, system. Lenient and truncation are reported,
  not used. The common-prompt number is always reported next to the chosen one. Only arm A is affected.
- Rejected: answer prefill ("{") and stripping reasoning before parsing (both change the scoring interface that the
  trained arms are held to); picking by lenient score (the primary score is strict); tuning the sentence after seeing
  outputs (the sentence is in the code and the test before any run).
- Size: 20 dev roots (120 samples, 80 non-edit) per variant. Estimate only: the common variant may use most of 768 tokens
  (160 tokens took 7 s/sample on the Colab T4), so the kernel could take ~1-2.5 h. 20 roots is a coarse selection; the
  rule does not need a significance claim, and the result is a floor, not a compared arm. The session counts against
  the same 7-day window as the locked test, hence not 200 roots.
- Not pushed now: jobV1 is running (one GPU kernel at a time). Order: jobA1 before jobF01.

## 2026-11-14T11:20Z: the frozen-base prompt kernel gets its own all-or-nothing collector
- `scripts/kaggle/collect_base_prompt.py` instead of extending collect.py: that script assumes one trained run with a
  train summary; jobA1 has three evaluation-only runs whose only joint result is the choice.
- Checks before anything is written: kernel finished (JOB-END); every variant present with metrics naming that variant,
  no adapter, dev and the local content digest; flags re-scored from the downloaded predictions equal the log line; the
  selection rule applied locally to the downloaded scans reproduces the kernel's BASE-PROMPT line.
- The token-budget scan is copied, not recomputed here (recomputing needs the tokenizer; the scan's strict scoring is
  the same code the flags check exercises at the run limit). Stated in each summary.json: flags are at the run limit,
  the choice uses each variant's own limit, so the two macros can differ.
- Budget line: sum of the three eval_seconds; `device` is null unless the log names the GPU (not guessed).

## 2026-11-14T11:42Z: negative results relabelled after the dev seed queue; "solid" is scoped to dev
- Entries 1 (D below B), 2 (gap is in edge criticality, not reachability) and the C part of entry 6 move from preliminary
  to "solid on the dev split (three seeds); locked test not run". Basis: three pre-registered seeds per arm, 200 dev roots,
  every seed-matched macro gap negative, pooled intervals excluding zero (`reports/pilot/seeds_dev_interim.json`).
- Kept preliminary: the perception split and the answer-bias / twin-pair counts (seed 3407 only), the oracle-text result
  (one retrained adapter, 100 roots), D2 (one seed).
- Wording for reachability: "D - B -0.014 [-0.037, +0.008]", reported as an interval containing zero, not as "equal"
  (no equivalence margin was pre-registered). The test asserts the interval straddles zero so the sentence cannot
  survive a changed result unnoticed.
- Alternative (wait for the locked test before relabelling) rejected: the RUNBOOK rule ties the label to seeds and
  sample size, and the scope is written into each heading; the test-split statement will be a separate line.


## 2026-11-14: VGCure-dev at three seeds

jobV1 collected. Pooled external macro on VGCure-dev: D 0.854, B 0.832, C 0.715. D − C +0.140 [+0.115, +0.163],
D − B +0.022 [+0.002, +0.044] (edge items +0.040, reachability +0.005). Report `reports/external/vgcure_dev_seeds.md`.
No retraining or style bridge was needed to pass C, so nothing was tuned on VGCure-dev; arms go to the pre-registered
subset unchanged, once, after the locked test.

## 2026-11-14T11:58Z: twin-pair counts over seeds are descriptive; jobA1 pushed while the locked test waits
- `scripts/seed_pairs.py` repeats analysis.tex's twin-pair and answer-bias counts for every seed and sums them. No interval
  and no reading rule, same reasoning as final_invariance.py (the decision uses the macro table; adding a bootstrap here
  would be an unplanned inferential statistic). D at seed 3407 is the Kaggle retrain, as in the seed table; the older
  "19 of 200" figure is the Colab run and stays labelled as such.
- The `critical` pair kind also yields a minimal_graph_repair row; it is kept in the report, the paper cites only the two
  non-edit tasks.
- jobA1 pushed now: jobV1 finished, no GPU kernel was running, and the session (estimate <= 3.64 h by quota.py's
  longest-session rule; own estimate 1-2.5 h) fits: 18.53 h counted + ~1.75 h for jobV1 (undownloaded, so invisible to
  quota.py; added by hand) + 3.64 h = ~23.9 h, under the 28 h line. The 3407 locked-test round (~12.6 h) does not fit this
  week either way, so jobA1 does not delay it beyond the window that already applies (its hours leave the window 2026-11-21).
- jobV1 not collected: PROJECT_STATE rule (interactive session's kernel; only if COMPLETE and untouched for 2 h).


## 2026-11-14T12:05Z: Kaggle quota resets weekly, not over a trailing 7 days

The owner's quota page showed 28 h 12 m of 30 h available at ~12:00Z on Saturday 14 Nov, while the trailing-7-day
count in quota.py said 18.5 h used. quota.py now counts sessions started since the last Saturday 00:00 UTC; kernels
without a start timestamp in their log (jobV*) take the push time from scripts/kaggle/session_starts.json. Locked test:
~37.6 h in all, so seed rounds 3407 and 2027 (~24.9 h) fit this week and round 9173 (~12.8 h) does not, unless the E
kernels (6.3-6.5 h each) get cheaper or a collaborator runs them on his own account.

## 2026-11-14T12:30Z: third locked-test seed round goes to a collaborator's Kaggle account

Owner's decision: no second account of his own; Tuna (project collaborator) runs seed round 9173 (jobF09-jobF12,
~12.8 h) on his own Kaggle account so the locked test does not spill into the next quota week. The project account
runs rounds 3407 and 2027 (jobF01-jobF08). `make_final_kernel.py --owner=<kaggle user>` changes the kernel id only;
adapters still come from the project account's kernels job17/job18/job19/job22 (shared with him as viewer) and are
checked against the recorded sha256. Outputs come back as files and are collected with collect_final.py as usual.
Waiting on: his Kaggle user name and the four notebooks being shared with him. The queue order inside a round is unchanged.


## 2026-11-14T12:15Z: the frozen base (arm A) gets a locked-test path, fixed before its dev choice and before any test output
- Gap: setup.tex lists arm A and analysis.tex says its prompt is chosen on dev before the locked splits, but
  final_eval.sh and the final queue only covered runs with an adapter. Without a path the floor would exist on 20 dev
  roots only.
- `final_eval.sh <job> base_prompt_<variant> --final-evaluation --base-prompt=<variant>:<limit>`: adapter none, same six
  splits, main + invariance copies, same strict scorer. `make_final_kernel.py <job> none A --final-evaluation --base-prompt`
  reads variant and limit from `reports/pilot/base_prompt.json` (the dev choice); they cannot be passed by hand, and the
  builder refuses while that file is missing. Written while jobA1 is still RUNNING: no variant score is known.
- Arm A stays a reported floor, not a compared arm: it is NOT added to final_queue.json's twelve, final_table.py's
  all-or-nothing set stays B, C, D, E x 3 seeds, and the kernel (working name jobF13) runs after jobF01-jobF08.
- Only the chosen variant is run on the test splits. If the choice is not `common`, the common prompt is reported on dev
  only (it may use most of 768 tokens per sample; a 900-root run could cost more than a seed round).
- Cost is unknown until jobA1 reports eval seconds; decide then whether jobF13 fits the week (after rounds 3407/2027).
- Not done: collector / artifact-manifest / final_plan coverage for jobF13 (collect_final.py needs a queue entry; add a
  separate `base` entry outside the twelve once the choice exists).
- Note: the interactive session's commit 0b493a4 (--owner) swept this run's uncommitted make_final_kernel.py edit into
  its commit; final_eval.sh and the tests are in 29278aa. Tested without a model (builder with a fake choice file,
  shell refusals before anything runs).


## 2026-11-14T12:46Z: frozen-base prompt chosen (jobA1); arm A final kernel gets a base entry; locked test opened with jobF01
- jobA1 (20 dev roots, 768-token scan, all three variants) collected from files; `base_prompt.choose` on the downloaded scans
  equals the kernel's BASE-PROMPT line: system 0.625 @96, direct 0.600 @96, common 0.1375 @768. Chosen: system, 96 new tokens.
  The system/direct gap is two of 80 questions; the rule was fixed before the run, so it stands, and the paper says the
  choice is not evidence that one variant is better.
- jobF13 (arm A) is a `base` key in final_queue.json, not a thirteenth `jobs` entry: final_table.py / final_invariance.py
  keep their twelve-run all-or-nothing set. Its entry is derived from reports/pilot/base_prompt.json by the same reader the
  kernel builder uses; collect_final.py refuses metrics with another adapter, variant or token limit.
- Cost: ~2.37 h upper bound (dev seconds of the chosen variant measured at the 768 scan limit). Rounds 3407+2027 (~24.9 h)
  leave ~0.8 h of this week's assumed quota after the margin, so jobF13 most likely runs in the next quota week.
- With the base prompt chosen, every dev decision listed in analysis.tex (reference baseline B, token limits, base prompt)
  is recorded. jobF01 pushed 2026-11-14T12:43Z under the standing rule (no kernel running, quota.py ok, rounds 3407+2027
  fit). Test numbers stay unread until all twelve are collected (final_table.py is all-or-nothing).



## 2026-11-14T12:55Z: twin-pair and answer-bias counts on the locked test get their script before any test output is read
- Gap: analysis.tex ("Twin pairs and answer bias") says the counts are reported for every arm and seed, and the dev finding
  the paper leans on (D gets few critical-edge twin pairs both right) needs its test-split counterpart. Only
  `scripts/seed_pairs.py` (dev) existed; writing the test version after seeing test numbers would be a choice made on the test.
- `scripts/final_pairs.py` applies seed_pairs.build unchanged to each of the six locked splits' main flags: same cells, same
  summing over seeds, no interval, no reading rule, no reference arm. All-or-nothing over the twelve queue entries, like
  final_table.py and final_invariance.py. Arm A (jobF13) is not part of it.
- Written while jobF01 is RUNNING and uncollected; tested with constant-answer flags only.


## 2026-11-14T13:25Z: density-matched slice on the locked test gets its script before any test output is read
- Gap: analysis.tex ("Slices") says size-OOD is also scored on roots whose edges per node lie inside the central 5-95% of the
  512 training roots, shown next to the full split. `scripts/density_slice.py` handles one flags file and no seeds; writing the
  three-seed test version after seeing test numbers would be a choice made on the test.
- `scripts/final_density.py`: window = analysis.density_window on pilot train (metadata only; a test pins it to
  reports/density/pilot_dev.json and to the paper's [0.875, 1.667]). Per split and arm: non-edit macro on all roots and on the
  matched roots, mean over the three seeds, per-seed values kept. No interval, no reading rule, no reference arm (same reasoning
  as final_invariance.py / final_pairs.py: the decision uses the main table only).
- All six locked splits are computed, not only size-OOD, so no split is selected after the fact; the paper's sentence is about
  size-OOD. A split with no matched root prints n/a. No minimum matched-root count is imposed (descriptive; the count is shown).
- All-or-nothing over the twelve queue entries. Arm A (jobF13) is not part of it. Written while jobF01 is RUNNING and
  uncollected; tested with constant-answer and metadata-derived flags only.

## 2026-11-14T15:20Z (autopilot)
Collected jobF01 without reading any test output (collect_final.py prints none); pushed jobF02 (D seed 3407) per final_queue.json order, one GPU kernel at a time, quota ok. No design change.


## 2026-11-14T17:25Z: slice breakdown on the locked test gets its script before any test output is read
- Gap: analysis.tex ("Slices") says accuracy is broken down by true answer, image variant, graph family, size bucket, layout
  and style. Only `scripts/breakdown.py` (one flags file, used on dev) existed; choosing facets, buckets or which rows to show
  after seeing test numbers would be a choice made on the test.
- `scripts/final_slices.py` applies analysis.breakdown unchanged (facets = analysis.FACETS, a test pins them to the paper's
  sentence) to each of the six locked splits' main flags. Per split, task, facet value and arm: right / questions summed over
  the three seeds, per-seed accuracy kept. Every facet value is printed, including small ones (n is shown), so nothing is
  selected. No interval, no reading rule, no reference arm (the decision uses the main table only).
- All-or-nothing over the twelve queue entries. Arm A (jobF13) is not part of it. Written while jobF02 is RUNNING and only
  jobF01 is collected (its numbers unread); tested with constant-answer and metadata-derived flags only.
- test_final_plan's queue-order assertion assumed no kernel was collected yet; it now checks open kernels only. final_plan.py
  itself is unchanged.

## 2026-11-14T17:36Z: per-task detail on the locked test gets its script before any test output is read; jobF02 collected, jobF03 pushed
- Gap: analysis.tex ("Cells and task means") says format failures (strict) and the lenient score are reported per task and, for
  repair, status accuracy, valid-edit rate and minimal-repair accuracy; "Perception or reasoning" (ii) says arm E is reported
  with its extraction quality. final_table.py's JSON holds the task means and their intervals, but nothing showed the other
  quantities for the locked splits; choosing which of them to show after seeing test numbers would be a choice made on the test.
- `scripts/final_metrics.py` reads the metrics files copied into final_summary.json by collect_final.py (main samples only).
  Cell = mean over the three seeds of the per-seed value (same samples every seed); per-seed values kept. Lenient score for
  arm E = accuracy under the lenient parse of the extraction (`metrics_lenient_parse`), for the other arms `lenient_accuracy`.
  Extraction quality = the strict-parse summary (exact, format ok, edge precision / recall, direction accuracy, node recall).
  No interval, no reading rule, no reference arm (the decision uses the main table only).
- Integrity rule: each strict per-task accuracy in a metrics file must equal the one recomputed from the tracked flags file
  (tolerance 1e-9), same split / roots / content digest; otherwise the script stops.
- All-or-nothing over the twelve queue entries. Arm A (jobF13) is not part of it. Written while jobF02 was RUNNING and only
  jobF01 was collected (its numbers unread); tested with constant-answer metrics only.
- jobF02 collected without reading any test output; jobF03 (E seed 3407) pushed per final_queue.json order, one GPU kernel at
  a time on the project account, quota ok (6.63 h used + ~6.4 h estimate). No design change.


## 2026-11-14T20:10Z: arm A floor on the locked test gets its report script before any test output is read
- Gap: jobF13 (frozen base, a reported floor) had a kernel, collector and manifest entry but no script that turns its collected
  files into the reported numbers; choosing what to show after seeing them would be a choice made on the test.
- `scripts/final_floor.py`: per locked split the non-edit macro and each task's accuracy with a 95% interval over root graphs
  (seeds.pooled_bootstrap on one run, same stream as the seed tables), plus lenient accuracy, format-failure rate, repair
  status / valid-edit / minimal-repair and truncation from the metrics file (final_metrics.run_cells). No difference to any
  arm and no reading rule (A is not compared; the decision uses final_table.py alone).
- Checks: metrics report adapter none and the dev-chosen variant / token limit (collect_final.base_entry "expect"); strict
  per-task accuracy equals the flags file; same split / roots / content digest; no missing responses.
- All-or-nothing over jobF13 and the twelve queue entries: earlier runs deferred this script as "written blind", but the other
  locked-test scripts were also written before any output and tested on constant answers; the same is done here.


## 2026-11-15T00:42Z: torch-dependent tests skip in the CPU-only venv
- `tests/test_vgcure_eval.py::test_eval_script_on_stub_model` and `tests/test_kaggle_final.py::test_eval_parse_solve_with_oracle_extraction` import `trace_vr.model` (torch). The runbook's `.venv` has no torch, so they failed there; both now `pytest.importorskip("torch")` and still run under `.venv-model` (317 passed). No code under src/ or scripts/ changed.


## 2026-11-15T01:30Z: one runner for the locked-test reports, fixed before any test number is read
- `scripts/final_reports.py` runs final_table, final_invariance, final_pairs, final_density, final_slices, final_metrics,
  final_floor in that order, each with --out-json/md/tex under reports/final/, and stops at the first non-zero exit, so no
  later report is opened while the main table refuses. It does not print the scripts' tables (the files hold them).
- Reason: the unlock step was seven hand-typed commands; a single fixed command removes order and path mistakes at the moment
  the test is opened. No script's content changed.


## 2026-11-15T04:05Z: final_plan.py calibrates arm E by its measured final-kernel time
- jobF11 (E, seed 9173) evaluation took 28040.5 s against a plain estimate of 22705 s (ratio 1.235); B, C, D final kernels
  ran at 0.91-1.01 of theirs. The parse-and-solve kernel has extra stages (parser pass, solver) the dev-based scale misses.
- final_plan.py now multiplies each arm's estimate by the largest measured/estimated ratio of its collected final kernels
  (runs/budget.jsonl "final locked-test" lines; floor 1.0) and uses that for weekly packing and the 11 h flag. The plain
  estimate stays in the table. Remaining E kernel jobF07: ~8.09 h, inside Kaggle's 12 h session limit.
- Only durations are read; no test outcome.

## 2026-11-15T04:25Z: jobF04 waits for the Kaggle weekly reset
- jobF03 collected (7 of 12 locked-test kernels in). quota.py: 28.96 h of the assumed 30 h used since 2026-11-14T00:00Z.
  jobF04 (C, ~2.27 h by final_plan) does not fit under the 2 h margin, so it is not pushed this week.
- Known conservatism: quota.py estimates a job not in queue.json (all final jobs) with the longest past session (8.09 h).
  Verdict is the same here; left unchanged so the check stays strict. Revisit if it blocks a job that clearly fits.

## 2026-11-15T04:50Z — push gate for final kernels is final_plan.py --gate
quota.py only knows queue.json, so for final kernels it estimated the next job with the arm's longest past session and, with --running, did not use the calibrated final estimate. The gate for jobF04-F08 and jobF13 is now `scripts/kaggle/final_plan.py --gate [--running JOB]` (same weekly accounting as quota.py, calibrated per-arm estimates, 2 h margin). quota.py stays for the training queue. Why: avoid both a false 'wait' and double-counting a running kernel.

## 2026-11-15T05:10Z — release audit before any upload
`scripts/release_audit.py` scans every tracked file: token-shaped strings (Kaggle, HF, GitHub, sk-, AWS, private keys) and home-directory paths fail the audit; the Kaggle account slug, Colab Drive link and e-mail addresses are review items for the owner (not secret, but personal), and RUNBOOK/PROJECT_STATE/DECISIONS/PLAN/BLOCKERS/master brief/prior_gpt_work are listed as internal notes to leave out of a code release. First run: pass (0 secrets, 0 home paths), 46 review lines. Why: the goal is a public code/benchmark release; this makes the 'no secrets' checklist row checkable instead of asserted. Strength: engineering.

## 2026-11-15T05:35Z — second overlap search; arm E framed as a reference arm
`scripts/overlap_search.py` (14 fixed arXiv API queries, raw XML and hits kept under docs/sources/overlap_2026-11-15/) found
no work that trains a VLM to output goal-satisfying graph edits and tests transfer to non-edit questions. It did find close
precedents for parse-and-solve: VGCompiler (arXiv 2609.22327, RL to recover a graph and compile an executable operation) and
VisionGraph's description-program-reasoning chain (2405.04950); also Knossos/Ariadne (complete diagram topology extraction)
and MapReason-OSM (exact-oracle scoring of structured map decisions). Decision: cite all four; the paper does not claim
parse-and-solve as a contribution; arm E stays the reference for "does edit training beat explicit graph recovery".
Claims come from abstracts only. Strength: engineering (literature hygiene), not evidence about TRACE results.

## 2026-11-15T06:00Z — evaluation code frozen at jobF01's commit; checked in the push gate
The final kernels are built from HEAD over two quota weeks, so their payload hashes differ. Rebuilding payloads from git
history matched each collected kernel's recorded payload sha256 to a commit: jobF01 4062044/dfb45eb, jobF02 d535597,
jobF03 6a9a9e3 (and later commits with the same payload), jobF09-F12 a4ca5a9 (matched against the kernel.py files in
dist/kaggle/, i.e. the last local build of each job). `scripts/kaggle/eval_freeze.py` hashes the git blob ids of every
file the final evaluation executes or reads (pyproject.toml, src/trace_vr, configs, manifests, final_eval.sh, eval_arm.py,
eval_parse_solve.py, export_flags.py, make_kernel.py) and records the digest at 4062044 in `scripts/kaggle/eval_freeze.json`.
All four commits above and HEAD give the same digest, so the seven collected kernels ran identical evaluation code.
`final_plan.py --gate` now prints `stop` (with the changed paths) if HEAD's digest differs. Dry builds of jobF04-F08 and
jobF13 from HEAD succeeded (not pushed); source kernels job12/14/15/16/21 are COMPLETE on Kaggle. Why: one locked test,
evaluated in twelve kernels, must be one evaluation. Strength: engineering. No test output read.

## 2026-11-15T06:15Z — the deciding six-cell primary statistic gets its own final script
- Gap found: protocol section 5 and paper setup define the primary statistic as the unweighted mean of six accuracies
  {reachability, critical edge} x {test_id, test_style_ood, test_struct_ood}, and say only it decides the hypothesis; but
  final_table.py reports each locked split separately and no final script computed the six-cell mean or its reading.
- `scripts/final_primary.py`, fixed before any locked-test number was read: per arm and seed the six cells (task accuracy
  within a split), their mean and the two task means; pooled = mean over the three seeds; interval = stratified paired
  bootstrap (each split resamples its own root graphs, `Rng("primary-v1", split, roots, n_boot)`, 2000 replicates, 95%
  percentile), all arms and seeds on the same draw. Reading vs the dev-chosen reference (B) by `seeds.reading` on the primary.
  Secondary flags (never upgrade a reading): task_drop (task-mean point > 0.02 below the reference: no "general improvement")
  and splits_same_sign (the three per-split differences share the primary's sign: "not driven by one style"). E - D reported.
- Alternative rejected: averaging final_table's three per-split macro intervals (no joint interval, no seed rule on the primary).
- Lives in scripts/, not src/trace_vr, so the frozen evaluation digest (eval_freeze.json) does not change; gate verified.
- final_reports.py runs it second, right after final_table.py. Strength: engineering. No test output read.

## 2026-11-15T11:50Z — paper's locked-test section is generated from reports/final, never typed
- New `paper/sections/test_results.tex` (after the dev results): one `\finaltable{name}{caption}{label}` per script in
  `final_reports.ORDER`, primary statistic first. Each table `\input`s `../reports/final/<name>.tex` if it exists, else
  prints a "pending" line, so the draft compiles now and fills itself when `final_reports.py` succeeds.
- Text contains no numbers (test checks digits outside labels/file names, only "95\%" allowed) and no reading; the reading
  comes from final_primary.py. Compiled both ways: pending (repo as is) and filled (copy with dummy tabulars in a scratch
  dir), no errors. `tests/test_paper_test_results.py`. Eval-freeze files untouched. Strength: engineering. No test output read.

## 2026-11-15T15:40Z — clean-clone reproduction check run and recorded

`scripts/clean_check.py --workdir <empty scratch> --out reports/clean_check.json` on HEAD 141e11d: ok.
Content and image digests of smoke/pilot/pilot_test/pilot_diag equal the tracked manifests on this
Mac (images match here because it is the machine that wrote them; across machines only the content
digest is required). Full suite exit 0 with TRACE_REQUIRE_DATA=1. Takes ~14 min CPU (pilot_test
alone 311 s), so it is a pre-release check, not an every-run check. The checklist row stays
"partial" until a GPU train+eval is reproduced from a clean clone (could be one Kaggle kernel after
the locked-test kernels; not before jobF13, to keep the quota for the final evaluation).

## 2026-11-15T16:40Z — dress rehearsal of the locked-test reporting path; paper build bug fixed
- `final_reports.py` had only ever run while refusing, so its eight scripts and the filled paper section had never run
  end to end. New `scripts/rehearse_final_reports.py`: clones HEAD, links data/, deletes every collected
  flags_test_*/final_summary.json in the clone (unread) and writes synthetic ones for all 12 queue entries + jobF13
  (hashed true/false/prose answers scored against the labels, collect_final.py layout, arm E and arm A metric fields),
  runs final_reports.py, compiles the paper. Only a status record leaves the clone (`reports/final_rehearsal.json`).
- First run: all eight scripts ok, but the paper did NOT compile: `final_slices.tex` is a longtable and was wrapped in a
  table float + \resizebox (the 11:50Z "dummy tabulars" check could not catch it). Fix: `longtable` package and a
  `\finallongtable` macro that inputs it directly (scriptsize, 4pt column padding). Re-run on 827ec11: final_reports exit
  0, paper exit 0, 0 overfull boxes, 0 pending lines. Test `test_longtable_reports_are_not_floated`.
- Only paper/, tests and a new script changed; eval-freeze set untouched (gate still `wait`, not `stop`).
  Strength: engineering. No test output read.

## 2026-11-16T08:44Z Record the GPU software environment in the remaining final kernels

- Found: no Kaggle kernel so far printed torch / CUDA / Python / image versions (only the peft and transformers pins and
  `nvidia-smi`). jobF04-08 and jobF13 run at least a week after jobF01-03/09-12, so a Kaggle image update between the
  two groups would be invisible.
- Decision: `make_final_kernel.py` now runs a small probe after the pins are installed and prints one `ENV-SUMMARY`
  line into the kernel's own log (not `<job>.log`, which `collect_final.py` checks). The evaluation code is untouched
  (`make_final_kernel.py` is not a frozen path; `final_plan.py --gate` still prints `wait`, not `stop`).
  `scripts/kaggle/kernel_env.py` reads only that line and exits 1 on any differing field.
- Limit, stated not hidden: the seven collected final kernels and all dev kernels have no such record
  (`kernel_env.py` prints `not recorded` for jobF01-03, jobF09-12). If jobF04-08 agree with each other, that still says
  nothing about the first group. No test number was read to make this change.

## 2026-11-16T09:33Z External evaluation gets a paper section; exploratory numbers are labelled as such

- Found: `paper/` mentioned VGCure only in related work, while `reports/external/vgcure_dev_seeds.md` (three seeds, 2026-11-14) existed.
- Decision: new section after the locked-test section. It states the adapted protocol (own prompts, strict JSON, 512x512 letterbox,
  balanced accuracy for reachability), separates the pre-registered subset (not evaluated, pending slot) from the exploratory one, and
  reads D - B on VGCure-dev as "close", not as D ahead: pooled interval just above zero, but only one of three seed intervals excludes zero.
  It also says why this does not contradict dev: VGCure has no critical-edge items, and on dev D's reachability is level with B's.
- Table generated by `scripts/vgcure_table.py` from the two pooling JSON files (refuses if they disagree on subset, runs or pooled scores).
  It prints `%.3f` of the stored values, so three cells differ in the last digit from the hand-rounded `vgcure_dev_seeds.md`
  (C macro 0.714 vs 0.715, B - C +0.117 vs +0.118, D - B upper bound +0.043 vs +0.044); same records, the JSON is the source.
- First build failed (tabular column spec one short) and was 21 pt overfull; fixed, a test now counts columns. Strength: engineering.
  No locked-test or pre-registered VGCure output read.

## 2026-11-16T09:43Z Model-agnostic export / score script
- Decision: `scripts/benchmark_io.py` is the public way to use the benchmark with another model. It only calls existing library
  functions (to_model_example, score_flags, metrics.evaluate, cluster_bootstrap, pair_table); no scoring rule is duplicated.
- Why: a released benchmark that can be scored only through our Qwen/LoRA scripts is of little use to others (paper contribution 1).
- It lives outside the frozen evaluation paths (`eval_freeze.json`), so the locked test stays one evaluation; it keeps the
  test-split lock. Missing predictions are wrong, unknown/repeated ids are errors (a silent partial file would inflate scores).

## 2026-11-16T09:52Z CPU demo shows the verifier, not a model
- Decision: the first demo (brief section 22) is a notebook without any model: `demo/walkthrough.py` scores hand-written candidate
  responses with `metrics.score_sample` on one root graph, its twins and its repair task. Each candidate is kept only if the scorer
  itself returns the verdict its label names, so no label is asserted by hand.
- Why not model outputs yet: the dev predictions of the trained arms live in the untracked `runs/kaggle/`, and a gallery of them needs
  a selection rule fixed before looking (brief: not a single picked success). That is the next M8 piece; locked-test outputs stay closed.
- The notebook is generated (`scripts/build_demo_notebook.py`) and stored without outputs; a test executes its code cells with plain
  `exec`, since neither venv has Jupyter. The picked root is by file position; `walkthrough.load` keeps the test-split lock.
- When `data/` is missing the notebook builds the smoke set into a temp folder with its own manifest, so the tracked
  `manifests/smoke.json` is never rewritten. Strength: engineering. No test output read.

## 2026-11-16T10:20Z Dev gallery: selection rule fixed before looking, responses only from recorded files
- Decision: the gallery (brief section 22: success, failure, critical-edge change, impossible goal; not one picked success) is generated
  by `scripts/dev_gallery.py` under rule `gallery-v1`. Candidates are ordered by sha256("gallery-v1:" + sample_id); three slots
  (critical pair, repair edited, repair impossible) use no response at all; three (reachability success / failure, repair failure)
  take the first sample in that order that arm D gets right / wrong. Each sample is used once; a slot without a candidate is printed empty.
- The rule and the code were written before any per-sample outcome was inspected; the first real run's output is the committed one
  (one later rerun changed only wording of the verdict column, not the selection).
- Pool = samples every listed arm answered. Including the frozen base (jobA1, 20 dev roots) limits the pool to those 20 roots; chosen
  so that every card shows the base next to the trained arms (brief: base and TRACE on the same input).
- Each predictions file must have the sha256 stored in the tracked flags file of that run, so the page shows the responses the reported
  numbers were computed from. Arm E rows state the parser's distance from the true graph (its answer is conditional on that reading).
- Limits, stated on the page: 7 cards are an illustration, not a measurement; style-change pairs have no dev responses (their variant
  images were evaluated only on the locked test, which stays closed); the edit is not yet drawn onto the image; seed 3407 only.
- Lives in scripts/ and reports/; frozen evaluation paths untouched (gate `wait`, not `stop`). Strength: engineering. No test output read.

## 2026-11-16T10:37Z Dev figures: copied values, full axes, refusal on mismatch
- Decision: figures are drawn by `scripts/dev_figures.py` from the seed-pooled dev record and the tracked run summaries only; the script
  copies values and re-estimates nothing. Accuracy axes run 0 to 1 and the time axis starts at 0 (brief section 22: no misleading axes).
- The cost figure uses measured `train_seconds + eval_seconds` per run (Kaggle T4; setup and data generation not counted, as in
  budget.jsonl). A summary whose macro or seed differs from the seed record is refused, so a point cannot mix two evaluations.
- Arm A is left out of both figures (evaluated on 20 dev roots, not comparable on one axis); no OOD-drop figure before the locked test opens.
- Paper: one figure (`figures/arms_dev.pdf`) and a cost paragraph; "more than three times" is E's mean dev evaluation time over the mean
  of B, C, D (tested to lie between 3 and 4). It is evaluation time on the development split, not a claim about deployment cost.
- Outputs are byte-reproducible (fixed SVG hash salt, no dates in metadata; checked by running twice). Lives outside the frozen
  evaluation paths (gate `wait`, not `stop`). Strength: engineering. No test output read.



## 2026-11-16T11:10Z Error-type figure: one bar per run, shares from the summaries
- Decision: `errors_dev` (same script, rule now `dev-figures-v2`) draws one stacked bar per run and task, not a seed average: strict
  accuracy and format-failure rate are copied from the run's `summary.json`; the only derived value is the parsed-but-wrong share,
  1 - accuracy - format failures (valid because a format failure always scores as wrong, `tasks.score_sample`).
- Refused when the summary's task accuracy differs from the seed record, when the shares exceed 1, or when arm E's count of
  `premise_fails_on_predicted_graph` is not exactly its critical-edge format failures (it is, in all three E runs: 47, 50, 62 of 400).
- What the recorded dev numbers say (dev only, nothing new measured): C and D have no format failure in any run; B none on the
  non-edit tasks and 29 / 7 / 1 of 400 on repair; lenient = strict in all nine B/C/D runs. So end-to-end errors are well-formed wrong
  answers. For E, 47 of 67, 50 of 72 and 62 of 84 critical-edge errors are unanswerable questions (premise fails on the transcribed graph).
- Paper: Figure `fig:errors-dev` and paragraph "Error types" in the results section; every count in it is asserted by a test.
- Lives outside the frozen evaluation paths (gate `wait`, not `stop`). Strength: engineering. No test output read.

## 2026-11-16T11:13Z Protocol raised to version 1.0 as executed; rules pinned to the commit that opened the locked test
- Gap: the protocol promised a 1.0 freeze before the main runs, but the file stayed "0.1 draft, nothing run on a model" while the
  locked test was opened (jobF01, 2026-11-14T12:43Z). A reader could not tell which rules were in force.
- Decision: do not rewrite history. Sections 1-6 stay byte-identical to commit 4062044 (they were last changed in e9f3292, an ancestor);
  the header states that the version number was raised on 2026-11-16, after opening and before any test number was read; section 8
  records what is frozen and the deviations. The largest one: the study is pilot scale only (512 training roots; the 5,000-root main
  build was never run), so every claim in the paper must be worded as a pilot-scale claim.
- `scripts/protocol_freeze.py` checks sections 1-6 (sha256) and five rule/choice files (git blob ids recomputed without git) against
  `docs/protocol_freeze.json`; all unchanged since 4062044. It complements `eval_freeze.py` (code) with the rules.
- Stated, not fixed: blank-image and wrong-image controls were never run on a model; path-length and crossing slices do not exist;
  arm E's 512-token limit is outside item 5's candidate list (0 truncated). Adding them now would need evaluation-code changes
  (frozen until jobF13) or new GPU runs on dev; left for after the locked test, as dev-only additions if at all.
- Strength: engineering / documentation. No test output read.

## 2026-11-16T11:30Z Paper states pilot scale and the controls that were not run
- Gap: protocol section 8 (deviations 1 and 7) was only in docs/; the paper's setup read as if the planned experiment had been run,
  and limitations.tex spoke of "the main build" as existing.
- Decision: state it in the setup (paragraph "Scale") and in the limitations ("Pilot scale", "Controls not run"). The image-use claim
  is worded as resting on the twin design (image-blind ceiling 0.50), not on an image ablation, because no blank / wrong-image run exists.
- Not done: running those controls. They need evaluation-code changes (frozen until jobF13 is collected) and GPU time; candidates for
  dev-only additions afterwards. Strength: documentation. No test output read.

## 2026-11-16T12:03Z Model-free baselines per split go into the paper
- Gap: the brief (section 16) asks for class balance and the majority-answer score of the Boolean tasks, and section 17 item 1 for a
  question-only floor. The numbers existed only in the validator reports and, for train/dev, in the datasheet; the paper had one sentence.
- Decision: one generated table (`scripts/modelfree_table.py`) covering every split of pilot, pilot_test and pilot_diag. The locked
  splits are included because these values come from labels and instructions at build time (2026-11-12, before any model saw the test),
  involve no model output and cannot be tuned; they are the floor the locked-test table will be read against.
- Edge existence is left out of the table: it is a secondary task without twins, its ceiling is already discussed in limitations.tex.
- The heuristic column keeps the report's four decimals (the values are not round); the text says "no text heuristic exceeds 0.27"
  (maximum over the eight splits is 0.27, test style OOD).
- Strength: documentation. No test output read.

## 2026-11-16T12:09Z Third-party inventory as a generated, tested file
- Gap: brief section 23 asks for a separate licence review of code, data, fonts, model and notebooks and a notice file if needed;
  the repository had one README sentence and a checklist row.
- Decision: `THIRD_PARTY.md` is generated (`scripts/third_party.py`) so that the parts that can drift (font hashes, model and dataset
  revisions, pinned package versions, tracked file types) are read from the repository, and each licence statement names its evidence:
  licence file read (fonts), model card + licence file of the pinned snapshot (base model), "none stated" (VGCure, checked 2026-11-13),
  package metadata (Python packages; the publisher's declaration, not a reading of the licence text). No legal conclusion is drawn.
- Consequence written down: no VGCure file may enter the repository or a release; results on it are numbers plus sample ids. The script
  refuses a tracked path under `external/`.
- Observed while doing it: `assets/fonts/` is byte-identical to matplotlib 3.9.4's bundled DejaVu files including the licence file;
  the files' name table says "Version 2.35". The Kaggle image's torch version is still unrecorded for the seven collected kernels.
- Not done: licences of tools (Tectonic, Kaggle, Colab) and of transitive packages; project licence stays the owner's decision.
- Strength: documentation. No test output read.

## 2026-11-16T12:40Z Gallery: named edits drawn onto the image, colour = operation, not verdict
- Decision (brief section 22 item 4: the edit is marked on the image): on repair cards `scripts/dev_gallery.py` draws, one image per
  distinct edit, every edit named by the generator's stored answer (`reference`) or by a response that parses under the strict repair
  schema (rule `overlay-v1`). Removal: the edge's own recorded route repainted, cross at its middle. Addition: a straight dashed arrow
  between the two nodes; it is a marker, not a generator layout, and may pass over other edges (said on the page).
- Colour encodes the operation only (blue add, vermillion remove). Colouring by right / wrong was rejected: the verdict has three parts
  (legal, goal met, minimal) that one colour would flatten, and it stays in the table next to the response.
- An edit that names an absent edge, an absent node, an edge already present or a self loop is listed with that reason and not drawn;
  a response that does not parse contributes no edit (arm A's `"src":"H->W"` on the first repair card is such a case).
- The marker is composited as a separate layer, so pixels away from it are the dataset image's own (tested).
- Selection rule `gallery-v1` untouched: apart from the new `overlay` / `edits` fields the regenerated record equals the previous one
  (compared before committing). Output on the tracked page: 3 overlays (remove G->H named by reference and E; add W->L named by D;
  add F->W named by reference only), 2 not drawn (A's removals of edges that are not in the image). Byte-identical on a second run.
- `reference` is one minimal solution; where several exist (first repair card: also remove H->D) only the stored one is drawn. Stated.
- Regenerate: `PYTHONPATH=src .venv/bin/python scripts/dev_gallery.py manifests/pilot.json dev --focus D
  A=runs/kaggle/jobA1/runs/base_prompt_system:runs/base_prompt_system/flags_dev.json
  B=runs/kaggle/job11/runs/pilot3_B:runs/pilot3_B_kaggle/flags_dev.json C=runs/kaggle/job12/runs/pilot3_C:runs/pilot3_C/flags_dev.json
  D=runs/kaggle/job20/runs/pilot3_D:runs/pilot3_D_kaggle/flags_dev.json
  E=runs/kaggle/job13/runs/pilot3_E:runs/pilot3_E/flags_dev.json:runs/kaggle/job13/runs/pilot3_E/extraction_dev.jsonl
  --out reports/pilot/gallery_dev.json --out-md reports/pilot/gallery_dev.md --copy-images reports/pilot/gallery_dev`.
- Lives in scripts/ and reports/; frozen evaluation paths and protocol rules untouched. Strength: engineering. No test output read.

## 2026-11-19T22:35Z Compute report generated from the budget log; totals are lower bounds
- Gap: brief sections 23-24 ask that real GPU hours and failed jobs be reported and that the budget record be part of the package;
  only the raw `runs/budget.jsonl` and a hand-summed `gpu_seconds_total` existed.
- Decision: `scripts/budget_report.py` sums the log as it is and estimates nothing. Lines without `gpu_seconds` (five lost Colab runs)
  count as zero and are listed with their status; the three `running` placeholder lines are named separately. The RUNBOOK's
  "about 3 T4-hours" for job5 and the log note's "25 GPU-minutes" are hand estimates and stay out of every total (the note is quoted).
- Grouping is by run_id pattern, first match wins; an unmatched run_id stops the script, so a new kind of run must be named first.
  "Failed" means the line's status starts with `failed` or `partial`; job9's D part (status text, time measured) counts as measured.
- Not done: Kaggle session wall time (what the quota counts) is in untracked logs under `runs/kaggle/` and is not in this report;
  the report says the logged seconds undercount it. Device is missing on 8 Kaggle lines (reported as "not recorded", not assumed T4).
- Reading final-kernel seconds is not reading a test result (timings only; no flags, metrics or predictions opened).
- Strength: documentation. No test output read.

## 2026-11-20T00:05Z Compute numbers in the paper come from the budget report through macros
- Gap: PLAN item 41 left the compute report out of the paper.
- Decision: `scripts/compute_table.py` (rule `compute-table-v1`) writes `paper/tables/compute.tex`: one `\\compute...` macro per
  number quoted in prose plus the table (by purpose, by platform, total). It recomputes the summary from `runs/budget.jsonl` and
  refuses `reports/budget.json` if it differs, so the paper cannot quote a stale report. The section text holds no compute digits
  (keeps `test_no_model_results` meaningful); the totals change as final kernels are collected, so rerun after each collect.
- Guards: stops if a device other than the T4 is recorded or an untimed run is not on Colab (the prose says both).
- Wording: "Kaggle's free weekly quota and Google Colab" (Colab is the owner's existing plan, not called free). "Lines not timed"
  counts lost runs and launch placeholders together; the caption says so.
- Current values (from the log): 48.24 h over 37 timed lines, locked-test kernels 24.67 h (7). PDF compiles cleanly.
- Strength: documentation. Frozen evaluation paths and protocol rules untouched. No test output read.

## 2026-11-19T23:40Z Style-pair cards come from the locked test only, after the final table is accepted
- Gap: brief section 22 asks the demo to show a style change; the dev gallery has none because no dev response to a restyled
  copy was recorded. PLAN listed the cards as "possible only after the locked test is opened" with no script.
- Decision: `scripts/final_style_cards.py`, rule `style-cards-v1`, written now so the selection is fixed before any locked-test
  output is read. Gate: `final_figures.load_table` must accept `reports/final/final_table.json` (exactly the twelve queue runs)
  before any prediction or flags file is opened; today it refuses (tested on the real repository).
- Selection: test_id and test_style_ood; pool = restyled copies (pair kind invariant_style) whose base is in the main set; order
  sha256("style-cards-v1:" + variant id); per split the first reachability and first critical-edge pair. No response or score
  enters the selection (tested by swapping all responses: same pairs chosen). Empty slot stays empty.
- Shown: per run (arm x seed, 12 rows) the recorded root and copy responses, verifier verdicts, same correctness. Each predictions
  file's sha256 must equal the one in the collected flags or the script stops. Arm A (jobF13) is outside the final table; not shown.
- Illustration only; rates stay in final_invariance.py's table. Rehearsal now also writes synthetic predictions files (recorded in
  the synthetic flags) and runs the card script in the clone.
- Frozen evaluation code (src/trace_vr, configs, manifests, eval scripts) and protocol rules untouched. Strength: engineering.
  No test output read.

## 2026-11-20T00:25Z Announcement draft and Turkish summary are generated from the primary file, wording fixed now
- Gap: PLAN item 39 left "short English announcement draft and Turkish result summary (after final_primary.py)" open with no rule,
  so the wording would have been chosen after seeing the result.
- Decision: `scripts/result_summary.py`, rule `summary-v1`. Only input: `reports/final/final_primary.json`. The reading of D against the
  recorded reference picks one of five fixed sentence pairs (EN/TR) written today; task_drop, splits_same_sign, E - D and C - reference
  are listed and never change the sentence. Both texts carry the pilot-scale paragraph (512 root graphs, Qwen3.5-2B LoRA, synthetic
  drawings; tested against limitations.tex) and are titled DRAFT; nothing is posted (owner confirms any publication).
- Refuses and writes nothing while the primary file is missing (true today) or has no D - reference row.
- Rehearsal (scripts/rehearse_final_reports.py) now runs it in the synthetic clone: ok, paper compiles (reports/final_rehearsal.json).
  The rehearsal numbers are synthetic and stay in the scratch clone.
- Full suite: 451 passed, 6 skipped. Frozen evaluation code and protocol rules untouched (protocol_freeze ok). Strength: documentation.
  No test output read.

## 2026-11-20T00:40Z Abstract and intro finding paragraph are generated from the primary file, wording fixed now
- Gap: PLAN items 37/43 left the abstract and the intro finding paragraph to be written "after the primary reading", i.e. with
  the result in view.
- Decision: `scripts/paper_finding.py`, rule `finding-v1`. Only input: `reports/final/final_primary.json`; the D - reference reading
  (taken through result_summary.parts, same refusals) picks one of five fixed LaTeX sentences written today. Fixed benchmark
  sentence (design bound 0.50 only), numbers sentence (point, 95% interval, seed count), pilot-scale sentence (tested against
  limitations.tex). Secondary facts stay out of the abstract; the intro paragraph points to Section test-results.
- Output goes to reports/final/{abstract,intro_finding}.tex; main.tex / intro.tex \input them only if present. Refuses while the
  primary file is missing (true today), so the real paper still claims no result.
- Rehearsal rerun on synthetic flags: ok, paper_finding exit 0, paper compiles with 0 overfull boxes, 0 pending lines
  (reports/final_rehearsal.json). Synthetic numbers stay in the scratch clone.
- Full suite: 460 passed, 6 skipped. protocol_freeze and eval_freeze ok. Strength: documentation. No test output read.


## 2026-11-20T09:20Z Conclusion is generated from the primary file, wording fixed now
- Gap: PLAN item 44 left the conclusion section to be written after the primary reading.
- Decision: `scripts/paper_conclusion.py`, rule `conclusion-v1`. Only input final_primary.json; finding sentence, numbers sentence and
  pilot-scale sentence imported from paper_finding.py (finding-v1) so the three texts match. Per reading, one fixed interpretation and
  one fixed next-step sentence; no secondary numbers. New section after Limitations, placeholder until the file exists.
- Rehearsal on synthetic flags: ok (paper_conclusion exit 0, 0 overfull, 0 pending). Full suite 469 passed, 6 skipped.
  protocol_freeze / eval_freeze ok. Strength: documentation. No test output read.

## 2026-11-20T09:40Z — GPU rerun from a clean clone: rule repro-v1 fixed before the run
- The release checklist row "Reproduced in a clean environment" stays partial until a GPU train+eval is repeated from a clean clone
  (2026-11-15T15:40Z). Chosen run: arm D, seed 3407, 200 dev roots, i.e. job20 (`runs/pilot3_D_kaggle`, Kaggle T4, dev macro 0.6625 as
  recorded), because it is the trained arm the paper's question is about and its record is already on Kaggle's machine type.
- Rule `repro-v1` (`scripts/repro_compare.py`): refuse unless arm, seed, manifest, epochs, base revision, content digest, samples and
  sample order match; verdict identical (predictions sha256 equal) / reproduced (each task |diff| <= 0.03, non-edit macro |diff| <= 0.02) /
  not_reproduced. Tolerances = about twice D's largest cross-seed spread on dev (task 0.015, macro 0.00625; tested against
  `reports/pilot/seeds_dev_interim.json`). Not changed after the rerun is read; a not_reproduced verdict is reported as such.
- Order: only after jobF04-08 and jobF13 are collected (locked test first), and only on a gate/quota `ok`. Commands in RUNBOOK.
- `scripts/budget_report.py` gains group `repro` (`_jobR\d+$`, before `pilot3`); report unchanged (`--check` ok).
- Strength: engineering. Frozen evaluation code and protocol rules untouched (eval_freeze ok, protocol_freeze ok). No test output read.


## 2026-11-20T10:05Z — jobR1 dry build from a clean clone; code path equal to job20's
- Gate `next jobF04 -> wait` (-0.96 h, reset 2026-11-21T00:00Z). CPU check of the prepared rerun, nothing pushed.
- `git clone` of HEAD 703dc12 into a scratch dir, `make_kernel.py jobR1 D 200:image --from-head`: builds (arm D, seed 3407, evals
  [200 image], private, GPU + internet, id `yiitefeerdomu/trace-jobr1`); the RUNBOOK command line works as written.
- Code that job20 ran (commit 5c49d73, pushed 2026-11-13T17:56Z) vs HEAD, paths src/, scripts/colab/, configs/, manifests/,
  build_payload.py, make_kernel.py: job.sh, configs, manifests and the training/eval modules unchanged; diffs are new files
  (base_prompt.py, seeds.py, base_prompt.sh), VGCure, final_eval.sh, an arm-A branch in make_kernel.py, and an optional `system`
  message in model.py that defaults to None (old message list unchanged). So jobR1 tests the same D code path; differences in its
  result are environment/nondeterminism, which is what repro-v1 measures. Kernel image may differ (record via kernel_env.py).
- Strength: engineering check. Freezes ok. No test output read.

## 2026-11-20T11:05Z — Next-week fit check before the Kaggle reset
- Gate still `wait` (-0.96 h). Estimated remaining GPU session time: jobF04-08 17.59 h (calibrated), jobF13 <= 2.73 h, jobR1 ~1.5 h
  (3367 s GPU of job20 + 2046 s assumed setup) = ~21.8 h; with the 2 h margin ~23.8 h of the assumed 30 h weekly quota.
- Decision: no reordering needed; push in queue order F04, F05, F06, F07, F08, F13, then jobR1, one at a time on gate ok.
  If the E kernel (F07, ~8.3 h) runs longer than calibrated, jobR1 may slip to the following week; that is acceptable.
- Strength: planning estimate only. Suite exit 0; freezes ok. No test output read.

## 2026-11-20T12:55Z — Count the account's other Kaggle kernels against the weekly quota
- Found: since 2026-11-19 the owner's belirledim.tv pipeline pushes GPU kernels (`btv-tts-*`, `belirledim-ses-*`) on the same Kaggle
  account. quota.py only read TRACE logs, so the gate overestimated the remaining quota. A final kernel pushed into a quota that is
  actually exhausted could be stopped mid-run and waste its session.
- Decision: `scripts/kaggle/other_usage.py` (network, run before each gate check) lists the account's kernels and, for each non-`trace-`
  slug, downloads only its `.log` (~kB) and appends (kernel, lastRunTime as start, max "time" field as session seconds) to
  `scripts/kaggle/other_sessions.json`. quota.check adds the in-window total to `used` (reported as other_used_hours); other kernels never
  feed TRACE's per-arm estimates. Conservative: every non-TRACE kernel counts as GPU. Gap: a kernel version replaced before the script
  runs is invisible; one such version (belirledim-ses-deneme, 300 s) was added by hand from the local log the belirledim session kept.
- Measured from Kaggle logs: 0.89 h in this week's window (8 sessions, 274-628 s each). Gate now: used 29.85 h, `next jobF04 -> wait`
  (-1.85 h). Next-week plan unchanged (queue order F04-F08, F13, R1); slack for other projects ~6 h of the assumed 30 h.
- Tests: two new in tests/test_kaggle_quota.py; full suite exit 0. Frozen eval code and protocol untouched (eval_freeze ok,
  protocol_freeze ok). No test output read.

## 2026-11-20T13:30Z — Gate still waits; paper build and freezes rechecked
- other_usage.py: 0 new non-TRACE sessions (10 total). Gate `next jobF04 -> wait` (-1.85 h), reset 2026-11-21T00:00Z. Nothing pushed.
- Paper compiles with tectonic (225 kB PDF, outside repo); only the known cosmetic underfull box in setup.tex:81.
- protocol_freeze ok, eval_freeze ok, full suite exit 0. No test output read. No decision changed.

## 2026-11-20T18:55Z — Gate still waits for the weekly reset
- other_usage.py: 0 new non-TRACE sessions (10 total). Gate `next jobF04 -> wait` (-1.85 h), reset 2026-11-21T00:00Z. Nothing pushed.
- protocol_freeze ok, eval_freeze ok, full suite exit 0. No test output read. No decision changed.

## 2026-11-20T19:55Z — Gate still waits; another belirledim.tv kernel counted
- other_usage.py: 1 new non-TRACE session (btv-tts-istila-1-1579, 739 s; 11 total). quota.py: used 30.06 h incl. 1.10 h other-project.
  Gate `next jobF04 -> wait` (-2.06 h), reset 2026-11-21T00:00Z. Nothing pushed.
- Note: the owner's belirledim.tv voice jobs keep using the shared quota (~1.1 h/day so far); next week's slack (~6 h) still covers this
  rate, but if it grows, jobR1 is the first item to slip (as decided at 11:05Z).
- protocol_freeze ok, eval_freeze ok, full suite finished with an empty pytest lastfailed cache (summary line not captured; Mac was busy,
  suite took ~20 min at nice 19). No test output read. No decision changed.

## 2026-11-20T22:30Z — Gate still waits for the weekly reset
- other_usage.py: 1 new non-TRACE session (btv-tts-geri-vites-1-8778, 672 s; 12 total). Gate `next jobF04 -> wait` (-2.24 h),
  reset 2026-11-21T00:00Z. Nothing pushed. No open CPU item left that needs no test numbers (PLAN: only licence (owner) and jobR1).
- protocol_freeze ok, eval_freeze ok; quota + protocol-freeze tests pass (code unchanged, full suite not rerun). No test output read.

## 2026-11-21T08:15Z — Weekly reset passed; jobF04 pushed
- other_usage.py: 0 new non-TRACE sessions (12 total). `final_plan.py --gate`: `next jobF04; left this week 28.00 h after the 2 h margin -> ok`.
  protocol_freeze ok, eval_freeze ok (HEAD matches 4062044 eval code).
- Built `make_final_kernel.py jobF04 job12 pilot3_C --final-evaluation --from-head` (adapter pilot3_C sha256 a9f171f8…, payload sha256
  20eef398…), pushed private kernel `yiitefeerdomu/trace-jobf04` with T4 at 08:12Z; status QUEUED. Gate with `--running jobF04`: next jobF05, 25.73 h left -> ok.
- Note: Kaggle kernel slugs are lowercase (`trace-jobf04`); `kernels status trace-jobF04` returns a permission error.
- No other kernel pushed (one GPU kernel at a time). No test output read.

## 2026-11-21T09:36Z — jobF04 still queued
- `kernels status yiitefeerdomu/trace-jobf04`: QUEUED (~1 h 23 min after push). No other kernel on the account is running
  (latest btv-tts kernels COMPLETE), so the wait is Kaggle's T4 queue, not our concurrency. Accelerator not changed (would break
  environment parity of the locked-test kernels). Nothing pushed. No test output read. No decision changed.

## 2026-11-21T10:30Z — jobF04 now running
- `kernels status yiitefeerdomu/trace-jobf04`: RUNNING at 10:14Z (was QUEUED at 09:54Z). Logs are unreadable until it finishes;
  poll status only. `protocol_freeze.py`: ok; full pytest suite exit 0. Nothing pushed, no test output read, no decision changed.
- 10:28Z recheck: still RUNNING; other_usage 0 new (12 total); gate `--running jobF04`: next jobF05, 25.73 h left -> ok (waits:
  one kernel at a time). Mac on AC.

## 2026-11-21T10:56Z — jobF04 still running
- `kernels status yiitefeerdomu/trace-jobf04`: RUNNING (since ~10:14Z; arm C kernels took ~6200 s of eval before, so expect
  COMPLETE around 12:30Z). other_usage 0 new (12 total); gate `--running jobF04`: next jobF05, 25.73 h left -> ok (waits).
  protocol_freeze ok. Mac on battery (no local compute anyway). Nothing pushed, no test output read, no decision changed.

## 2026-11-21T11:20Z — jobF04 still running
- `kernels status yiitefeerdomu/trace-jobf04`: RUNNING (since ~10:14Z; expect COMPLETE around 12:30Z). other_usage 0 new (12 total);
  gate `--running jobF04`: next jobF05, 25.73 h left -> ok (waits: one kernel at a time). protocol_freeze ok. Mac on AC.
  Nothing pushed, no test output read, no decision changed.

## 2026-11-21T11:50Z — jobF04 still running
- `kernels status yiitefeerdomu/trace-jobf04`: RUNNING (since ~10:14Z; expect COMPLETE around 12:30Z). other_usage 0 new (12 total);
  gate `--running jobF04`: next jobF05, 25.73 h left -> ok (waits). protocol_freeze ok. Mac on battery. Nothing pushed, no test
  output read, no decision changed.

## 2026-11-21T12:25Z — jobF04 collected; jobF05 pushed
- jobF04 (C seed 3407) COMPLETE; collected with collect_final.py (all twelve evaluations re-scored equal to the kernel's FLAGS lines),
  budget line 5875.4 s. 8 of 12 locked-test kernels in. kernel_env: first kernel with ENV-SUMMARY, so no drift comparison possible yet.
- Gate ok (26.10 h left); eval_freeze ok. jobF05 (B seed 2027, source job14) built from HEAD with the same payload sha256 as jobF04
  and pushed 12:16Z; started RUNNING immediately. Gate with it running: next jobF06, 23.77 h left -> ok.
- No test output read.

## 2026-11-21T12:45Z — jobF05 still running
- `kernels status yiitefeerdomu/trace-jobf05`: RUNNING (since 12:16Z; B kernels ~1.5-2 h, expect COMPLETE ~14:00Z). other_usage 0 new
  (12 total); gate `--running jobF05`: next jobF06, 23.77 h left -> ok (waits: one kernel at a time). protocol_freeze ok; full pytest
  suite exit 0. Mac on battery. Nothing pushed, no test output read, no decision changed.

## 2026-11-21T12:55Z — jobF05 still running
- `kernels status yiitefeerdomu/trace-jobf05`: RUNNING (since 12:16Z; expect COMPLETE ~14:00Z). other_usage 1 new (btv-probe, 11 s;
  13 total); gate `--running jobF05`: next jobF06, 23.76 h left -> ok (waits). protocol_freeze ok. Mac on AC. Nothing pushed,
  no test output read, no decision changed.

## 2026-11-21T13:40Z: jobF07 (E, seed 2027) on the shared account

jobF07 is the longest remaining locked-test kernel (~8 h; jobF11 took 28,040 GPU-s). Running it on efeerdogmus1 in parallel with
the main-account queue (F05 -> F06 -> F08 -> F13) ends the locked test about 8 h sooner. Same procedure as seed round 9173: the
adapter (pilot3_C_desc_s2027, sha256 2a31a98e...) is the downloaded job21 output, uploaded as a private dataset and checked against
the recorded sha256 inside the kernel; payload built from HEAD. The evaluation itself is unchanged, so this changes where and when
the kernel runs, not what it measures.

## 2026-11-21T14:20Z — jobF05 collected, jobF06 pushed
- jobF05 (B seed 2027) COMPLETE; collected with `collect_final.py jobF05 --budget` (5986.6 GPU-s, 12 flags files); kernel image identical
  to jobF04. Gate `next jobF06, 23.95 h -> ok` after other_usage (0 new) and protocol_freeze ok; jobF06 built from HEAD and pushed 14:15Z.
  Payload sha256 differs from jobF04/F05 (9559cf0e... vs 20eef398...) because HEAD moved; the gate's eval-code freeze check passed,
  so the evaluated code is unchanged. No test number read.

## 2026-11-21T14:42Z — jobF06 still queued
- `kernels status yiitefeerdomu/trace-jobf06`: QUEUED (~23 min after push; jobF04 waited ~2 h in the queue, so normal). No other kernel
  on the account running (btv-render last ran 12:44Z). other_usage 0 new (14 total); gate `--running jobF06`: next jobF07, 21.46 h left
  -> ok, but jobF07 runs on the shared account, so the next autopilot push after jobF06 is jobF08. protocol_freeze ok; full pytest suite
  exit 0. Mac on AC. Nothing pushed, no test output read, no decision changed.

## 2026-11-21T14:52Z — jobF06 still queued
- `kernels status yiitefeerdomu/trace-jobf06`: QUEUED (~35 min after push). A belirledim.tv `btv-render` kernel started 14:46Z on the
  same account (no log yet, so other_usage did not record it; it will be counted once it finishes). other_usage 0 new (14 total); gate
  `--running jobF06`: next jobF07, 21.46 h left -> ok (jobF07 is on the shared account, so the next autopilot push is jobF08).
  protocol_freeze ok. Mac on AC. Nothing pushed, no test output read, no decision changed.

## 2026-11-21T15:14Z — jobF06 still queued (~1 h)
- `kernels status yiitefeerdomu/trace-jobf06`: QUEUED, ~59 min after push. `btv-render` (belirledim.tv, same account) is RUNNING
  (lastRunTime 15:02Z). Hypothesis, not verified: the queue is the account's concurrent GPU-session limit while btv-render holds a GPU.
  Autopilot does not touch other projects' kernels; it waits. Nothing pushed, no test output read, no decision changed.

## 2026-11-21T15:35Z — jobF06 still queued (~80 min)
- `kernels status yiitefeerdomu/trace-jobf06`: QUEUED. btv-render COMPLETE (other_usage: +941 s recorded). Gate ok (21.20 h left).
  No action: jobF04 also waited ~2 h. Re-pushing would only create a new version and lose the queue position; keep waiting.

## 2026-11-21T15:52Z — jobF06 still queued (~95 min)
- `kernels status yiitefeerdomu/trace-jobf06`: QUEUED. btv-render (belirledim.tv) started again 15:49Z on the same account,
  consistent with the concurrent-GPU hypothesis (unverified). Gate ok (21.20 h). Keep waiting; no re-push. No test output read.

## 2026-11-21T16:16Z — jobF06 still queued (~2 h)
- `kernels status yiitefeerdomu/trace-jobf06`: QUEUED (pushed 14:15Z). btv-render (15:49Z run) finished; other_usage recorded it
  (+938 s, 16 total). No kernel on the account is running now, so the concurrent-GPU hypothesis no longer explains the wait;
  likely Kaggle T4 queue (jobF04 also waited ~2 h). Gate `--running jobF06`: next jobF07, 20.94 h left -> ok; protocol_freeze ok.
  Keep waiting; no re-push. No test output read, no decision changed.
- 16:25Z: still QUEUED. pytest ran (exit 0 via pipe) but its summary line was not captured, so not recorded as passing this run. Mac now on battery.

## 2026-11-21T17:49Z — jobF06 queued ~3.5 h (unusual, keep waiting)
- `kernels status yiitefeerdomu/trace-jobf06`: QUEUED at 17:47Z (pushed 14:15Z). Longest earlier wait was jobF03 ~2.5 h.
  No other kernel on the main account is running (btv-render, btv-probe COMPLETE), so it is not our own concurrency limit.
- Decision: do not re-push or cancel; a new version re-enters the queue from the back and the T4 queue is Kaggle-side.
  Re-evaluate at ~6 h (20:15Z): if still QUEUED then, push a fresh version once (same payload, `--from-head`) and log it here.
- Gate `--running jobF06`: next jobF07 (shared account, skipped), 20.94 h left -> ok. protocol_freeze ok.

## 2026-11-21T18:45Z: project licence (owner's choice)

The owner chose MIT for the code and CC BY 4.0 for everything else the project produces: generated
benchmark data and images, manifests, reports, figures, paper text and the adapter weights. Files:
`LICENSE` (MIT), `LICENSE-DATA.md` (scope of CC BY 4.0 and what neither licence covers: base model,
DejaVu fonts, VGCure). Model cards, `THIRD_PARTY.md`, the datasheet and README now say so.
`pyproject.toml` is one of the eval-freeze paths, so its `license = "MIT"` field waits until jobF13 is
collected; changing it now would make the gate refuse the remaining final kernels. This supersedes
"2026-11-10: No project licence chosen yet".

## 2026-11-21T18:40Z — inventory accepts the root LICENSE
- Commit 735c01e (owner's licence choice) added a suffix-less root `LICENSE`; `third_party.facts()` classified "" as a third-party
  font file and refused it, so 4 tests in tests/test_third_party.py failed. Fix: explicit `OWN_FILES = {"LICENSE"}` (exact path), so a
  stray `src/LICENSE` is still refused (new test). scripts/third_party.py is not in eval_freeze FROZEN_PATHS nor protocol RULE_FILES.
- jobF06 QUEUED at 18:36Z (~4.4 h). Keep the 20:15Z re-push rule.

## 2026-11-21T18:45Z — jobF06 queued ~4.4 h, no change
- QUEUED at 18:39Z; no other kernel running on the main account. Rule unchanged: one re-push at ~20:15Z if still QUEUED.
- Suite: exit 0, 488 tests / 5 skipped (from progress dots). Gate ok (20.94 h). No test output read.

## 2026-11-21T19:05Z — licence statement in the paper; jobF06 queued ~4.7 h
- The paper said nothing about licences. Added a "Licences" paragraph to the reproducibility appendix, worded from LICENSE-DATA.md
  (no new choice made); `test_licences_match_files` keeps it in step with the files. Allowed bare numbers in that appendix now include
  "4.0" and "2.0" (licence versions, not results). paper/ is not an eval-freeze or protocol rule path.
- jobF06 QUEUED at 18:54Z. Re-push rule at ~20:15Z unchanged.


## 2026-11-21T19:36Z: jobF06 started, re-push rule dropped
- `kernels status yiitefeerdomu/trace-jobf06`: RUNNING at 19:34Z (QUEUED since the 14:15Z push, ~5 h). The 20:15Z one-time
  re-push from 17:49Z no longer applies. Next: collect on COMPLETE, then gate for jobF08 (skip jobF07).

## 2026-11-21T20:01Z: jobF13 (frozen base) on the shared account

jobF13 needs no adapter, so it can run on efeerdogmus1 in the free second GPU slot beside jobF07 instead of waiting behind
jobF06 and jobF08 on the main account (jobF06 alone queued ~5 h there). Built from HEAD with the same make_final_kernel.py
options as the main account plus `--owner=efeerdogmus1`; eval_freeze and protocol_freeze both ok at push time. The queue's
"runs after jobF08" order only kept whole seed rounds if the test stopped early; arm A is outside the twelve compared kernels
and final_table.py does not read it, so running it in parallel changes where and when it runs, not what it measures.

Disclosure: at 20:00Z a `tail` of runs/kaggle/jobF05/jobF05.log (looking for run times) printed one arm-B invariance-copy
summary line from the locked test before all kernels were in. No analysis, threshold or code was changed after it; every
rule the result depends on was already frozen at 4062044 (protocol_freeze ok).

## 2026-11-21T20:10Z: jobF06 running, no change
- `kernels status`: RUNNING at 19:54Z. jobF02 (arm D, seed 3407) needed 6950.8 eval GPU-seconds, so expect COMPLETE ~21:30-22:30Z.
- other_usage.py: 0 new non-TRACE sessions (16 total). Gate `--running jobF06`: next jobF07 (shared account, skipped), 20.94 h -> ok.
  protocol_freeze ok. Committed the 19:36Z state notes left uncommitted by the previous run.
- Suite: exit 0 (pytest -q -x). jobF06 still RUNNING at re-check.

## 2026-11-21T20:45Z: jobF06 running, no change
- `kernels status`: RUNNING at 20:34Z. other_usage.py: 0 new non-TRACE sessions (16 total). Gate `--running jobF06`: next jobF07
  (shared account, skipped), 20.94 h -> ok. protocol_freeze ok. No new CPU item: every PLAN item that needs no test number is done;
  remaining work (collect jobF06, push jobF08, final reports, jobR1) waits on kernels.

## 2026-11-21T21:55Z: jobF06 collected, jobF08 pushed
- jobF06 (D seed 2027) COMPLETE; collected with `collect_final.py jobF06 --budget` (6731.7 GPU-s, 12 flags files); kernel image identical
  to jobF04/F05. other_usage 0 new; protocol_freeze ok; gate `next jobF07` -> jobF07 runs on the shared account, so pushed jobF08 instead
  (21.30 h left this week). jobF08 is the last final kernel on the main account; afterwards only jobR1 (after jobF13 is collected).
  No test number read.
- 22:05Z check: jobF08 QUEUED; gate `--running jobF08` ok (19.07 h); nothing else to push on the main account.

## 2026-11-21T22:40Z: jobF08 running, no change
- `kernels status`: RUNNING at 22:34Z. other_usage 0 new (16 total). Gate `--running jobF08`: next jobF07 (shared, skipped), 16.79 h -> ok.
  protocol_freeze ok. jobF13 collected by the interactive session, so after jobF08 is collected the next main-account push is jobR1
  (clean-clone rerun of D seed 3407, RUNBOOK commands) if the gate says ok. No test number read.

## 2026-11-21T23:00Z: jobF08 running, no change
- `kernels status`: RUNNING at 22:58Z. other_usage 1 new non-TRACE session (btv-render, 537 s; 17 total). Gate `--running jobF08`:
  next jobF07 (shared, skipped), 16.64 h -> ok. protocol_freeze ok. No CPU item left that needs no test number; waiting on jobF08,
  then jobR1. No test number read.

## 2026-11-21T23:56Z: jobF08 running, no change
- `kernels status`: RUNNING at 23:54Z. other_usage 0 new (17 total). Gate `--running jobF08`: next jobF07 (shared, skipped), 16.64 h -> ok.
  protocol_freeze ok. Waiting on jobF08, then jobR1. No test number read.

## 2026-11-22T00:45Z: jobF08 collected, jobR1 pushed
- jobF08 (arm C seed 2027) COMPLETE; collected with `collect_final.py --budget` (5944 GPU-s). Kernel image identical to the other final
  kernels (kernel_env: no differing fields). All main-account locked-test kernels are in; only jobF07 (shared account) is outstanding.
- Gate and quota ok, so jobR1 (repro-v1 rerun of D seed 3407) was built from a fresh clone at 6ab4954 per RUNBOOK and pushed 00:36Z.
- budget_report.py / compute_table.py regenerated; gpu_seconds_total had lagged jobF13 and is now set from the report. No test number read.

## 2026-11-22T03:15Z: jobR1 collected, reproduction identical
- jobR1 COMPLETE; `kaggle kernels output` -> runs/kaggle/jobR1 (65 MB, gitignored); `collect.py jobR1 D --out runs/repro_D_s3407 --budget`
  (3359.5 GPU-s). `repro_compare.py runs/pilot3_D_kaggle runs/repro_D_s3407`: verdict identical (macro diff +0.0000, bit agreement
  1.0000, predictions byte-identical). Rule repro-v1 was fixed before the run, so no tolerance was needed.
- budget_report.py: 208830.3 s over 43 timed lines; gpu_seconds_total set to 208830. compute_table.py refused the new `repro` group,
  so added it to SHORT/ORDER ("Clean-clone GPU rerun (dev, arm D)") and regenerated paper/tables/compute.tex.
- Paper: one sentence in limitations ("Reproducibility across machines") stating within-environment byte-identical retraining; no
  accuracy number quoted. No test number read. Main account now has nothing left to push; jobF07 is on the shared account.
- Tests: the full suite run was stopped after 2 h (little CPU; a second autopilot session started its own full pytest in this repo at
  05:06Z on the uncommitted tree). Targeted tests for every touched file pass: budget_report, compute_table, repro_compare and all
  test_paper_* (108 passed). test_paper_limitations whitelists numbers, so the sentence names "its recorded seed" instead of 3407.
  Next run: full suite once.

## 2026-11-22T08:50Z: locked test opened; pre-registered reading

All twelve compared kernels and the frozen base collected (artifact_manifest 12 of 12 + jobF13; kernel_env jobF04-08/F13 same
image). `PYTHONPATH=src scripts/final_reports.py --out-dir reports/final` ran all eight reports in the fixed order, none refused.
Primary statistic (reports/final/final_primary.md, the only one that decides H1), pooled over seeds 3407/2027/9173:
- D - B = -0.105 [-0.117, -0.093], every seed negative, critical_edge drop > 0.02 -> reading `negative`. H1 is not supported:
  repair training lowers non-edit accuracy against QA fine-tuning (reachability about equal, critical_edge much lower).
- E - B = +0.065 [+0.048, +0.082], every seed positive -> `positive_clear` (parse with the C_desc parser, then exact solver).
- C - B = -0.224 -> `negative`. E - D = +0.171 (reported only).
No rule, threshold, split or script was changed after reading; protocol_freeze ok at 4062044. Next: pre-registered VGCure
kernel, then paper results section from reports/final (negative-result path of protocol section 5).

## 2026-11-22T09:48Z: negative-results section no longer says the test is unopened
- negative.tex's three-seed paragraph ended with "the locked test splits have not been evaluated", untrue since 08:50Z. Replaced
  by a pointer to Section test-results; the development-split numbers in that section are unchanged and still labelled as such.
  The test now requires the pointer (and forbids the old phrase) when reports/final/final_primary.json exists.
- Full suite passes (the run left open at 03:15Z). VGCure kernels jobV2/V3 QUEUED; left to the interactive session.

## 2026-11-22T10:35Z: README and negative-results log carry the locked-test reading
- README status still said "dev pilot, no locked-test result yet" (2026-11-14). Replaced with the pre-registered primary statistic
  from reports/final/final_primary.json and its readings, labelled pilot scale; dev numbers removed from the status (they stay in
  the paper's dev section). tests/test_readme_status.py fails if the README numbers drift from final_primary.json.
- docs/negative_results.md entries 1 and 2 upgraded to "solid on the dev split and on the locked test (three seeds, pilot scale)"
  with numbers copied from final_primary.json by script output (D-B primary, per-split critical-edge and reachability gaps).
- No rule, split, threshold or eval code touched (protocol_freeze / eval_freeze tests pass). VGCure jobV2/V3 left alone (QUEUED).

## 2026-11-22T10:55Z: model cards carry the locked-test evaluation
- All 13 generated cards still said "Not evaluated on the locked test splits", untrue for the 12 runs in final_queue.json since 08:50Z.
  model_cards.py now reads final_queue.json + reports/final/final_table.json: each queued run (and each arm-E parser) gets its own
  seed's per-split accuracies (repair, reachability, critical edge, non-edit macro), copied, never recomputed; parser cards state
  that the numbers are arm E's. Seed-level numbers only; pooled intervals and the reading stay in final_primary.md (one source).
- pilot3_D2 (revision 1, not in the queue) keeps the dev-only scope. A queued run missing from the table makes the script refuse.
- No eval code, rule or split touched (protocol_freeze / eval_freeze ok). VGCure jobV2/V3 left to the interactive session (QUEUED).

2026-11-22T11:45Z (autopilot): RELEASE_CHECKLIST rows marked "done for the pilot" rather than "done" for baselines and locked test: the claims are pilot-scale (one base model, synthetic drawings). The external VGCure row stays open until jobV2-V4 are collected by the interactive session; autopilot does not touch those kernels.

## 2026-11-22T12:15Z: GPU environment of the final kernels recorded
- kernel_env.py over all final kernels and jobR1, written to reports/final/kernel_env.json (reads only ENV-SUMMARY lines, no scores).
  Six kernels recorded (jobF04-08, jobF13), no differing field; eight not recorded (jobF01-03, jobF09-12 built before the probe;
  jobR1 built with make_kernel.py, which has no probe). Nothing is inferred for the unrecorded ones.
- The paper names only the recorded versions and says which kernels did not record them; the test reads the JSON, so a later
  rerun with a different image fails it instead of leaving stale text.
- docs/environment.md's 2026-11-09 access table kept as observed then, with a dated current-state note below it.


## 2026-11-22T12:40Z: locked-test style-card page restored, README points to the locked-test reports
- 09c49a9 committed reports/final/style_cards/*.png without the page that shows them (style_cards.md/json absent from disk too).
  Re-ran final_style_cards.py with the documented arguments; same eight images (byte-identical), page and JSON now tracked.
  Running it reads recorded predictions only, after the locked test was opened; no rule, split or eval code changed.
- README lists the locked-test reports; a test keeps the list and the card images from going stale.

## 2026-11-22T12:50Z: generated reading paragraph for the locked-test section; style-card paths repo-relative
- test_results.tex had tables only. scripts/test_reading.py (rule reading-v1) renders each difference row of final_primary.json
  (D-B, E-B, C-B, E-D in file order) with one fixed frame: six-cell point + interval, per-task points + intervals, seed-matched
  range, split-sign flag, and the reading label the pre-registered script recorded. Written AFTER the test was opened, so it is
  deliberately descriptive: no row choice, ordering or wording depends on a value, labels are copied, nothing is re-derived; the
  test forbids evaluative words. It does not replace finding-v1 (abstract/intro) and decides nothing.
- Full suite failed on test_release_audit: 37f73f0's style_cards.json carried 48 absolute home paths (prediction file names).
  final_style_cards.py now writes them repo-relative; re-ran with the same arguments: images and style_cards.md unchanged, JSON
  differs only in those paths. Suite: 485 passed, 17 skipped. Paper compiles (one known cosmetic underfull box).
- VGCure jobV2-V4 left to the interactive session; no eval code, rule or split touched.

## 2026-11-22T13:00Z: PLAN.md milestone table matches the repository
- M4/M6 rows still said "7 of 12 locked-test kernels collected" and "in progress". Rewritten from existing reports only:
  M4 done at pilot scale (dev seeds from seeds_dev_interim.md), M5 not run at main scale (protocol section 8 deviation 1),
  M6 done at pilot scale (numbers copied from final_primary.md, repro.md), M7 running (jobV2/V3 owned by the interactive
  session), M9 open items listed. No number recomputed; no rule, split or eval code touched.

## 2026-11-22T13:25Z: release audit rerun; shared Kaggle account slug listed for owner review
- reports/release_audit.md was from 2026-11-15 (424 files); rerun over 663 tracked files: pass (0 secrets, 0 home paths).
- The review pattern named only the main Kaggle account; the shared account slug appears in RUNBOOK/STATE/DECISIONS and was not
  listed. Pattern widened to both; it stays a review item (owner decides before upload), never a failure.

## 2026-11-22T13:45Z (autopilot) — paper build check while jobV2/V3 run
Compiled the paper (25 pages, no errors/overfull) and confirmed from the TeX log that every generated locked-test file and figure is
included; only the VGCure table remains pending by design. No change needed; nothing pushed, jobV2-V4 left to the interactive session.

## 2026-11-22T14:13Z (autopilot) — waiting on jobV4
Status-only run: jobV4 QUEUED; VGCure tests and both freeze checks pass. No action taken; VGCure pooling waits for all three seeds.

## 2026-11-22T14:36Z (autopilot) — waiting on jobV4
Status-only run on battery power: jobV4 QUEUED; both freeze checks pass. No action; VGCure pooling waits for all three seeds.

## 2026-11-22T14:55Z (autopilot) — waiting on jobV4
Status-only run on battery power: jobV4 QUEUED (pushed 13:48Z by the interactive session); both freeze checks pass. No action; VGCure pooling waits for all three seeds.

## 2026-11-22T15:15Z (autopilot) — waiting on jobV4
Status-only run on battery power: jobV4 QUEUED (~1.5 h since push); both freeze checks pass. No action; VGCure pooling waits for all three seeds.

## 2026-11-22T16:15Z (autopilot) — waiting on jobV4
Status-only run on battery power: jobV4 QUEUED (~2.5 h since push); both freeze checks pass. No action; VGCure pooling waits for all three seeds.

## 2026-11-22T16:35Z (autopilot) — waiting on jobV4
Status-only run on battery power: jobV4 QUEUED (~2.75 h since push; no other recent kernel on the main account); both freeze checks pass. No action; VGCure pooling waits for all three seeds.

## 2026-11-22T18:35Z (autopilot) — waiting on jobV4
Status-only run: jobV4 QUEUED (~4.75 h since push); both freeze checks pass; no code change since the last full-suite pass, so no rerun. No action; VGCure pooling waits for all three seeds.

## 2026-11-22T18:55Z (autopilot) — waiting on jobV4
Status-only run: jobV4 QUEUED (~5.1 h since push); both freeze checks pass; no code change since the last full-suite pass, so no rerun. No action; VGCure pooling waits for all three seeds.

## 2026-11-22T20:15Z — autopilot
Status-only run: jobV4 RUNNING (~40 min); both freeze checks pass; no code change since the last full-suite pass, so no rerun. No action; VGCure pooling waits for all three seeds.

## 2026-11-22T21:45Z — autopilot
Status-only run on battery: jobV4 RUNNING (~2.1 h); both freeze checks pass; no code change since the last full-suite pass. No action; VGCure pooling waits for all three seeds.

## 2026-11-23T06:30Z — autopilot
Status-only run on battery: jobV4 is COMPLETE on Kaggle, but it belongs to the interactive session, so autopilot did not collect or read it. Both freeze checks pass. Next: the interactive session collects jobV4, then VGCure pooling over three seeds and the pyproject license line.

## 2026-11-23T07:20Z: pre-registered VGCure subset read (jobV2-V4); pyproject stays frozen
- jobV4 (seed 9173) completed 06:28Z and was collected by the interactive session (adapters not downloaded; 4 x 1,500
  responses, prediction sha256 equal to the FLAGS-SUMMARY lines; budget line `vgcure_final_jobV4` 10,871.6 s, total
  267,597.6 s). No VGCure number was read before all three seeds were in.
- Pooled over the three locked-test seeds (`reports/external/vgcure_final_seeds.md`, records
  `vgcure_final_seeds_vs_{B,C}.json`, table `paper/tables/vgcure_final.tex`): external macro B 0.816, C 0.720,
  D 0.872, E 0.725. D - B +0.056 [+0.038, +0.075], every seed interval above zero, from both edge existence (+0.049)
  and reachability (+0.063). C - B -0.096 (reachability). E - B -0.092: the parser gave no usable graph for 27 / 36 / 46
  of 500 images, the style-OOD transcription failure named in `docs/external_eval.md`.
- No decision rule was pre-registered for VGCure, so this changes no reading: H1 stays not supported (locked test,
  D - B -0.105). Paper section external.tex states it as an observation (D's locked-test deficit is in critical edge,
  which VGCure lacks; on reachability D and B are level on the locked test and D is ahead on VGCure).
- Licence field: `pyproject.toml` is one of the eval-freeze paths, and with every kernel done the freeze now serves the
  release: HEAD's evaluation code is byte-identical to what the locked test ran (`eval_freeze.py` prints ok). Adding
  `license = "MIT"` would break that check for a metadata field, and the string form also needs setuptools >= 77 while
  the file requires >= 61. Decision: pyproject.toml is not edited; the licence is carried by `LICENSE` (MIT, included in
  wheels by setuptools' default license-files) and `LICENSE-DATA.md` (CC BY 4.0). Reverses the earlier plan line.

## 2026-11-23T07:34Z — autopilot
Status-only run on battery: both freeze checks pass; no code change since 25a98fb's targeted tests. Full suite still waits for AC power. No action.

## 2026-11-23T08:30Z — autopilot
Status-only run on battery: both freeze checks pass; no code change since 25a98fb's targeted tests. Full suite still waits for AC power. No action.

## 2026-11-23T09:53:00Z — autopilot
Status-only run on battery: both freeze checks pass; no code change since 25a98fb's targeted tests. Full suite still waits for AC power. No action.

## 2026-11-23T10:49Z — autopilot
Status-only run on battery (25%): both freeze checks pass; no code change since 25a98fb's targeted tests. Full suite still waits for AC power. No action.

## 2026-11-23T11:14Z — autopilot
Status-only run on battery (16%): both freeze checks pass; no code change since 25a98fb's targeted tests. Full suite still waits for AC power. No action.

## 2026-11-23T12:00Z — autopilot
Status-only run on battery (5%): both freeze checks pass; no code change since 25a98fb's targeted tests. Full suite still waits for AC power. No action.

## 2026-11-23T14:49Z — autopilot
On AC power: ran the owed full suite for 25a98fb (HEAD 9ef9c23, no src/scripts/tests/configs change since): 495 passed, 16 skipped, 0 failed. Both freeze checks ok. Future autopilot runs need no suite until code changes.

## 2026-11-23T18:14Z — autopilot
Status-only run on AC power: both freeze checks pass (protocol_freeze, eval_freeze ok). No src/scripts/tests/configs change since 25a98fb (only paper/main.tex co-author and docs/endorsement_request.md); full suite not rerun. No action.

## 2026-11-23T18:35Z — autopilot
Status-only run: both freeze checks pass (protocol_freeze, eval_freeze ok). No commit since a430e63; full suite not rerun. No action.

## 2026-11-23T20:30Z: owner asks to lead with the VGCure result; framing-v2 (chosen after reading)
- Owner instruction (2026-11-23): put the VGCure result first and explain why D struggles on our test and succeeds on
  VGCure. His suggested story ("VGCure first, then we tried our own test") would misstate the order: the locked test
  was opened and read first (2026-11-22) and the VGCure subset was run after it (jobV2-V4, read 2026-11-23), and the
  pre-registration names the locked test as the only hypothesis test. Not done.
- Done instead: new rule `framing-v2` (`scripts/paper_framing.py`) writes `framing_{abstract,intro,conclusion}.tex`;
  the paper shows these. They lead with both evaluations (VGCure sentence first), keep the finding-v1 locked-test
  sentence verbatim with "not supported at this scale", and say in the intro and in the new section
  `paper/sections/why.tex` that this emphasis was chosen after both results were read. finding-v1 / conclusion-v1
  outputs stay in `reports/final/` unchanged as the pre-registered record. The script refuses if any direction a
  fixed sentence states stops holding in the reports.
- New descriptive evidence: `scripts/vgcure_breakdown.py` -> `reports/external/vgcure_final_breakdown.json`
  (per item class, per source; counts and rates only). Section why.tex: on the locked test D's deficit is in critical
  edge (says true to 0.792, both twins right 228/1800 vs B 856), reachability level; on VGCure D's gain is mostly
  on absent edges and unreachable pairs (false edge claims 0.854 vs 0.711, no-path 0.894 vs 0.795), B leans to "yes".
  Explanations (removal targets are always critical edges; repair targets force drawn-vs-absent reading) are labelled
  untested; the style-OOD split does not reproduce the VGCure advantage, stated.

## 2026-11-23T21:30Z: review fixes to framing-v2 and why.tex
- Owner forwarded three review points; all checked against the sources and applied.
  1. "So the problem is not perception" was too strong: the text oracle also changes the input format, and B
     (images only) scores lower with the edge list as text (non-edit 0.750) than with the image (0.820) on the same
     100 dev roots (`reports/pilot/go_no_go_rev1.md`). why.tex and negative.tex now say "providing the correct edge
     list as text did not resolve D's critical-edge deficit"; analysis.tex no longer calls the text oracle an upper
     bound; intro softened ("help separate", limits stated).
  2. The style comparison now uses the same task: reachability D - B is -0.016 on test_style_ood and -0.004 on
     test_id (final_metrics.json) against +0.063 on VGCure (both balanced between yes and no). The repair prompt lists
     the addable edges and `validate.py` requires them to be absent from the drawing, so "repair targets force the
     model to find missing arrows" is dropped; the explanation is left as a hypothesis, with a third untested run
     (repair prompt without the addable list).
  3. Abstract: VGCure called "an adapted 500-graph VGCure evaluation" with its adaptations named; the hypothesis is
     stated as in setup.tex (repair training improves the questions more than direct QA training); E's parser named as
     a separately trained extraction adapter; setup.tex arm E now says it reads the graph with `C_desc`, not the base.
- Correction to the 20:30Z entry: the locked-test sentence in framing-v2 is reworded, not kept verbatim from
  finding-v1; numbers and reading are the same. why.tex and the paper_framing.py docstring now say so.

## 2026-11-24T13:30:35Z: autopilot check at 489423d
- protocol_freeze.py and eval_freeze.py: ok. Mac on battery and in use, so only the paper tests were run after
  the 21:30Z review-fix commit changed scripts/paper_framing.py and two test files: tests/test_paper_*.py 100 passed.
  Full suite still to run on AC. Nothing to run on Kaggle.

## 2026-11-24T00:30Z: second review round (owner forwarded), applied
1. why.tex wrongly said VGCure's reachability items are balanced: they are 409 yes / 91 no; balanced accuracy weighs
   the classes equally. why.tex and external.tex now say so and give raw accuracy from
   `vgcure_final_breakdown.json`: D 0.866, B 0.826 (+0.040), against +0.063 balanced.
2. Leftover "reasoning, not perception" claims softened in results.tex, negative.tex, related.tex and the generated
   conclusion: the strongest observation is that on the drawings E transcribes exactly D is still 0.148 below B
   (`perception_split_dev.json`), so the deficit does not come from hard-to-read drawings; the pre-registered split
   label "reasoning_gap" is reported as a label resting on E's parser. analysis.tex: the text oracle for D compares
   two checkpoints trained separately with the same recipe (`reports/pilot/oracle.md`), not one adapter.
3. Abstract gives E's external result as a number (0.092 below B on VGCure, no usable graph for 7% of images)
   instead of "fails to transcribe". framing-v2 now writes its own contribution paragraph for the conclusion (the
   conclusion-v1 one said the diagnostics "separate" perception from reasoning). Abstract 1,714 characters.

## 2026-11-24T14:07Z: autopilot check at d824856
- protocol_freeze.py and scripts/kaggle/eval_freeze.py: ok. No code change since 02df68a (paper tests 102 passed
  at 13:56Z), Mac on battery (36%), so no tests rerun. Full suite still waits for AC. Nothing to run on Kaggle.

## 2026-11-24T14:35Z: autopilot check at 1343c79
- protocol_freeze.py and scripts/kaggle/eval_freeze.py: ok. No code change since 02df68a, Mac on battery (22%),
  so no tests rerun. Full suite still waits for AC. Nothing to run on Kaggle.

## 2026-11-25T08:06Z: autopilot check at 16d6826
- protocol_freeze.py and scripts/kaggle/eval_freeze.py: ok. No code change since 02df68a, Mac on battery (68%) and in use,
  so no tests rerun. Full suite still waits for AC. Nothing to run on Kaggle.

## 2026-11-25T08:32:27Z: autopilot check at 0a35263, release snapshot identity fixed
- protocol_freeze.py and scripts/kaggle/eval_freeze.py: ok. Mac on battery (65%) and in use, so no full suite.
- Built the release snapshot from 0a35263 into a scratch folder: 673 files, private notes left out, release_audit
  pass (secrets 0, local paths 0, review 72), eval_freeze ok inside it. But the snapshot commit carried
  `<user>@<hostname>.local`: this repo has no user.email, so git used <user>@<hostname>.local, which a
  public upload would expose. release_export.py now commits with an explicit identity: `--author "Name <email>"`,
  else the anonymous default `TRACE authors <trace-authors@example.invalid>` (also fits double-blind review); it
  refuses `.local` e-mails. New tests/test_release_export.py; release/budget tests 13 passed. The public author line
  is the owner's choice at upload time.

## 2026-11-25T10:49:20Z autopilot
Freeze checks (protocol_freeze, eval_freeze) ok at d116351; no code change since f73bb3f. On battery (55%) and owner active, so the owed full suite was not run. Nothing else to do.


## 2026-11-25T10:50:47Z — autopilot

protocol_freeze.py ok, eval_freeze.py ok at b2194a1. No code change since f73bb3f; on battery (54%), owner active, so the owed full test suite still waits for AC. Nothing run.
