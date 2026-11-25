# RUNBOOK — how an unattended run continues this project

Read by the hourly scheduled task (`trace-autopilot`). A run has no memory of earlier
conversations; this file, `PROJECT_STATE.md`, `PLAN.md`, `DECISIONS.md` and
`TRACE_ASTRA_MASTER_PROMPT.md` are the memory. Write everything you learn back into them.

## Owner's standing instructions (from chat, 2026-11-09 … 2026-11-11)

- Full authority over this project; make routine technical decisions without asking.
- Keep working whenever the computer is on. Do not stop to ask for confirmation.
- Zero extra cash. The owner's Colab account is Pro; using its existing compute units
  is allowed. Never buy, upgrade, or accept new paid terms.
- Internet is sometimes metered: stay under about 1 GB of downloads per day on this Mac.
  Do not re-download the base model (it is in `~/.cache/huggingface`).
- Do not slow the owner's Mac down while they use it (they complained twice; 2026-11-12: no local
  model/render work at all).
- Main goal (2026-11-11): a contribution useful to the AI literature, published (paper +
  released code/benchmark). Prepare everything for publication; the upload itself is
  confirmed with the owner right before it happens.
- Reply to the owner in Turkish, short and direct; code, docs and commits in English.

## Hard limits for unattended runs

- Never enter passwords, 2FA codes, or sign in anywhere. If Colab shows a Google sign-in
  page, record `blocking_user_action: "Google sign-in in the app's browser panel"` in
  `PROJECT_STATE.md` and continue with CPU work.
- Never publish anything (no public repo, no Hugging Face upload, no posts or messages).
- Never delete user files outside this repository. Never force-push or rewrite history.
- Never write numbers that were not produced by a run. Mark copied-by-hand numbers as such.
- One run at a time; if `PROJECT_STATE.md` says a job is active, check it before starting another.

## Local heavy compute (model on the Mac's GPU)

**2026-11-12: not allowed at all.** The owner games/works on the Mac and asked not to load it.
All GPU work goes to Colab/Kaggle. Local CPU work: `nice -n 19`, short commands only.
The rules below are kept for reference only.

Allowed only when ALL hold, checked right before starting and every few minutes during:
1. On AC power: `pmset -g batt | head -1` contains `AC Power`.
2. The owner is away: `ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1e9); exit}'` ≥ 900 seconds.
3. Run under `nice -n 19`. Stop the job (and save a checkpoint first if the script supports it)
   as soon as idle time drops below 60 seconds or power switches to battery.
Otherwise do CPU-only work at `nice -n 10` and keep each command short.

## Colab (GPU) work

- Notebook: https://colab.research.google.com/drive/1bKM72l7C8fc40YbiD74iAVLX0ieAGQyi (Untitled10.ipynb, owner's Drive).
- Open it in the app's built-in browser. If not signed in: see hard limits above.
- Getting code onto the VM: serve a base64 tarball from `dist/index.html` with the
  `payload` launch config, open it in the built-in browser, put it in `window.name`,
  navigate to the notebook (window.name survives), and write it into an empty cell via
  `monaco.editor.getModels()`. See git history of 2026-11-10 for the exact cells.
- Jobs: launch with `nohup bash /content/jobN.sh > /content/jobN.log 2>&1 &` from a cell,
  then re-run a short status cell that prints the filtered log. Copy every
  `TRAIN-SUMMARY` / `EVAL-SUMMARY` line into `runs/<run>/summary.json` the moment it
  appears: the VM and everything on it disappears when the runtime is recycled.
- Notebook layout (2026-11-11): cells are not in time order. Cell 0 is the first payload
  cell, **cell 1 is the status cell** (read-only: filtered `job6.log`, then runs
  `/content/export_flags_standalone.py` for every `runs/pilot3_*` that has
  `metrics_dev.json`), cell 2 is the guarded job6 launcher (does nothing if
  `/content/job6.sh` exists). Find a cell by its text, never by position on screen.
- Running a cell from automation: the page's JS coordinates are not the screenshot's
  (innerWidth 1024 vs an 800-wide frame, factor 0.78). Clicking a run button at raw JS
  coordinates hit the wrong cell once. Do this instead: `scrollIntoView` the cell, take
  the `.monaco-editor` rectangle, multiply by `800 / innerWidth`, click inside the
  editor, press cmd+Enter, then check `.cell.focused` is the cell you meant.
- Memory: the open notebook starts `pyright-langserver` processes on the VM (1.6–1.8 GB
  each, four seen at once) and nearly exhausted the 12.7 GB RAM during an evaluation.
  The status cell begins with `pkill -f 'pyright-langserve[r]'` (the bracket stops pkill
  from killing its own shell). Edit cells as little as possible, close the tab when done,
  and do not put new large base64 payloads into the notebook while a job runs.
- 2026-11-11: the status cell (cell 1) also holds a guarded job8 launcher (does nothing once
  `/content/job8.sh` exists) and prints job8 lines after `POLL7-END`, ending `POLL8-END`.
- Simplest way to run a cell (worked 2026-11-11, also with the browser pane hidden):
  `cell.querySelector('colab-run-button').click()` from page JS, wait, read `.output`.
  Keep each JS call under 45 s.
- Decode job logs with `errors='replace'` (progress bars contain cut multi-byte characters;
  `text=True` raised and showed no output).
- Per-sample results leave the VM as a `FLAGS-SUMMARY` line (bit strings plus a hash of
  the sample order and of the predictions file). Save it as `runs/<run>/flags_dev.json`;
  `scripts/bootstrap_ci.py` and `scripts/breakdown.py` work from those files alone.
  The standalone exporter is rebuilt into `dist/export_flags_standalone.py` (see
  DECISIONS 2026-11-11); paste it into a cell base64-encoded.
- Before training on Colab: `pip -q uninstall -y torchao` (peft 0.21.2 rejects Colab's 0.10.0).
- Disconnect the runtime when no job is running (Runtime → Disconnect and delete runtime).
- Budget: at most about 4 T4-hours per day. Log every run in `runs/budget.jsonl`.

## What to do in a run

1. `cd` here, `git status`, read `PROJECT_STATE.md` and the last entries of `DECISIONS.md`.
2. If a Colab job is recorded as active: try to read its log and save results. If it is
   gone, record that it was lost.
3. Pick the single most useful next step from `PROJECT_STATE.md` → `next_action` and the
   brief's section 19/20 order (diagnose before scaling; at most two evidence-based revisions
   after a failed pilot). Prefer steps that fit in one run.
4. Do it, verify it (tests: `PYTHONPATH=src .venv/bin/python -m pytest`), record results.
5. Update `PROJECT_STATE.md` (yaml block, `last_updated_utc`, `next_action`), `PLAN.md`,
   and `DECISIONS.md` for any decision. Commit with a clear message ending in
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
6. End with a 2–4 line Turkish summary of what changed and what is next.

## Useful CPU-only work while the GPU is unavailable

- (done 2026-11-11: `bootstrap.py`, `analysis.py`, `scripts/bootstrap_ci.py`, `scripts/breakdown.py`)
- (done 2026-11-12: resumable training, `train_arm.py ... [ckpt_every]`; torch tests need `.venv-model`)
- (done 2026-11-12: exposure/token accounting, `scripts/exposure.py`)
- (done 2026-11-12: topology capacity, `scripts/topology_capacity.py`; protocol item 7)
- (done 2026-11-12: `node_quota: equal`, `scripts/layout_capacity.py`, `docs/layout_capacity.md`)
- (done 2026-11-12: layout budget test, `docs/layout_budget.md`; size-OOD [13, 16])
- (done 2026-11-12: edges-per-node-matched slice, `scripts/density_slice.py`, `reports/density/`)
- (done 2026-11-12: protocol items 1 and 5 procedure, `token_budget.py`, `scripts/colab/base_budget.sh`)
- (done 2026-11-12: Kaggle result collector, `scripts/kaggle/collect.py`)
- (done 2026-11-12: invariance-pair eval path, `sample_set invariance`; paper analysis plan)
- (done 2026-11-13: final-kernel collector, `scripts/kaggle/collect_final.py <jobFxx> --budget` after `kernels output`)
- (done 2026-11-14: locked-test invariance-pair table, `scripts/final_invariance.py`, after `final_table.py`)
- (done 2026-11-14: jobF coverage in `scripts/kaggle/artifact_manifest.py`; run after every `collect_final.py`)
- (done 2026-11-14: frozen-base prompt variants + rule, `base_prompt.py`; GPU run `make_kernel.py jobA1 A 20:image`,
  then `scripts/base_prompt_choice.py runs/kaggle/jobA1/runs --out reports/pilot/base_prompt.json --out-md reports/pilot/base_prompt.md`)
- (done 2026-11-14: jobA1 collector, `scripts/kaggle/collect_base_prompt.py jobA1 --budget` after `kernels output`, then
  `base_prompt_choice.py runs --out reports/pilot/base_prompt.json --out-md reports/pilot/base_prompt.md`)
- (done 2026-11-14: twin-pair / answer-bias counts per seed, `scripts/seed_pairs.py`; same LABEL@SEED entries as seed_table.py)
- (done 2026-11-14: locked-test path for arm A, `make_final_kernel.py <job> none A --final-evaluation --base-prompt`;
  needs `reports/pilot/base_prompt.json`)
- (done 2026-11-14: jobF13 = `base` key of final_queue.json; `collect_final.py jobF13 --budget`, artifact_manifest and
  final_plan base row cover it; jobA1 collected, choice = system message at 96 tokens; setup.tex arm A item written)
- (done 2026-11-14: locked-test twin-pair / answer-bias table, `scripts/final_pairs.py`, after `final_invariance.py`)
- (done 2026-11-14: locked-test density-matched slice, `scripts/final_density.py`, after `final_pairs.py`)
- (done 2026-11-14: locked-test slice breakdown, `scripts/final_slices.py`, after `final_density.py`)
- (done 2026-11-14: locked-test per-task detail (format failures, lenient, repair detail, E extraction), `scripts/final_metrics.py`, after `final_slices.py`)
- (done 2026-11-15: all locked-test reports in order with one command, `scripts/final_reports.py --out-dir reports/final`; stops at the first refusal)
- (done 2026-11-15T06:15Z: `scripts/final_primary.py`, second in final_reports, computes the pre-registered six-cell primary statistic and its reading; read that one first, it alone decides the hypothesis)
- (done 2026-11-16: final kernels print ENV-SUMMARY (torch/CUDA/Python/image); after each `collect_final.py` run
  `scripts/kaggle/kernel_env.py jobF04 ...` over all kernels that have it; exit 1 = Kaggle image changed between kernels -> record in
  DECISIONS and docs/environment.md before the next push, do not hide it)
- (done 2026-11-16: model-agnostic `scripts/benchmark_io.py export|score`, README "Scoring your own model"; next M8 piece: CPU demo notebook)
- (done 2026-11-16: CPU verifier demo, `demo/walkthrough.py` + `notebooks/demo_verifier_cpu.ipynb`; edit cells only in
  `scripts/build_demo_notebook.py` and rerun it; next M8 piece: fixed dev gallery with recorded arm outputs, selection rule first)
- (done 2026-11-16: dev gallery of recorded outputs, `scripts/dev_gallery.py` -> `reports/pilot/gallery_dev.md`; rule `gallery-v1`, do not
  change the rule to get nicer cards; next M8 piece: report figures from recorded dev metrics, e.g. `results_dev.json`, standard plotting, SVG/PNG)
- (done 2026-11-16: dev figures, `scripts/dev_figures.py` -> `reports/pilot/figures/` + `paper/figures/arms_dev.pdf`; rerun it whenever
  `seeds_dev_interim.json` changes (`--check` tells); next M8 piece: error-type figure from the dev `summary.json` metrics, same script)
- (done 2026-11-16T11:10Z: error-type figure `errors_dev` in the same script (rule `dev-figures-v2`) + paper Figure `fig:errors-dev`;
  remaining M8 pieces need the opened locked test (OOD-drop figure, style-pair cards) or are cosmetic (edit drawn onto the gallery image))
- (done 2026-11-16T11:13Z: protocol v1.0 as executed + `scripts/protocol_freeze.py` (exit 1 = a pre-registered rule or choice file changed since the
  locked test opened: revert it); never edit `docs/protocol.md` sections 1-6; next CPU piece: make setup.tex / limitations.tex say
  "pilot scale" and name the controls not run, each sentence backed by a test)
- (done 2026-11-16T11:30Z: setup.tex "Scale", limitations.tex "Pilot scale" / "Controls not run"; when abstract, conclusion and intro finding
  are written after `final_primary.py`, they must say "pilot scale" too)
- (done 2026-11-16T12:03Z: model-free baseline table per split, `scripts/modelfree_table.py --out paper/tables/modelfree.tex`; reads only
  `reports/*/validation.json`; remaining CPU pieces before the reset are cosmetic (edit drawn onto the gallery image))
- (done 2026-11-16: third-party inventory, `scripts/third_party.py` -> `THIRD_PARTY.md`; rerun it when a pin, a font or a tracked file
  type changes (its test fails otherwise); never track anything under `external/` (VGCure states no licence))
- (done 2026-11-16T12:40Z: edit overlays on the gallery's repair cards, `dev_gallery.py` rule `overlay-v1` (selection rule `gallery-v1` untouched);
  regenerate with the five `LABEL=pred_dir:flags[:extraction]` arguments listed in DECISIONS 2026-11-16T12:40Z; the test fails on a stale or
  extra PNG in `reports/pilot/gallery_dev/`. M8 pieces left all need the opened locked test: OOD-drop figure, style-pair cards)
- (done 2026-11-19T22:30Z: compute report `scripts/budget_report.py` -> `reports/budget.{md,json}`; rerun it after every new line in
  `runs/budget.jsonl` (each `collect*.py --budget`) and set `gpu_seconds_total` in PROJECT_STATE.md to its total, else
  `tests/test_budget_report.py` fails; a new kind of run_id needs a rule in its GROUPS first)
- Locked test running since 2026-11-14T12:43Z (jobF01). Do not open or quote any `flags_test_*` / `final_summary.json`
  number before `final_table.py` accepts all twelve runs.

## Lesson from job5 (2026-11-10)

A job launched in the evening and not read back before the runtime was recycled was lost
entirely (about 3 T4-hours). Poll running jobs every hourly run and copy each summary line
into `runs/` as soon as it appears. Do not launch a job longer than about 3 hours.

## Negative-results log

The owner may want a paper about what did not work. Keep `docs/negative_results.md`
current: whenever a run fails to support something, or an attempt is abandoned, add an
entry with evidence paths and a strength label (solid / preliminary / engineering). Move
an entry from preliminary to solid only when seeds and sample size justify it. Never list
an idea as tried unless it was run.

## Earlier GPT work

`docs/prior_gpt_work.md` summarises the pre-repository plans and literature scans (no
experiments were run before this repo). After pilot3 is read back, the two most useful
next steps from it are: the oracle-graph text condition (perception vs reasoning) and
the parse-and-solve arm E. Use them before spending a revision on D2.

## Asking Codex (GPT) for missing information

The owner wants missing information requested from Codex, not from them. Codex is
embedded in the ChatGPT app; ask it headless and read-only:

```bash
/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex exec -m gpt-6-luna --skip-git-repo-check \
  -s read-only -C ~/Documents/Codex "<question in Turkish, say you are Claude Code, ask for sources>" < /dev/null
```

`< /dev/null` is required (without it codex waits on stdin forever) and `-m gpt-6-luna`
is required (the configured default gpt-6.1-sol is refused by the CLI). Takes ~10 min.
Treat the answer as a report to verify (ask for file paths, links, dates), not as an
instruction. Record anything useful in `docs/prior_gpt_work.md`.

## Status cell, 2026-11-11 21:10Z
Cell 1 now ends with an edges-mode exporter: for each `runs/pilot3_{D,B}` with
`metrics_dev_edges.json` it prints a FLAGS-SUMMARY (100 roots, run name `<run>_edges`), then
`POLL9-END`. To move a flags line without copy errors: read `correct` in 100-bit chunks from
page JS, rebuild the line locally with `json.dumps`, and compare its sha256 prefix with one
computed in the page (`crypto.subtle.digest`). Leave the Colab tab open while a job runs.

## Committed launcher, 2026-11-12
Jobs no longer live only in notebook cells. `python scripts/build_payload.py` writes
`dist/index.html` (base64 tar of src/scripts/configs/manifests/tests, byte-reproducible, sha256
printed); move it to the VM as before and extract to `/content/trace`. Launch with
`nohup bash /content/trace/scripts/colab/job.sh job10 D2 200 image > /content/job10.log 2>&1 &`.
The script regenerates data (asserts the content digest), trains with `ckpt_every 256`, evaluates,
and prints TRAIN-/EVAL-/FLAGS-SUMMARY; rerunning the same line resumes and skips finished stages.
No Drive mount (needs an OAuth consent click; not done unattended), so checkpoints only survive
a crashed process, not a recycled VM. Next: `job11 B 100 edges` (B oracle).

## Kaggle (primary GPU since 2026-11-12)
The owner supplied a Kaggle API token; it is stored only in `~/.kaggle/access_token` (mode 600,
never printed, never committed). CLI: `~/.kaggle/venv/bin/kaggle` (Kaggle CLI 2.2.5 on Python 3.12;
1.7 on the system Python 3.9 does not accept the new KGAT token). Account `yiitefeerdomu`; it can run
internet-enabled kernels, so the base model is downloaded on Kaggle's side, not the Mac's.
1. Commit, then `python scripts/kaggle/make_kernel.py <job> <arm> <roots> <mode> --from-head`
   (embeds the byte-reproducible payload; private script kernel; pins transformers 5.22.2 / peft 0.21.2).
2. `kaggle kernels push -p dist/kaggle/<job> --accelerator NvidiaTeslaT4`.
3. `kaggle kernels status|logs yiitefeerdomu/trace-<job>`; on COMPLETE
   `kaggle kernels output yiitefeerdomu/trace-<job> -p runs/kaggle/<job>` (adapter, metrics, predictions,
   `<job>.log` survive because runs/ lives in /kaggle/working). No browser, no sign-in, nothing lost
   when the Mac sleeps. Then `python scripts/kaggle/collect.py <job> <arm> [--mode M] --budget` writes
   runs/pilot3_<arm>/summary.json + flags (checked against the log) and the budget line. Free weekly GPU quota applies: before each push run `python scripts/kaggle/quota.py --running <job or omit>` and push only on `ok` (for the locked-test final kernels use `python scripts/kaggle/final_plan.py --gate [--running JOB]` instead; since 2026-11-15; the gate also prints `stop` if the frozen evaluation code changed, see `scripts/kaggle/eval_freeze.py`); one GPU kernel at a time; log runs in budget.jsonl.
Shared account (2026-11-20): the owner's other projects (belirledim.tv voice jobs, `btv-tts-*`, `belirledim-*`) push GPU
kernels on the same account and use the same weekly quota. Before every `final_plan.py --gate` / `quota.py` run
`~/.kaggle/venv/bin/python scripts/kaggle/other_usage.py` (lists the account's kernels, downloads only each non-TRACE
kernel's `.log`, appends (kernel, lastRunTime, session seconds) to `scripts/kaggle/other_sessions.json`); quota.py adds
those in the window to `used`. Never push, cancel or edit the other projects' kernels.
Kaggle machines may differ from the Colab T4 (library versions, two GPUs; the kernel uses GPU 0 only):
record the environment from the log before comparing with Colab numbers.
Note: `kaggle kernels logs` returns nothing while a script kernel is RUNNING; the log (and
`/kaggle/working/<job>.log`) is only readable after it finishes. Poll `status` (every few minutes), not logs.

- (done 2026-11-20: GPU rerun rule `repro-v1`, `scripts/repro_compare.py` (verdict identical / reproduced / not_reproduced, tolerances fixed before the run); after jobF13 is collected and only if final_plan.py --gate / quota.py says ok: `git clone ~/Documents/TRACE <scratch>/trace && cd <scratch>/trace && python3 scripts/kaggle/make_kernel.py jobR1 D 200:image --from-head`, push dist/kaggle/jobR1 from that clone, then in the repo `kaggle kernels output yiitefeerdomu/trace-jobR1 -p runs/kaggle/jobR1`, `scripts/kaggle/collect.py jobR1 D --out runs/repro_D_s3407 --budget`, `scripts/repro_compare.py runs/pilot3_D_kaggle runs/repro_D_s3407 --out reports/repro.json --out-md reports/repro.md`; budget group `repro` for `_jobR` lines)
- (done 2026-11-20: paper abstract + intro finding paragraph, `scripts/paper_finding.py`, after `result_summary.py`; refuses until final_primary.json exists)
- (done 2026-11-20: EN announcement draft + TR result summary, `scripts/result_summary.py`, after `final_style_cards.py`; refuses until final_primary.json exists; drafts only)
- (done 2026-11-20: locked-test style-pair cards, `scripts/final_style_cards.py`, after `final_figures.py`; refuses until the final table exists)

### Adapter datasets on a collaborator account (shared account efeerdogmus1)

`make_final_kernel.py ... --owner=efeerdogmus1 --adapter-dataset=efeerdogmus1/<slug>` attaches a private dataset
instead of the source kernel. Upload the adapter as `<dir>/runs/<run>/adapter/*` with
`kaggle datasets create|version -p <dir> --dir-mode zip`, then check `kaggle datasets files` lists
`<run>/adapter/adapter_model.safetensors`. Kaggle unpacks a dataset that has a single top-level folder
without that folder: uploading `<dir>/<run>/adapter/` alone gave `adapter/...` at the root, and jobF07 v1
(2026-11-21) failed its adapter check. The `runs/` wrapper keeps the run name. Token:
`~/.kaggle/access_token_shared` (never print it).

## Building the paper PDF

Tectonic 0.17.0 (official release binary, `~/.local/tectonic/tectonic`, fetched with
`gh release download 'tectonic@0.17.0' -R tectonic-typesetting/tectonic`) compiles `paper/main.tex`
in a few seconds; first run downloads its TeX bundle. Output stays outside the repo:

    cd paper && nice -n 19 ~/.local/tectonic/tectonic -X compile main.tex --outdir <scratch dir>

2026-11-13: compiles with no errors and no overfull boxes (one cosmetic underfull box in setup).
