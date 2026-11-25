#!/usr/bin/env bash
# One Colab job: train one arm (resumable) and evaluate it, printing TRAIN-/EVAL-/FLAGS-SUMMARY lines.
# Usage (from a cell): nohup bash /content/trace/scripts/colab/job.sh <job> <arm> <eval_roots> <input_mode> \
#                        > /content/<job>.log 2>&1 &
# Rerunning the same command after a crash resumes training from runs/<run>/ckpt (same VM only:
# Drive is not mounted, see DECISIONS 2026-11-12). Each stage is skipped if its output exists.
set -euo pipefail
JOB=$1; ARM=$2; ROOTS=${3:-200}; MODE=${4:-image}
EPOCHS=3; ACCUM=4; SEED=3407; CKPT=256; MAXNEW=96
# Training seed: TRACE_SEED overrides the default, but only with a seed pre-registered in
# docs/protocol.md (3407, 2027, 9173). The default seed keeps the old run name.
SEED=${TRACE_SEED:-$SEED}
case "$SEED" in 3407|2027|9173) ;; *) echo "seed $SEED is not pre-registered" >&2; exit 2 ;; esac
RUN=runs/pilot3_${ARM}; [ "$SEED" != 3407 ] && RUN=runs/pilot3_${ARM}_s${SEED}
# TRACE_HOME / TRACE_TMP let the same script run on Kaggle (scripts/kaggle/make_kernel.py).
export TRACE_HOME=${TRACE_HOME:-/content/trace} TRACE_TMP=${TRACE_TMP:-/content}
cd "$TRACE_HOME"
export PYTHONPATH=src
echo "JOB-START $JOB arm=$ARM seed=$SEED roots=$ROOTS mode=$MODE $(date -u +%FT%TZ)"
pip -q uninstall -y torchao >/dev/null 2>&1 || true
python -c "import peft, transformers; print('peft', peft.__version__, 'transformers', transformers.__version__)"
if [ ! -f data/pilot/.built ]; then
  cp manifests/pilot.json $TRACE_TMP/pilot_laptop.json
  python -m trace_vr generate --config configs/data_pilot.yaml | tail -2
  python - <<'PY'
import json
import os
a, b = json.load(open(os.environ.get('TRACE_TMP', '/content') + '/pilot_laptop.json')), json.load(open('manifests/pilot.json'))
assert a['content_digest'] == b['content_digest'], 'content digest differs from the laptop build'
print('content digest identical')
PY
  touch data/pilot/.built
fi
if [ ! -f "$RUN/train_summary.json" ]; then
  python scripts/train_arm.py manifests/pilot.json train "$ARM" "$RUN" $EPOCHS $ACCUM $SEED $CKPT 2>&1 | grep -v 'Loading weights'
else
  echo "TRAIN-SUMMARY $(python -c "import json;print(json.dumps(json.load(open('$RUN/train_summary.json'))))")"
fi
SUFFIX=dev; [ "$MODE" != image ] && SUFFIX="dev_${MODE//+/_}"
if [ ! -f "$RUN/metrics_${SUFFIX}.json" ]; then
  python scripts/eval_arm.py manifests/pilot.json dev "$ROOTS" "$RUN" "$RUN/adapter" $MAXNEW "$MODE" 2>&1 | grep -v 'Loading weights'
fi
python scripts/export_flags.py manifests/pilot.json dev "$ROOTS" "$RUN" "$MODE"
echo "JOB-END $JOB $(date -u +%FT%TZ)"
