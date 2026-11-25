#!/usr/bin/env bash
# Arm E: train C_desc (graph description only, same recipe as job.sh), then parse each dev image with
# that adapter and answer every question with the exact solver (scripts/eval_parse_solve.py).
# Usage: bash scripts/colab/job_e.sh <job> <eval_roots>   (TRACE_HOME / TRACE_TMP as in job.sh)
set -euo pipefail
JOB=$1; ROOTS=${2:-200}
EPOCHS=3; ACCUM=4; SEED=3407; CKPT=256; MAXNEW=512
# Seed: TRACE_SEED (pre-registered seeds only) retrains the parser at that seed; seed 3407 keeps job13's dirs.
SEED=${TRACE_SEED:-$SEED}
case "$SEED" in 3407|2027|9173) ;; *) echo "seed $SEED is not pre-registered" >&2; exit 2 ;; esac
TRAIN=runs/pilot3_C_desc; RUN=runs/pilot3_E
if [ "$SEED" != 3407 ]; then TRAIN=${TRAIN}_s${SEED}; RUN=${RUN}_s${SEED}; fi
export TRACE_HOME=${TRACE_HOME:-/content/trace} TRACE_TMP=${TRACE_TMP:-/content}
cd "$TRACE_HOME"
export PYTHONPATH=src
echo "JOB-START $JOB arm=E (C_desc parser) seed=$SEED roots=$ROOTS $(date -u +%FT%TZ)"
pip -q uninstall -y torchao >/dev/null 2>&1 || true
python -c "import peft, transformers; print('peft', peft.__version__, 'transformers', transformers.__version__)"
if [ ! -f data/pilot/.built ]; then
  cp manifests/pilot.json $TRACE_TMP/pilot_laptop.json
  python -m trace_vr generate --config configs/data_pilot.yaml | tail -2
  python - <<'PY'
import json, os
a, b = json.load(open(os.environ['TRACE_TMP'] + '/pilot_laptop.json')), json.load(open('manifests/pilot.json'))
assert a['content_digest'] == b['content_digest'], 'content digest differs from the laptop build'
print('content digest identical')
PY
  touch data/pilot/.built
fi
if [ ! -f "$TRAIN/train_summary.json" ]; then
  python scripts/train_arm.py manifests/pilot.json train C_desc "$TRAIN" $EPOCHS $ACCUM $SEED $CKPT 2>&1 | grep -v 'Loading weights'
fi
if [ ! -f "$RUN/metrics_dev.json" ]; then
  python scripts/eval_parse_solve.py manifests/pilot.json dev "$ROOTS" "$RUN" "$TRAIN/adapter" $MAXNEW 2>&1 | grep -v 'Loading weights'
fi
python scripts/export_flags.py manifests/pilot.json dev "$ROOTS" "$RUN" image
echo "JOB-END $JOB $(date -u +%FT%TZ)"
