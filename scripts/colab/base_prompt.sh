#!/usr/bin/env bash
# Frozen base (arm A) on dev under each prompt variant (src/trace_vr/base_prompt.py), long token limit,
# then the token-budget scan per variant and the pre-registered choice. Dev only; no training.
# Usage: bash scripts/colab/base_prompt.sh <job> [roots=30] [limit=768]
# Prints per variant EVAL-SUMMARY, TOKEN-BUDGET, FLAGS-SUMMARY, then one BASE-PROMPT line.
# Rerunning skips variants whose metrics exist.
set -euo pipefail
JOB=$1; ROOTS=${2:-30}; LIMIT=${3:-768}
export TRACE_HOME=${TRACE_HOME:-/content/trace} TRACE_TMP=${TRACE_TMP:-/content}
cd "$TRACE_HOME"
export PYTHONPATH=src
echo "JOB-START $JOB arm=A base_prompt roots=$ROOTS limit=$LIMIT $(date -u +%FT%TZ)"
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
for VARIANT in common direct system; do
  RUN=runs/base_prompt_${VARIANT}
  echo "VARIANT-START $VARIANT $(date -u +%FT%TZ)"
  if [ ! -f "$RUN/metrics_dev.json" ]; then
    python scripts/eval_arm.py manifests/pilot.json dev "$ROOTS" "$RUN" none "$LIMIT" image main "$VARIANT" 2>&1 | grep -v 'Loading weights'
  fi
  python scripts/token_budget.py manifests/pilot.json dev "$ROOTS" "$RUN/predictions_dev.jsonl" "$LIMIT" "$RUN/token_budget_dev.json"
  python scripts/export_flags.py manifests/pilot.json dev "$ROOTS" "$RUN" image
done
python scripts/base_prompt_choice.py runs --out runs/base_prompt_choice.json
echo "JOB-END $JOB $(date -u +%FT%TZ)"
