#!/usr/bin/env bash
# Frozen base (arm A) on dev with a long limit, then the token-budget scan (protocol items 1, 5).
# Usage: nohup bash /content/trace/scripts/colab/base_budget.sh <job> [roots=10] [limit=768] > /content/<job>.log 2>&1 &
# Prints EVAL-SUMMARY, TOKEN-BUDGET and FLAGS-SUMMARY lines. Dev only; no training.
set -euo pipefail
JOB=$1; ROOTS=${2:-10}; LIMIT=${3:-768}
RUN=runs/base_budget_${LIMIT}
cd /content/trace
export PYTHONPATH=src
echo "JOB-START $JOB base_budget roots=$ROOTS limit=$LIMIT $(date -u +%FT%TZ)"
pip -q uninstall -y torchao >/dev/null 2>&1 || true
if [ ! -f data/pilot/.built ]; then
  cp manifests/pilot.json /content/pilot_laptop.json
  python -m trace_vr generate --config configs/data_pilot.yaml | tail -2
  python - <<'PY'
import json
a, b = json.load(open('/content/pilot_laptop.json')), json.load(open('manifests/pilot.json'))
assert a['content_digest'] == b['content_digest'], 'content digest differs from the laptop build'
print('content digest identical')
PY
  touch data/pilot/.built
fi
if [ ! -f "$RUN/metrics_dev.json" ]; then
  python scripts/eval_arm.py manifests/pilot.json dev "$ROOTS" "$RUN" none "$LIMIT" image 2>&1 | grep -v 'Loading weights'
fi
python scripts/token_budget.py manifests/pilot.json dev "$ROOTS" "$RUN/predictions_dev.jsonl" "$LIMIT" "$RUN/token_budget_dev.json"
python scripts/export_flags.py manifests/pilot.json dev "$ROOTS" "$RUN" image
echo "JOB-END $JOB $(date -u +%FT%TZ)"
