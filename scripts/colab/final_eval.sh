#!/usr/bin/env bash
# FINAL evaluation of one trained run on the locked test splits (brief section 19, M6).
# Run only once every design decision is frozen on dev; results must not feed back into choices.
# Usage (from a cell, after job.sh has trained runs/<run>/adapter on the same VM):
#   nohup bash /content/trace/scripts/colab/final_eval.sh <job> <run> --final-evaluation \
#     > /content/<job>.log 2>&1 &
# Regenerates pilot (needed for the isomorphism exclusion), pilot_test and pilot_diag (family-only
# and size-only OOD, protocol open item 2), asserts every content digest matches the committed
# manifest, then evaluates every locked split in image mode on all of its roots (main samples, then the
# relabelled/restyled invariance copies of the same roots) and prints EVAL-SUMMARY and FLAGS-SUMMARY per split. Finished splits are skipped.
# Arm E (parse-and-solve): pass --parse-solve=<parser_run> as the fourth argument, e.g.
#   final_eval.sh jobF5 pilot3_E --final-evaluation --parse-solve=pilot3_C_desc
# The parser adapter is runs/<parser_run>/adapter; each image is transcribed once with scripts/eval_parse_solve.py
# (max_new 512, as job_e.sh on dev) and the exact solver answers; outputs go to runs/<run>.
# Arm A (frozen base, no adapter): pass --base-prompt=<variant>:<limit> with <run> = base_prompt_<variant>, e.g.
#   final_eval.sh jobF13 base_prompt_direct --final-evaluation --base-prompt=direct:96
# <variant> and <limit> are the dev choice in reports/pilot/base_prompt.json (make_final_kernel.py reads them from
# that file); the prompt variant is passed to eval_arm.py, which refuses a variant together with an adapter.
set -euo pipefail
JOB=$1; RUN=runs/$2; CONFIRM=${3:-}; MODE=${4:-}
[ "$CONFIRM" = "--final-evaluation" ] || { echo "refusing: pass --final-evaluation as the third argument"; exit 2; }
MAXNEW=96; MAXNEW_E=512; ADAPTER=$RUN/adapter; PARSE_SOLVE=0; VARIANT=common
case "$MODE" in
  "") ;;
  --parse-solve=pilot3_*) PARSE_SOLVE=1; ADAPTER=runs/${MODE#--parse-solve=}/adapter ;;
  --base-prompt=*:*)
    SPEC=${MODE#--base-prompt=}; VARIANT=${SPEC%%:*}; MAXNEW=${SPEC##*:}; ADAPTER=none
    case "$VARIANT" in common|direct|system) ;; *) echo "unknown base prompt variant $VARIANT"; exit 2 ;; esac
    case "$MAXNEW" in ''|*[!0-9]*) echo "token limit must be a number, got $MAXNEW"; exit 2 ;; esac
    [ "$2" = "base_prompt_$VARIANT" ] || { echo "run must be base_prompt_$VARIANT for --base-prompt=$VARIANT"; exit 2; } ;;
  *) echo "unknown fourth argument $MODE"; exit 2 ;;
esac
SPLITS="pilot_test:test_id:200 pilot_test:test_style_ood:200 pilot_test:test_struct_ood:200 pilot_test:test_combined_ood:100 pilot_diag:test_family_ood:100 pilot_diag:test_size_ood:100"
# TRACE_HOME / TRACE_TMP let the same script run on Kaggle (scripts/kaggle/make_final_kernel.py).
export TRACE_HOME=${TRACE_HOME:-/content/trace} TRACE_TMP=${TRACE_TMP:-/content}
cd "$TRACE_HOME"
export PYTHONPATH=src TRACE_UNLOCK_TEST=final-evaluation
echo "FINAL-START $JOB run=$RUN adapter=$ADAPTER parse_solve=$PARSE_SOLVE prompt_variant=$VARIANT max_new=$MAXNEW $(date -u +%FT%TZ)"
[ "$ADAPTER" = none ] || [ -d "$ADAPTER" ] || { echo "missing $ADAPTER (train with job.sh / job_e.sh first)"; exit 3; }
pip -q uninstall -y torchao >/dev/null 2>&1 || true
for NAME in pilot pilot_test pilot_diag; do
  CFG=configs/data_${NAME}.yaml
  if [ ! -f data/$NAME/.built ]; then
    cp manifests/$NAME.json $TRACE_TMP/${NAME}_laptop.json
    python -m trace_vr generate --config $CFG | tail -2
    python - "$NAME" <<'PY'
import json, os, sys
n = sys.argv[1]
a, b = json.load(open(os.environ['TRACE_TMP'] + '/%s_laptop.json' % n)), json.load(open('manifests/%s.json' % n))
assert a['content_digest'] == b['content_digest'], '%s content digest differs from the laptop build' % n
print(n, 'content digest identical')
PY
    touch data/$NAME/.built
  fi
done
for ITEM in $SPLITS; do
  MAN=manifests/${ITEM%%:*}.json; REST=${ITEM#*:}; SPLIT=${REST%%:*}; ROOTS=${REST##*:}
  if [ ! -f "$RUN/metrics_${SPLIT}.json" ]; then
    if [ $PARSE_SOLVE = 1 ]; then
      python scripts/eval_parse_solve.py "$MAN" "$SPLIT" "$ROOTS" "$RUN" "$ADAPTER" $MAXNEW_E main 2>&1 | grep -v 'Loading weights'
    else
      python scripts/eval_arm.py "$MAN" "$SPLIT" "$ROOTS" "$RUN" "$ADAPTER" $MAXNEW image main "$VARIANT" 2>&1 | grep -v 'Loading weights'
    fi
  fi
  python scripts/export_flags.py "$MAN" "$SPLIT" "$ROOTS" "$RUN" image
  # invariance pairs (relabelled/restyled copies of the same roots; partners are in the main run)
  if [ ! -f "$RUN/metrics_${SPLIT}_inv.json" ]; then
    if [ $PARSE_SOLVE = 1 ]; then
      python scripts/eval_parse_solve.py "$MAN" "$SPLIT" "$ROOTS" "$RUN" "$ADAPTER" $MAXNEW_E invariance 2>&1 | grep -v 'Loading weights'
    else
      python scripts/eval_arm.py "$MAN" "$SPLIT" "$ROOTS" "$RUN" "$ADAPTER" $MAXNEW image invariance "$VARIANT" 2>&1 | grep -v 'Loading weights'
    fi
  fi
  python scripts/export_flags.py "$MAN" "$SPLIT" "$ROOTS" "$RUN" image invariance
done
echo "FINAL-END $JOB $(date -u +%FT%TZ)"
