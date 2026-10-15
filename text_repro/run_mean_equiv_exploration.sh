#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$(dirname "$ROOT"):$ROOT/text_repro${PYTHONPATH:+:$PYTHONPATH}"
TASKS="${TASKS:-stsb sprint}"
SEEDS="${SEEDS:-0}"
QUICK="${QUICK:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
FORCE="${FORCE:-0}"

echo "Mean-equivalent FLaG exploratory pilot: tasks=$TASKS seeds=$SEEDS QUICK=$QUICK"
python -u text_repro/check_flag_b2_identity.py
python -m py_compile text_repro/explore_mean_equiv_trajectory.py

args=(--tasks $TASKS --seeds $SEEDS)
if [[ "$QUICK" == "1" ]]; then
  echo "SMOKE TEST ONLY: 64 training / 256 validation pairs / 1 epoch"
  args+=(--max_epochs 1 --train_limit 64 --val_limit 256 --output_dir outputs/text/mean_equiv_smoke)
fi
if [[ "$SAVE_CHECKPOINTS" == "1" ]]; then
  args+=(--save_checkpoints)
fi
if [[ "$FORCE" == "1" ]]; then
  args+=(--force)
fi
python -u text_repro/explore_mean_equiv_trajectory.py "${args[@]}"
