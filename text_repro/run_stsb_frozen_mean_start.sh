#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$(dirname "$ROOT"):$ROOT/text_repro${PYTHONPATH:+:$PYTHONPATH}"

SEED="${SEED:-0}"
QUICK="${QUICK:-0}"
FORCE="${FORCE:-0}"
MEAN_CHECKPOINT="${MEAN_CHECKPOINT:-}"

python -m py_compile text_repro/train_stsb_frozen_mean_start.py
python -u text_repro/check_flag_b2_identity.py

args=(--seed "$SEED")
if [[ "$QUICK" == "1" ]]; then
    echo "[SMOKE ONLY] 64 training pairs, 256 val pairs, 1+1 epochs"
    args+=(--reference_epochs 1 --flag_epochs 1
           --train_limit 64 --val_limit 256
           --min_reference_spearman -1
           --output_dir outputs/text/stsb_frozen_trained_mean_smoke)
fi
if [[ -n "$MEAN_CHECKPOINT" ]]; then
    args+=(--mean_checkpoint "$MEAN_CHECKPOINT")
fi
if [[ "$FORCE" == "1" ]]; then
    args+=(--force)
fi
python -u text_repro/train_stsb_frozen_mean_start.py "${args[@]}"
