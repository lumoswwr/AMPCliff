#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-0}"
SKIP_STSB="${SKIP_STSB:-0}"
SKIP_SPRINT="${SKIP_SPRINT:-0}"

echo "============================================================"
echo "Sanity check | learned frequency scorer"
echo "============================================================"
python -u text_repro/check_learned_frequency_gate.py

if [[ "${SKIP_STSB}" != "1" ]]; then
  echo "============================================================"
  echo "STSB | UNFROZEN | FLaG_LearnedFreqGate | seed ${SEED}"
  echo "============================================================"

  python -u text_repro/train_sts.py \
    --pooling FLaG_LearnedFreqGate \
    --experiment_name stsb_unfrozen_flag_learnedfreqgate \
    --seed "${SEED}" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_length 128 \
    --backbone_lr 1e-5 \
    --pool_lr 1e-3 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --pool_dropout 0.0 \
    --post_pool_norm 1
fi

if [[ "${SKIP_SPRINT}" != "1" ]]; then
  echo "============================================================"
  echo "Sprint | FROZEN | FLaG_LearnedFreqGate | seed ${SEED}"
  echo "============================================================"

  python -u text_repro/train_sprint.py \
    --pooling FLaG_LearnedFreqGate \
    --experiment_name sprint_frozen_flag_learnedfreqgate \
    --seed "${SEED}" \
    --epochs 10 \
    --batch_size 32 \
    --eval_batch_size 64 \
    --max_length 128 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --checkpoint_metric accuracy
fi

python -u text_repro/summarize_learned_frequency_gate.py --seed "${SEED}"
