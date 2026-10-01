#!/usr/bin/env bash
set -euo pipefail

# Frozen-backbone STS-B FLaG control.
#
# This keeps the original STS-B FLaG training recipe unchanged except for
# freezing pretrained RoBERTa. Only the FLaG pooling module is optimized.
#
# Original STS-B settings retained:
#   epochs=3
#   micro-batch=4
#   grad_accum=4
#   effective batch=16 sentence pairs
#   pool lr=1e-3
#   MSE on cosine similarity
#   validation Spearman checkpoint selection
#   global FLaG defaults: dropout=0.0, post_pool_norm=True

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SEEDS="${SEEDS:-0}"

for SEED in ${SEEDS}; do
  echo "============================================================"
  echo "Frozen-backbone STS-B FLaG | seed ${SEED}"
  echo "============================================================"

  python -u text_repro/train_sts.py \
    --pooling FLaG \
    --experiment_name stsb_frozen_flag \
    --seed "${SEED}" \
    --freeze_backbone \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_length 128 \
    --pool_lr 1e-3 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --pool_dropout 0.0 \
    --post_pool_norm 1
done
