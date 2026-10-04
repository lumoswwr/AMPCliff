#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-0}"

echo "============================================================"
echo "Sanity | Alignment-aware Mean anchor"
echo "============================================================"
python -u text_repro/check_alignment_anchor.py

echo "============================================================"
echo "STSB | UNFROZEN | AlignmentAnchor | seed ${SEED}"
echo "============================================================"
python -u text_repro/train_sts.py \
  --pooling FLaG_AlignmentAnchor \
  --experiment_name stsb_unfrozen_flag_alignmentanchor \
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
  --post_pool_norm 1 \
  --mean_anchor_beta_init 0.1

echo "============================================================"
echo "Sprint | FROZEN | AlignmentAnchor | seed ${SEED}"
echo "============================================================"
python -u text_repro/train_sprint.py \
  --pooling FLaG_AlignmentAnchor \
  --experiment_name sprint_frozen_flag_alignmentanchor \
  --seed "${SEED}" \
  --epochs 10 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --max_length 128 \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --checkpoint_metric accuracy \
  --mean_anchor_beta_init 0.1

python -u text_repro/summarize_alignment_anchor.py --seed "${SEED}"
