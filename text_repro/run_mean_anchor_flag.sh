#!/usr/bin/env bash
set -euo pipefail

# Proposal 1.1: Mean-anchored orthogonal residual FLaG.
#
# m = normalize(Mean(H))
# f = normalize(FLaG(H))
# r = f - <f,m>m
# z = normalize(m + beta*r)
#
# beta=0 is exact Mean under cosine scoring.
# The residual cannot cancel the Mean direction.
#
# STSB: original UNFROZEN protocol, seed 0 pilot by default.
# Sprint: original FROZEN protocol, 3 seeds (0,1,2) by default.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

STSB_SEEDS="${STSB_SEEDS:-0}"
SPRINT_SEEDS="${SPRINT_SEEDS:-0 1 2}"
BETA_INIT="${BETA_INIT:-0.1}"
SKIP_STSB="${SKIP_STSB:-0}"
SKIP_SPRINT="${SKIP_SPRINT:-0}"

echo "============================================================"
echo "Sanity check | Mean-anchor residual"
echo "============================================================"
python -u text_repro/check_mean_anchor_residual.py

if [[ "${SKIP_STSB}" != "1" ]]; then
  for SEED in ${STSB_SEEDS}; do
    echo "============================================================"
    echo "STSB | UNFROZEN | FLaG_MeanAnchor | seed ${SEED}"
    echo "============================================================"

    python -u text_repro/train_sts.py \
      --pooling FLaG_MeanAnchor \
      --experiment_name stsb_unfrozen_flag_meananchor \
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
      --mean_anchor_beta_init "${BETA_INIT}"
  done
fi

if [[ "${SKIP_SPRINT}" != "1" ]]; then
  for SEED in ${SPRINT_SEEDS}; do
    echo "============================================================"
    echo "Sprint | FROZEN | FLaG_MeanAnchor | seed ${SEED}"
    echo "============================================================"

    python -u text_repro/train_sprint.py \
      --pooling FLaG_MeanAnchor \
      --experiment_name sprint_frozen_flag_meananchor \
      --seed "${SEED}" \
      --epochs 10 \
      --batch_size 32 \
      --eval_batch_size 64 \
      --max_length 128 \
      --learning_rate 1e-3 \
      --weight_decay 0.01 \
      --checkpoint_metric accuracy \
      --mean_anchor_beta_init "${BETA_INIT}"
  done
fi

python -u text_repro/summarize_mean_anchor_flag.py
