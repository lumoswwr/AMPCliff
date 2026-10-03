#!/usr/bin/env bash
set -euo pipefail

# Original-style STS-B fine-tuning experiment:
#   RoBERTa is UNFROZEN in every condition.
#
# Four matched conditions:
#   1) Full FLaG
#   2) No-DC FLaG: exact rFFT k=0 removed throughout train/val/test
#   3) DC-only FLaG: only exact rFFT k=0 retained throughout train/val/test
#   4) Mean: original masked mean pooling; RoBERTa is still fine-tuned
#
# This intentionally does NOT align STSB with Sprint's frozen-backbone setup.
# It matches the original STSB reproduction protocol instead.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

SEEDS="${SEEDS:-0}"

echo "============================================================"
echo "Sanity check | exact spectral controls"
echo "============================================================"
python -u text_repro/check_exact_dc_removal.py

for SEED in ${SEEDS}; do
  echo "============================================================"
  echo "STSB | UNFROZEN RoBERTa | Full FLaG | seed ${SEED}"
  echo "============================================================"
  python -u text_repro/train_sts.py \
    --pooling FLaG \
    --experiment_name stsb_unfrozen_flag_full \
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

  echo "============================================================"
  echo "STSB | UNFROZEN RoBERTa | No-DC FLaG | seed ${SEED}"
  echo "============================================================"
  python -u text_repro/train_sts.py \
    --pooling FLaG \
    --experiment_name stsb_unfrozen_flag_nodc \
    --seed "${SEED}" \
    --remove_dc \
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

  echo "============================================================"
  echo "STSB | UNFROZEN RoBERTa | DC-only FLaG | seed ${SEED}"
  echo "============================================================"
  python -u text_repro/train_sts.py \
    --pooling FLaG \
    --experiment_name stsb_unfrozen_flag_dconly \
    --seed "${SEED}" \
    --dc_only \
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

  echo "============================================================"
  echo "STSB | UNFROZEN RoBERTa | Mean | seed ${SEED}"
  echo "============================================================"
  python -u text_repro/train_sts.py \
    --pooling mean \
    --experiment_name stsb_unfrozen_mean \
    --seed "${SEED}" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_length 128 \
    --backbone_lr 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1

  python -u text_repro/summarize_stsb_unfrozen_dc_ablation.py \
    --seed "${SEED}"
done
