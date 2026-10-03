#!/usr/bin/env bash
set -euo pipefail

# Proposal 1: explicit Mean/DC safety path.
#
# z = (1-alpha) * normalize(Mean(H))
#     + alpha * normalize(FLaG(H))
#
# alpha is one learned scalar in [0,1].
# Under cosine scoring:
#   alpha=0 -> exact Mean endpoint
#   alpha=1 -> exact original FLaG endpoint
#
# STSB follows its original reproduction protocol: RoBERTa UNFROZEN.
# Sprint follows its original reproduction protocol: RoBERTa FROZEN.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

SEEDS="${SEEDS:-0}"
SKIP_STSB="${SKIP_STSB:-0}"
SKIP_SPRINT="${SKIP_SPRINT:-0}"
MEAN_MIX_INIT="${MEAN_MIX_INIT:-0.5}"

echo "============================================================"
echo "Sanity check | Mean-residual endpoints"
echo "============================================================"
python -u text_repro/check_mean_residual_endpoints.py

for SEED in ${SEEDS}; do
  if [[ "${SKIP_STSB}" != "1" ]]; then
    echo "============================================================"
    echo "STSB | UNFROZEN | FLaG_MeanResidual | seed ${SEED}"
    echo "============================================================"

    python -u text_repro/train_sts.py \
      --pooling FLaG_MeanResidual \
      --experiment_name stsb_unfrozen_flag_meanresidual \
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
      --mean_mix_init "${MEAN_MIX_INIT}"
  fi

  if [[ "${SKIP_SPRINT}" != "1" ]]; then
    echo "============================================================"
    echo "Sprint | FROZEN | FLaG_MeanResidual | seed ${SEED}"
    echo "============================================================"

    python -u text_repro/train_sprint.py \
      --pooling FLaG_MeanResidual \
      --experiment_name sprint_frozen_flag_meanresidual \
      --seed "${SEED}" \
      --epochs 10 \
      --batch_size 32 \
      --eval_batch_size 64 \
      --max_length 128 \
      --learning_rate 1e-3 \
      --weight_decay 0.01 \
      --checkpoint_metric accuracy \
      --mean_mix_init "${MEAN_MIX_INIT}"
  fi

  # Refresh the parameter-free frozen Mean reference used by Sprint summary.
  if [[ "${SKIP_SPRINT}" != "1" ]]; then
    python -u text_repro/eval_frozen_mean_dc_only.py
  fi

  python -u text_repro/summarize_mean_residual_flag.py \
    --seed "${SEED}"
done
