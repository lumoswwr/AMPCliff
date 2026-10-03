#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-0}"

echo "============================================================"
echo "Sanity | Mean-residual + Attn-FreqGate"
echo "============================================================"
python -u text_repro/check_mean_residual_attnfreq_endpoints.py

echo "============================================================"
echo "STSB | UNFROZEN | MeanResidual + AttnFreq | seed ${SEED}"
echo "============================================================"
python -u text_repro/train_sts.py \
  --pooling FLaG_MeanResidualAttnFreq \
  --experiment_name stsb_unfrozen_flag_meanresidual_attnfreq \
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
  --mean_mix_init 0.5

echo "============================================================"
echo "Sprint | FROZEN | MeanResidual + AttnFreq | seed ${SEED}"
echo "============================================================"
python -u text_repro/train_sprint.py \
  --pooling FLaG_MeanResidualAttnFreq \
  --experiment_name sprint_frozen_flag_meanresidual_attnfreq \
  --seed "${SEED}" \
  --epochs 10 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --max_length 128 \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --checkpoint_metric accuracy \
  --mean_mix_init 0.5

python -u text_repro/summarize_mean_residual_attnfreq.py --seed "${SEED}"

echo "============================================================"
echo "Validation-only alpha sweep on the trained combined checkpoint"
echo "============================================================"
python -u text_repro/probe_mean_residual_attnfreq_alpha_sweep.py --seed "${SEED}"
