#!/usr/bin/env bash
set -euo pipefail

# Train-without-DC vs DC-only/Mean mechanism experiment.
#
# Main question:
#   If FLaG never sees the exact DC coefficient during training, can the
#   remaining non-DC frequencies compensate?
#
# Controls:
#   1) Full frozen FLaG: existing canonical checkpoints (not retrained here)
#   2) No-DC FLaG: exact token-axis mean subtraction before FLaG at every
#      training and evaluation forward
#   3) DC-only: frozen RoBERTa + masked Mean pooling, evaluated directly
#
# Default is a seed-0 pilot. After checking the effect, run e.g.
#   SEEDS="0 1 2 3 4 5 6 7 8 9" bash text_repro/run_dc_training_ablation.sh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

SEEDS="${SEEDS:-0}"
SKIP_STSB="${SKIP_STSB:-0}"
SKIP_SPRINT="${SKIP_SPRINT:-0}"
SKIP_MEAN="${SKIP_MEAN:-0}"

for SEED in ${SEEDS}; do
  if [[ "${SKIP_STSB}" != "1" ]]; then
    echo "============================================================"
    echo "STSB | frozen FLaG | exact DC removed during training | seed ${SEED}"
    echo "============================================================"

    python -u text_repro/train_sts.py \
      --pooling FLaG \
      --experiment_name stsb_frozen_flag_nodc \
      --seed "${SEED}" \
      --freeze_backbone \
      --remove_dc \
      --epochs 3 \
      --batch_size 4 \
      --grad_accum 4 \
      --max_length 128 \
      --pool_lr 1e-3 \
      --weight_decay 0.01 \
      --warmup_ratio 0.1 \
      --pool_dropout 0.0 \
      --post_pool_norm 1
  fi

  if [[ "${SKIP_SPRINT}" != "1" ]]; then
    echo "============================================================"
    echo "Sprint | frozen FLaG | exact DC removed during training | seed ${SEED}"
    echo "============================================================"

    python -u text_repro/train_sprint.py \
      --pooling FLaG \
      --experiment_name sprint_frozen_flag_nodc \
      --seed "${SEED}" \
      --remove_dc \
      --epochs 10 \
      --batch_size 32 \
      --eval_batch_size 64 \
      --max_length 128 \
      --learning_rate 1e-3 \
      --weight_decay 0.01 \
      --checkpoint_metric accuracy
  fi
done

if [[ "${SKIP_MEAN}" != "1" ]]; then
  echo "============================================================"
  echo "Exact DC-only control | frozen RoBERTa + Mean pooling"
  echo "============================================================"

  python -u text_repro/eval_frozen_mean_dc_only.py
fi

echo
echo "Done."
echo "No-DC STSB:"
echo "  outputs/text/stsbenchmark/experiments/stsb_frozen_flag_nodc/seed_*/metrics.json"
echo "No-DC Sprint:"
echo "  outputs/text/sprintduplicatequestions/experiments/sprint_frozen_flag_nodc/seed_*/metrics.json"
echo "DC-only Mean:"
echo "  outputs/text/dc_training_ablation/mean_dc_only_metrics.json"
