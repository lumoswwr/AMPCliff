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
#   2) No-DC FLaG: train the same FLaG with only k>0 available
#   3) DC-only FLaG: train the same FLaG with only k=0 available
#   4) Exact DC / Mean: parameter-free frozen-RoBERTa reference
#
# The primary fair comparison is 1/2/3: same frozen backbone, same trainable
# FLaG capacity, same optimizer/training recipe; only spectral availability
# changes. Mean is retained because it is the literal DC/global-mean readout.
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
SKIP_NODC="${SKIP_NODC:-0}"
SKIP_DCONLY="${SKIP_DCONLY:-0}"
SKIP_MEAN="${SKIP_MEAN:-0}"

echo "============================================================"
echo "Sanity check | exact spectral DC removal"
echo "============================================================"
python -u text_repro/check_exact_dc_removal.py

for SEED in ${SEEDS}; do
  if [[ "${SKIP_STSB}" != "1" && "${SKIP_NODC}" != "1" ]]; then
    echo "============================================================"
    echo "STSB | frozen FLaG | No-DC | seed ${SEED}"
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

  if [[ "${SKIP_STSB}" != "1" && "${SKIP_DCONLY}" != "1" ]]; then
    echo "============================================================"
    echo "STSB | frozen FLaG | DC-only | seed ${SEED}"
    echo "============================================================"

    python -u text_repro/train_sts.py \
      --pooling FLaG \
      --experiment_name stsb_frozen_flag_dconly \
      --seed "${SEED}" \
      --freeze_backbone \
      --dc_only \
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

  if [[ "${SKIP_SPRINT}" != "1" && "${SKIP_NODC}" != "1" ]]; then
    echo "============================================================"
    echo "Sprint | frozen FLaG | No-DC | seed ${SEED}"
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

  if [[ "${SKIP_SPRINT}" != "1" && "${SKIP_DCONLY}" != "1" ]]; then
    echo "============================================================"
    echo "Sprint | frozen FLaG | DC-only | seed ${SEED}"
    echo "============================================================"

    python -u text_repro/train_sprint.py \
      --pooling FLaG \
      --experiment_name sprint_frozen_flag_dconly \
      --seed "${SEED}" \
      --dc_only \
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

for SEED in ${SEEDS}; do
  STSB_FULL="outputs/text/stsbenchmark/experiments/stsb_frozen_flag/seed_${SEED}/metrics.json"
  STSB_NODC="outputs/text/stsbenchmark/experiments/stsb_frozen_flag_nodc/seed_${SEED}/metrics.json"
  STSB_DCONLY="outputs/text/stsbenchmark/experiments/stsb_frozen_flag_dconly/seed_${SEED}/metrics.json"
  SPRINT_FULL="outputs/text/sprintduplicatequestions/experiments/sprint_frozen_flag/seed_${SEED}/metrics.json"
  SPRINT_NODC="outputs/text/sprintduplicatequestions/experiments/sprint_frozen_flag_nodc/seed_${SEED}/metrics.json"
  SPRINT_DCONLY="outputs/text/sprintduplicatequestions/experiments/sprint_frozen_flag_dconly/seed_${SEED}/metrics.json"
  MEAN_JSON="outputs/text/dc_training_ablation/mean_dc_only_metrics.json"

  if [[ -f "${STSB_FULL}" && -f "${STSB_NODC}" && -f "${STSB_DCONLY}" \
        && -f "${SPRINT_FULL}" && -f "${SPRINT_NODC}" && -f "${SPRINT_DCONLY}" \
        && -f "${MEAN_JSON}" ]]; then
    echo "============================================================"
    echo "Summary | matched DC ablation | seed ${SEED}"
    echo "============================================================"
    python -u text_repro/summarize_dc_training_ablation.py \
      --seed "${SEED}"
  else
    echo "[summary skipped] seed ${SEED}: one or more comparison files are missing."
  fi
done

echo
echo "Done."
echo "No-DC STSB:"
echo "  outputs/text/stsbenchmark/experiments/stsb_frozen_flag_nodc/seed_*/metrics.json"
echo "No-DC Sprint:"
echo "  outputs/text/sprintduplicatequestions/experiments/sprint_frozen_flag_nodc/seed_*/metrics.json"
echo "DC-only STSB FLaG:"
echo "  outputs/text/stsbenchmark/experiments/stsb_frozen_flag_dconly/seed_*/metrics.json"
echo "DC-only Sprint FLaG:"
echo "  outputs/text/sprintduplicatequestions/experiments/sprint_frozen_flag_dconly/seed_*/metrics.json"
echo "Exact DC / Mean reference:"
echo "  outputs/text/dc_training_ablation/mean_dc_only_metrics.json"
