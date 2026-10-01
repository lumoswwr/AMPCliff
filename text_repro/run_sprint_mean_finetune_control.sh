#!/usr/bin/env bash
set -euo pipefail

# Sprint Mean-pooling backbone-finetuning control.
#
# This is NOT the published Sprint protocol. It is a mechanism control used to
# test whether allowing RoBERTa to adapt to Sprint causes task information to
# become increasingly readable from the DC/global-mean component.
#
# Default is a single seed first. Scale after the smoke/control result:
#   SEEDS="0 1 2" bash text_repro/run_sprint_mean_finetune_control.sh
#
# Optional overrides:
#   EPOCHS=3
#   BATCH_SIZE=8
#   EVAL_BATCH_SIZE=64
#   BACKBONE_LR=1e-5
#   HEAD_LR=1e-3

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

# train_sprint.py imports this repository as the top-level package "AMPCliff".
# Therefore Python must see the directory containing the repo.
PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
HEAD_LR="${HEAD_LR:-1e-3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sprint_mean_backbone_ft_control}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data/home/wwr_lumos/AMPCliff/outputs/text/sprintduplicatequestions}"

echo "============================================================"
echo "Sprint Mean + trainable RoBERTa mechanism control"
echo "============================================================"
echo "Seeds        : ${SEEDS}"
echo "Epochs       : ${EPOCHS}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Backbone LR  : ${BACKBONE_LR}"
echo "Head LR      : ${HEAD_LR}"
echo "Experiment   : ${EXPERIMENT_NAME}"
echo

for seed in ${SEEDS}; do
  echo
  echo "---------------- seed ${seed} ----------------"
  python -u text_repro/train_sprint.py \
    --pooling mean \
    --finetune_backbone \
    --backbone_lr "${BACKBONE_LR}" \
    --learning_rate "${HEAD_LR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --seed "${seed}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_dir "${OUTPUT_DIR}"
done

echo
echo "Finished. Checkpoints:"
find "${OUTPUT_DIR}/experiments/${EXPERIMENT_NAME}" \
  -path '*/seed_*/best_model.pt' -type f | sort -V
