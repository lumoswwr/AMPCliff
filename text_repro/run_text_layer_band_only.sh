#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${REPO_ROOT}:${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-0}"
STSB_CKPT="${STSB_CKPT:-outputs/text/stsbenchmark/experiments/stsb_frozen_flag/seed_${SEED}/best_model.pt}"
SPRINT_CKPT="${SPRINT_CKPT:-outputs/text/sprintduplicatequestions/experiments/sprint_frozen_flag/seed_${SEED}/best_model.pt"
STSB_SPLIT="${STSB_SPLIT:-test}"
SPRINT_SPLIT="${SPRINT_SPLIT:-validation}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/text/layer_band_only_frozen_seed${SEED}}"

echo "============================================================"
echo "Frozen FLaG layer x DCT band-only sufficiency"
echo "============================================================"
echo "STS-B checkpoint : ${STSB_CKPT}"
echo "Sprint checkpoint: ${SPRINT_CKPT}"
echo "STS-B split      : ${STSB_SPLIT}"
echo "Sprint split     : ${SPRINT_SPLIT}"
echo "Bands            : 8"
echo "Norm restore     : false"
echo "Output           : ${OUTPUT_DIR}"
echo

python -u text_repro/probe_text_layer_band_only.py \
  --stsb_checkpoint "${STSB_CKPT}" \
  --sprint_checkpoint "${SPRINT_CKPT}" \
  --stsb_split "${STSB_SPLIT}" \
  --sprint_split "${SPRINT_SPLIT}" \
  --batch_size "${BATCH_SIZE}" \
  --k_bands 8 \
  --base 4 \
  --output_dir "${OUTPUT_DIR}"
