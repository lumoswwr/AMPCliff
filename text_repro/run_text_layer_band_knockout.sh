#!/usr/bin/env bash
set -euo pipefail

# First-pass text layer x frequency-band knockout.
#
# Uses seed 0 by default.
# STS-B: task-finetuned FLaG checkpoint, test split.
# Sprint: published frozen-backbone FLaG checkpoint, validation split first
#         because the official test is ~10x larger. After the heatmap passes
#         sanity checks, rerun with SPRINT_SPLIT=test for the final version.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${REPO_ROOT}:${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-0}"
STSB_FLAG_DIR="${STSB_FLAG_DIR:-outputs/text/stsbenchmark/experiments/stsb_frozen_flag}"
SPRINT_FLAG_DIR="${SPRINT_FLAG_DIR:-outputs/text/sprintduplicatequestions/experiments/sprint_frozen_flag}"
STSB_SPLIT="${STSB_SPLIT:-test}"
SPRINT_SPLIT="${SPRINT_SPLIT:-validation}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SKIP_STSB="${SKIP_STSB:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/text/layer_band_knockout_frozen_seed${SEED}}"

STSB_CKPT="${STSB_CKPT:-${STSB_FLAG_DIR}/seed_${SEED}/best_model.pt}"
SPRINT_CKPT="${SPRINT_CKPT:-${SPRINT_FLAG_DIR}/seed_${SEED}/best_model.pt}"

if [[ ! -f "${STSB_CKPT}" ]]; then
  echo "ERROR: STS-B checkpoint not found: ${STSB_CKPT}" >&2
  echo "Try: find outputs/text/stsbenchmark -path '*/seed_${SEED}/best_model.pt' | sort" >&2
  exit 2
fi

if [[ ! -f "${SPRINT_CKPT}" ]]; then
  echo "ERROR: Sprint checkpoint not found: ${SPRINT_CKPT}" >&2
  echo "Try: find outputs/text/sprintduplicatequestions -path '*/seed_${SEED}/best_model.pt' | sort" >&2
  exit 2
fi

echo "============================================================"
echo "Text FLaG layer x DCT-band knockout (frozen backbones)"
echo "============================================================"
echo "STS-B checkpoint : ${STSB_CKPT}"
echo "Sprint checkpoint: ${SPRINT_CKPT}"
echo "STS-B split      : ${STSB_SPLIT}"
echo "Sprint split     : ${SPRINT_SPLIT}"
echo "Bands            : 8"
echo "Preserve norm    : true (AMP-compatible)"
echo "Output           : ${OUTPUT_DIR}"
echo "Skip completed STS-B: ${SKIP_STSB}"
echo

EXTRA_ARGS=()
if [[ "${SKIP_STSB}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_stsb)
fi

python -u text_repro/probe_text_layer_band_knockout.py \
  --stsb_checkpoint "${STSB_CKPT}" \
  --sprint_checkpoint "${SPRINT_CKPT}" \
  --stsb_split "${STSB_SPLIT}" \
  --sprint_split "${SPRINT_SPLIT}" \
  --batch_size "${BATCH_SIZE}" \
  --k_bands 8 \
  --base 4 \
  --preserve_norm 1 \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
