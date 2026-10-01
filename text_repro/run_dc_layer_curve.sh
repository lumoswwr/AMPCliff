#!/usr/bin/env bash
set -euo pipefail

# Layer-wise DC-only probe for STS-B + Sprint.
#
# Usage:
#   STSB_MEAN_DIR=/path/to/mean_experiment \
#     bash text_repro/run_dc_layer_curve.sh
#
# STSB_MEAN_DIR must contain seed_*/best_model.pt from train_sts.py.
# Example:
#   outputs/text/stsbenchmark/experiments/<mean_experiment_name>
#
# Optional overrides:
#   SPLIT=test
#   BATCH_SIZE=64
#   OUTPUT_DIR=/home/data/home/wwr_lumos/AMPCliff/outputs/text/dc_layer_curve

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

STSB_MEAN_DIR="${STSB_MEAN_DIR:-}"
if [[ -z "${STSB_MEAN_DIR}" ]]; then
  echo "ERROR: STSB_MEAN_DIR is required." >&2
  echo "Find candidate Mean checkpoints with:" >&2
  echo "  find outputs/text/stsbenchmark -path '*/seed_*/best_model.pt' | sort" >&2
  exit 2
fi

if [[ ! -d "${STSB_MEAN_DIR}" ]]; then
  echo "ERROR: directory not found: ${STSB_MEAN_DIR}" >&2
  exit 2
fi

mapfile -t STSB_CKPTS < <(
  find "${STSB_MEAN_DIR}" -maxdepth 2 -type f -path '*/seed_*/best_model.pt' | sort -V
)

if [[ "${#STSB_CKPTS[@]}" -eq 0 ]]; then
  echo "ERROR: no seed_*/best_model.pt found under ${STSB_MEAN_DIR}" >&2
  exit 2
fi

# Guard against accidentally probing a non-Mean STS-B experiment.
for ckpt in "${STSB_CKPTS[@]}"; do
  cfg="$(dirname "${ckpt}")/config.json"
  if [[ -f "${cfg}" ]]; then
    pooling="$(python - "${cfg}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    print(json.load(f).get("pooling", ""))
PY
)"
    if [[ "${pooling}" != "mean" ]]; then
      echo "ERROR: ${cfg} says pooling=${pooling}; expected mean." >&2
      exit 2
    fi
  fi
done

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_PATH="${MODEL_PATH:-/home/data/home/wwr_lumos/models/roberta-base}"
STSB_DATA_PATH="${STSB_DATA_PATH:-/home/data/home/wwr_lumos/AMPCliff/data/text/stsbenchmark}"
SPRINT_DATA_PATH="${SPRINT_DATA_PATH:-/home/data/home/wwr_lumos/AMPCliff/data/text/sprintduplicatequestions_adaptation}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data/home/wwr_lumos/AMPCliff/outputs/text/dc_layer_curve}"

echo "============================================================"
echo "Layer-wise DC-only probe"
echo "============================================================"
echo "STSB Mean dir : ${STSB_MEAN_DIR}"
echo "Checkpoints   : ${#STSB_CKPTS[@]}"
printf '  %s\n' "${STSB_CKPTS[@]}"
echo "Split         : ${SPLIT}"
echo "Output        : ${OUTPUT_DIR}"
echo

python -u text_repro/probe_dc_layer_curve.py \
  --stsb_checkpoints "${STSB_CKPTS[@]}" \
  --model_path "${MODEL_PATH}" \
  --stsb_data_path "${STSB_DATA_PATH}" \
  --sprint_data_path "${SPRINT_DATA_PATH}" \
  --split "${SPLIT}" \
  --batch_size "${BATCH_SIZE}" \
  --output_dir "${OUTPUT_DIR}"
