#!/usr/bin/env bash
set -euo pipefail

# Controls for layer-wise DC-only probe.
#
# Required:
#   STSB_MEAN_DIR=...  directory containing seed_*/best_model.pt
#   STSB_FLAG_DIR=...  directory containing seed_*/best_model.pt
#
# Example:
#   STSB_MEAN_DIR=outputs/text/stsbenchmark/mean \
#   STSB_FLAG_DIR=outputs/text/stsbenchmark/experiments/<flag-run> \
#   bash text_repro/run_dc_layer_controls.sh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

STSB_MEAN_DIR="${STSB_MEAN_DIR:-}"
STSB_FLAG_DIR="${STSB_FLAG_DIR:-}"

if [[ -z "${STSB_MEAN_DIR}" || -z "${STSB_FLAG_DIR}" ]]; then
  echo "ERROR: STSB_MEAN_DIR and STSB_FLAG_DIR are both required." >&2
  echo >&2
  echo "Mean candidates:" >&2
  find outputs/text/stsbenchmark -path '*/seed_*/config.json' -print0 2>/dev/null |
    while IFS= read -r -d '' f; do
      grep -q '"pooling": "mean"' "$f" && dirname "$(dirname "$f")"
    done | sort -u >&2 || true
  echo >&2
  echo "FLaG candidates:" >&2
  find outputs/text/stsbenchmark -path '*/seed_*/config.json' -print0 2>/dev/null |
    while IFS= read -r -d '' f; do
      grep -q '"pooling": "FLaG"' "$f" && dirname "$(dirname "$f")"
    done | sort -u >&2 || true
  exit 2
fi

collect_ckpts() {
  local root="$1"
  find "${root}" -maxdepth 2 -type f -path '*/seed_*/best_model.pt' | sort -V
}

mapfile -t MEAN_CKPTS < <(collect_ckpts "${STSB_MEAN_DIR}")
mapfile -t FLAG_CKPTS < <(collect_ckpts "${STSB_FLAG_DIR}")

if [[ "${#MEAN_CKPTS[@]}" -eq 0 ]]; then
  echo "ERROR: no Mean checkpoints found under ${STSB_MEAN_DIR}" >&2
  exit 2
fi

if [[ "${#FLAG_CKPTS[@]}" -eq 0 ]]; then
  echo "ERROR: no FLaG checkpoints found under ${STSB_FLAG_DIR}" >&2
  exit 2
fi

check_pooling() {
  local expected="$1"
  shift

  for ckpt in "$@"; do
    local cfg
    cfg="$(dirname "${ckpt}")/config.json"

    if [[ ! -f "${cfg}" ]]; then
      echo "WARNING: missing config for ${ckpt}; cannot verify pooling." >&2
      continue
    fi

    local pooling
    pooling="$(python - "${cfg}" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    print(json.load(f).get("pooling", ""))
PY
)"

    if [[ "${pooling}" != "${expected}" ]]; then
      echo "ERROR: ${cfg} says pooling=${pooling}; expected ${expected}." >&2
      exit 2
    fi
  done
}

check_pooling "mean" "${MEAN_CKPTS[@]}"
check_pooling "FLaG" "${FLAG_CKPTS[@]}"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_PATH="${MODEL_PATH:-/home/data/home/wwr_lumos/models/roberta-base}"
STSB_DATA_PATH="${STSB_DATA_PATH:-/home/data/home/wwr_lumos/AMPCliff/data/text/stsbenchmark}"
SPRINT_DATA_PATH="${SPRINT_DATA_PATH:-/home/data/home/wwr_lumos/AMPCliff/data/text/sprintduplicatequestions_adaptation}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data/home/wwr_lumos/AMPCliff/outputs/text/dc_layer_controls}"

echo "============================================================"
echo "Layer-wise DC controls"
echo "============================================================"
echo "Mean dir       : ${STSB_MEAN_DIR}"
echo "Mean ckpts     : ${#MEAN_CKPTS[@]}"
echo "FLaG dir       : ${STSB_FLAG_DIR}"
echo "FLaG ckpts     : ${#FLAG_CKPTS[@]}"
echo "Split          : ${SPLIT}"
echo "Output         : ${OUTPUT_DIR}"
echo

python -u text_repro/probe_dc_layer_controls.py \
  --stsb_mean_checkpoints "${MEAN_CKPTS[@]}" \
  --stsb_flag_checkpoints "${FLAG_CKPTS[@]}" \
  --model_path "${MODEL_PATH}" \
  --stsb_data_path "${STSB_DATA_PATH}" \
  --sprint_data_path "${SPRINT_DATA_PATH}" \
  --split "${SPLIT}" \
  --batch_size "${BATCH_SIZE}" \
  --output_dir "${OUTPUT_DIR}"
