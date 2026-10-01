#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-sprint_mean_backbone_ft_control}"
ROOT="${ROOT:-/home/data/home/wwr_lumos/AMPCliff/outputs/text/sprintduplicatequestions/experiments/${EXPERIMENT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data/home/wwr_lumos/AMPCliff/outputs/text/sprint_dc_finetune_control}"

mapfile -t CKPTS < <(
  find "${ROOT}" -maxdepth 2 -type f -path '*/seed_*/best_model.pt' | sort -V
)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "ERROR: no checkpoints found under ${ROOT}" >&2
  exit 2
fi

echo "Using ${#CKPTS[@]} checkpoint(s):"
printf '  %s\n' "${CKPTS[@]}"

python -u text_repro/probe_sprint_dc_finetune_control.py \
  --checkpoints "${CKPTS[@]}" \
  --output_dir "${OUTPUT_DIR}"
