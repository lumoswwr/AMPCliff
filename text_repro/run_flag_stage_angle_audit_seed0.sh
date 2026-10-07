#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"

SEED="${SEED:-0}"
TASKS="${TASKS:-stsb sprint}"
BASE_OUT="outputs/text/flag_stage_angle_audit_seed0"

echo "================================================================================"
echo "FLaG STAGE-WISE ANGLE AUDIT"
echo "Tasks: ${TASKS}"
echo "Seed:  ${SEED}"
echo
echo "Matched settings:"
echo "  time_pool = mean"
echo "  post_pool_norm = false"
echo "  dropout = 0"
echo
echo "Variants:"
echo "  FLaG-Mean : residual_sigmoid + random gate + random projection"
echo "  FLaG-A1   : residual_sigmoid + random gate + identity projection"
echo "  FLaG-B1   : centered_sigmoid + random gate + identity projection"
echo "  FLaG-zero : centered_sigmoid + zero-init gate + identity projection"
echo
echo "Audit checkpoints: epoch0, after epoch1, selected best checkpoint"
echo "Angles are measured on the validation split."
echo "================================================================================"

echo
echo "[sanity] Checking strict B2 Mean-equivalent initialization..."
python -u text_repro/check_flag_b2_identity.py

run_one () {
  local task="$1"
  local pooling="$2"

  echo
  echo "================================================================================"
  echo "ANGLE AUDIT | task=${task} | pooling=${pooling} | seed=${SEED}"
  echo "================================================================================"

  python -u text_repro/train_author_text_protocol.py \
    --task "${task}" \
    --pooling "${pooling}" \
    --flag_time_pool mean \
    --disable_post_pool_norm \
    --stage_angle_audit \
    --seed "${SEED}" \
    --output_dir "${BASE_OUT}"
}

for task in ${TASKS}; do
  run_one "${task}" "FLaG"
  run_one "${task}" "FLaG_A1"
  run_one "${task}" "FLaG_B1"
  run_one "${task}" "FLaG_B2"
done

python -u text_repro/summarize_flag_stage_angle_audit_seed0.py \
  --base_dir "${BASE_OUT}" \
  --seed "${SEED}"
