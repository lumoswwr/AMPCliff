#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"

SEED="${SEED:-0}"
TASKS="${TASKS:-stsb sprint}"
FORCE="${FORCE:-0}"

BASE_OUT="outputs/text/time_pool_ablation_seed0"

run_one () {
  local task="$1"
  local pooling="$2"
  local time_pool="$3"

  local method_dir
  case "${pooling}" in
    FLaG) method_dir="flag" ;;
    FLaG_AlignmentAnchor) method_dir="alignment" ;;
    *)
      echo "Unknown pooling: ${pooling}" >&2
      exit 2
      ;;
  esac

  local out_root="${BASE_OUT}/${time_pool}"
  local metrics="${out_root}/${task}/${method_dir}/seed_${SEED}/metrics.json"

  if [[ "${FORCE}" != "1" && -f "${metrics}" ]]; then
    echo "[skip] ${task} ${pooling} time_pool=${time_pool} seed=${SEED}"
    return
  fi

  echo
  echo "======================================================================"
  echo "TIME-POOL ABLATION | task=${task} | pooling=${pooling} | time_pool=${time_pool} | seed=${SEED}"
  echo "======================================================================"

  python -u text_repro/train_author_text_protocol.py \
    --task "${task}" \
    --pooling "${pooling}" \
    --flag_time_pool "${time_pool}" \
    --seed "${SEED}" \
    --output_dir "${out_root}"
}

echo "======================================================================"
echo "FLaG TIME-POOL ABLATION"
echo "Tasks: ${TASKS}"
echo "Seed:  ${SEED}"
echo "Variants:"
echo "  FLaG-Max"
echo "  FLaG-Mean"
echo "  Alignment-Max"
echo "  Alignment-Mean"
echo "All other settings follow the author text protocol."
echo "======================================================================"

for task in ${TASKS}; do
  run_one "${task}" "FLaG" "max"
  run_one "${task}" "FLaG" "mean"
  run_one "${task}" "FLaG_AlignmentAnchor" "max"
  run_one "${task}" "FLaG_AlignmentAnchor" "mean"
done

python -u text_repro/summarize_time_pool_ablation_seed0.py \
  --base_dir "${BASE_OUT}" \
  --reference_dir "outputs/text/author_protocol_5seed" \
  --seed "${SEED}"
