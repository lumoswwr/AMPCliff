#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"

SEED="${SEED:-0}"
TASKS="${TASKS:-stsb sprint}"
FORCE="${FORCE:-0}"
BASE_OUT="outputs/text/flag_init_ablation_seed0"

echo "======================================================================"
echo "FLaG INITIALIZATION ABLATION"
echo "Tasks: ${TASKS}"
echo "Seed:  ${SEED}"
echo
echo "All FLaG variants use:"
echo "  time_pool = mean"
echo "  post_pool_norm = false"
echo "  dropout = 0"
echo
echo "Variants:"
echo "  FLaG-Mean : original random gate + random output projection"
echo "  FLaG-A1   : original random gate + identity-initialized output projection"
echo "  FLaG-A1Z  : zero-centered identity gate init + identity output projection"
echo "======================================================================"

echo
echo "[1/2] Verifying that A1Z is Mean-equivalent at initialization..."
python -u text_repro/check_flag_a1z_identity.py

run_one () {
  local task="$1"
  local pooling="$2"

  local folder
  case "${pooling}" in
    FLaG) folder="flag" ;;
    FLaG_A1) folder="flag_a1" ;;
    FLaG_A1Z) folder="flag_a1z" ;;
    *)
      echo "Unknown pooling: ${pooling}" >&2
      exit 2
      ;;
  esac

  local metrics="${BASE_OUT}/${task}/${folder}/seed_${SEED}/metrics.json"
  if [[ "${FORCE}" != "1" && -f "${metrics}" ]]; then
    echo "[skip] task=${task} pooling=${pooling} seed=${SEED}"
    return
  fi

  echo
  echo "======================================================================"
  echo "INIT ABLATION | task=${task} | pooling=${pooling} | seed=${SEED}"
  echo "======================================================================"

  python -u text_repro/train_author_text_protocol.py \
    --task "${task}" \
    --pooling "${pooling}" \
    --flag_time_pool mean \
    --disable_post_pool_norm \
    --seed "${SEED}" \
    --output_dir "${BASE_OUT}"
}

echo
echo "[2/2] Training matched variants..."

for task in ${TASKS}; do
  run_one "${task}" "FLaG"
  run_one "${task}" "FLaG_A1"
  run_one "${task}" "FLaG_A1Z"
done

python -u text_repro/summarize_flag_init_ablation_seed0.py \
  --base_dir "${BASE_OUT}" \
  --reference_dir "outputs/text/author_protocol_5seed" \
  --seed "${SEED}"
