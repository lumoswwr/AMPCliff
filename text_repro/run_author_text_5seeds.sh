#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"

SEEDS_STR="${SEEDS:-0 1 2 3 4}"
TASKS_STR="${TASKS:-stsb imdb sprint}"
POOLINGS_STR="${POOLINGS:-mean FLaG FLaG_AlignmentAnchor}"
FORCE="${FORCE:-0}"

OUTPUT_DIR="outputs/text/author_protocol_5seed"

folder_for_pooling () {
  case "$1" in
    mean) echo "mean" ;;
    FLaG) echo "flag" ;;
    FLaG_AlignmentAnchor) echo "alignment" ;;
    *)
      echo "Unknown pooling: $1" >&2
      exit 2
      ;;
  esac
}

run_one () {
  local task="$1"
  local pooling="$2"
  local seed="$3"
  local folder
  folder="$(folder_for_pooling "${pooling}")"

  local metrics="${OUTPUT_DIR}/${task}/${folder}/seed_${seed}/metrics.json"

  if [[ "${FORCE}" != "1" && -f "${metrics}" ]]; then
    echo
    echo "[skip] task=${task} pooling=${pooling} seed=${seed}"
    echo "       existing: ${metrics}"
    return
  fi

  echo
  echo "======================================================================"
  echo "AUTHOR PROTOCOL | task=${task} | pooling=${pooling} | seed=${seed}"
  echo "======================================================================"

  python -u text_repro/train_author_text_protocol.py \
    --task "${task}" \
    --pooling "${pooling}" \
    --seed "${seed}" \
    --output_dir "${OUTPUT_DIR}"
}

echo "======================================================================"
echo "FINAL AUTHOR-PROTOCOL TEXT BENCHMARK"
echo "Tasks:    ${TASKS_STR}"
echo "Poolings: ${POOLINGS_STR}"
echo "Seeds:    ${SEEDS_STR}"
echo "======================================================================"
echo
echo "Protocol locked to Kewei2023/Pooling-img-text:"
echo "  IMDB : E2E, 3 epochs, maxlen 512, batch 8/16"
echo "  STSB : E2E, 3 epochs, maxlen 128, batch 32/64"
echo "  Sprint: frozen RoBERTa, 10 epochs, maxlen 128, batch 32/64"
echo "  Adam, wd=0, no scheduler, FLaG dropout=0, post-pool LayerNorm=True"
echo

for task in ${TASKS_STR}; do
  for pooling in ${POOLINGS_STR}; do
    for seed in ${SEEDS_STR}; do
      run_one "${task}" "${pooling}" "${seed}"
    done
  done
done

echo
echo "======================================================================"
echo "All requested runs complete. Summarizing..."
echo "======================================================================"

SEEDS_CSV="$(echo "${SEEDS_STR}" | tr ' ' ',')"

python -u text_repro/summarize_author_text_5seeds.py \
  --base_dir "${OUTPUT_DIR}" \
  --seeds "${SEEDS_CSV}"
