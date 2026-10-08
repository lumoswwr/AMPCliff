#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="$(dirname "${ROOT}"):${PYTHONPATH:-}"

SEED="${SEED:-0}"
TASKS="${TASKS:-stsb sprint}"
METHODS="${METHODS:-FLaG FLaG_B2}"
SPLIT="${SPLIT:-validation}"
MAX_PAIRS="${MAX_PAIRS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

echo "[probe] Syntax-checking probe and summarizer"
python -m py_compile text_repro/probe_flag_gate_counterfactual.py text_repro/summarize_flag_gate_counterfactual.py

echo "[probe] Existing best_model.pt, no training"
echo "[probe] TASKS=${TASKS} METHODS=${METHODS} SPLIT=${SPLIT} MAX_PAIRS=${MAX_PAIRS}"

for task in ${TASKS}; do
  for method in ${METHODS}; do
    python -u text_repro/probe_flag_gate_counterfactual.py \
      --task "${task}" --pooling "${method}" --seed "${SEED}" \
      --split "${SPLIT}" --max_pairs "${MAX_PAIRS}" \
      --batch_size "${BATCH_SIZE}"
  done
done

python -u text_repro/summarize_flag_gate_counterfactual.py --seed "${SEED}" --split "${SPLIT}"
