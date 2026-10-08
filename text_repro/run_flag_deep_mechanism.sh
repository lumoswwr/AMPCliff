#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="$(dirname "${ROOT}"):${PYTHONPATH:-}"

SEED="${SEED:-0}"
TASKS="${TASKS:-stsb sprint}"
METHODS="${METHODS:-FLaG FLaG_B2}"
SPLIT="${SPLIT:-validation}"
TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
EVAL_PAIRS="${EVAL_PAIRS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

echo "[deep] Running standalone math tests before checkpoint evaluation"
python -m pytest -q tests/test_flag_deep_math.py
python -m py_compile text_repro/probe_flag_deep_mechanism.py \
  text_repro/summarize_flag_deep_mechanism.py
echo "[deep] No training; uses existing best_model.pt"
echo "[deep] tasks=${TASKS} methods=${METHODS} split=${SPLIT}"
echo "[deep] train_pairs=${TRAIN_PAIRS} eval_pairs=${EVAL_PAIRS}"

for task in ${TASKS}; do
  for method in ${METHODS}; do
    echo "[deep] ${task} ${method}"
    python -u text_repro/probe_flag_deep_mechanism.py \
      --task "${task}" --pooling "${method}" --seed "${SEED}" \
      --split "${SPLIT}" --train_pairs "${TRAIN_PAIRS}" \
      --eval_pairs "${EVAL_PAIRS}" --batch_size "${BATCH_SIZE}"
  done
done
python -u text_repro/summarize_flag_deep_mechanism.py \
  --seed "${SEED}" --split "${SPLIT}"
