#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="$(dirname "${ROOT}"):${PYTHONPATH:-}"

SEEDS="${SEEDS:-0}"
TASKS="${TASKS:-stsb sprint}"
METHODS="${METHODS:-StaticFLaG_ReIm_B2 StaticFLaG_Diag_B2 MeanProj_B2}"
ROOT_OUT="${ROOT_OUT:-outputs/text/flag_static_mechanism}"

echo "[static] Testing exact FFT-free control and gradient invariants..."
python tests/test_flag_static_mechanism.py
python -m py_compile factory/pooling/static_flag_mechanism_pooling.py \
  text_repro/train_author_text_protocol.py text_repro/summarize_flag_static_mechanism.py

for seed in ${SEEDS}; do
  for task in ${TASKS}; do
    for method in ${METHODS}; do
      echo "[static] task=${task} pooling=${method} seed=${seed}"
      python -u text_repro/train_author_text_protocol.py \
        --task "${task}" --pooling "${method}" --seed "${seed}" \
        --flag_time_pool mean --disable_post_pool_norm \
        --output_dir "${ROOT_OUT}"
    done
  done
done

python -u text_repro/summarize_flag_static_mechanism.py \
  --root "${ROOT_OUT}" --seeds "${SEEDS}"
