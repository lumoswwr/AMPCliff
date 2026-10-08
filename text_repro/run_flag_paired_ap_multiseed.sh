#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"

SEEDS="${SEEDS:-0 1 2}"
TASKS="${TASKS:-sprint stsb}"
METHODS="${METHODS:-FLaG FLaG_B2 MeanProj_B2}"
OUT="${OUT:-outputs/text/flag_paired_ap_multiseed}"
FORCE="${FORCE:-0}"

python -m py_compile \
  text_repro/train_author_text_protocol.py \
  text_repro/summarize_flag_paired_multiseed.py \
  factory/pooling/static_flag_mechanism_pooling.py
python tests/test_flag_static_mechanism.py

for task in ${TASKS}; do
  for seed in ${SEEDS}; do
    for method in ${METHODS}; do
      case "${method}" in
        FLaG) folder="flag" ;;
        FLaG_B2) folder="flag_b2" ;;
        MeanProj_B2) folder="mean_proj_b2" ;;
        StaticFLaG_ReIm_B2) folder="static_reim_b2" ;;
        StaticFLaG_Diag_B2) folder="static_diag_b2" ;;
        *) echo "Unrecognized method ${method}" >&2; exit 2 ;;
      esac
      run_dir="${OUT}/${task}/${folder}/seed_${seed}"
      if [[ "${FORCE}" != "1" && -s "${run_dir}/metrics.json" && -s "${run_dir}/best_model.pt" ]]; then
        echo "[skip] ${task} s${seed} ${method}: checkpoint and metrics exist"
        continue
      fi
      echo "================================================================================"
      echo "[paired] task=${task}, seed=${seed}, pooling=${method}"
      echo "[paired] selection: Sprint val AP / STSB val Spearman"
      python -u text_repro/train_author_text_protocol.py \
        --task "${task}" --pooling "${method}" --seed "${seed}" \
        --flag_time_pool mean --disable_post_pool_norm \
        --sprint_selection_metric ap \
        --output_dir "${OUT}"
    done
  done
done

python -u text_repro/summarize_flag_paired_multiseed.py \
  --root "${OUT}" --seeds "${SEEDS}" --tasks "${TASKS}"
