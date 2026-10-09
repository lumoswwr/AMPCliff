#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export SEEDS="${SEEDS:-0 1 2}"
export TASKS="${TASKS:-sprint}"
export METHODS="FLaG MeanProjRand"
export MATCHED_PROJECTION=1
export OUT="${OUT:-outputs/text/flag_identical_proj_pairs}"

echo "[exact-match] FLaG and MeanProjRand get identical W and b at initialization"
echo "[exact-match] A SHA256 guard will compare both initial tensors per seed"
bash text_repro/run_flag_paired_ap_multiseed.sh
