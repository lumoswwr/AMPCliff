#!/usr/bin/env bash
# STSB reproducibility pilot: independent trained-Mean starting points and
# fixed frozen-RoBERTa FLaG_B2 adaptation, 5 seeds, sequential GPU use.
#
# Default: SEEDS="0 1 2 3 4"; seed0 should be skipped and reused.
# No CUDA overlap, no test evaluation, no modification of prior experiments.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Match preceding scripts' package imports.
export PYTHONPATH="$(dirname "$ROOT"):$ROOT/text_repro${PYTHONPATH:+:$PYTHONPATH}"

SEEDS="${SEEDS:-0 1 2 3 4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/text/stsb_frozen_trained_mean}"
MIN_REFERENCE_SPEARMAN="${MIN_REFERENCE_SPEARMAN:-0.84}"
MODEL_PATH="${MODEL_PATH:-/home/data/home/wwr_lumos/models/roberta-base}"
STSB_DATA_PATH="${STSB_DATA_PATH:-data/text/stsbenchmark}"
FORCE="${FORCE:-0}"

if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
    echo "FORCE must be 0 or 1" >&2
    exit 2
fi
read -r -a seed_list <<< "$SEEDS"
if [[ "${#seed_list[@]}" -eq 0 ]]; then
    echo "SEEDS is empty" >&2
    exit 2
fi
for seed in "${seed_list[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed: $seed" >&2
        exit 2
    fi
done

python -m py_compile \
    text_repro/train_stsb_frozen_mean_start.py \
    text_repro/analyze_stsb_frozen_mean_trainval.py \
    text_repro/summarize_stsb_frozen_mean_multiseed.py

echo "=============================================================="
echo "STSB 5-seed pilot: trained Mean -> frozen RoBERTa -> FLaG_B2"
echo "SEEDS: $SEEDS"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "Reference threshold: $MIN_REFERENCE_SPEARMAN"
echo "FORCE: $FORCE"
echo "=============================================================="

for seed in "${seed_list[@]}"; do
    run_dir="$OUTPUT_ROOT/seed_$seed"
    mkdir -p "$run_dir"
    echo
    echo "========== seed $seed =========="
    if [[ -f "$run_dir/summary.json" && "$FORCE" != "1" ]]; then
        echo "[reuse] $run_dir/summary.json (not retraining)"
    else
        args=(
            --seed "$seed"
            --model_path "$MODEL_PATH"
            --stsb_data_path "$STSB_DATA_PATH"
            --output_dir "$OUTPUT_ROOT"
            --reference_epochs 3
            --flag_epochs 3
            --min_reference_spearman "$MIN_REFERENCE_SPEARMAN"
        )
        if [[ "$FORCE" == "1" ]]; then
            args+=(--force)
        fi
        python -u text_repro/train_stsb_frozen_mean_start.py "${args[@]}" \
            2>&1 | tee "$run_dir/training_multiseed.log"
    fi

    if [[ ! -f "$run_dir/summary.json" || ! -f "$run_dir/last_pool_head.pt" ]]; then
        echo "[error] seed $seed incomplete after training; stopping" >&2
        exit 1
    fi

    if [[ -f "$run_dir/posthoc_train_val_comparison.json" && "$FORCE" != "1" ]]; then
        echo "[reuse] $run_dir/posthoc_train_val_comparison.json"
    else
        python -u text_repro/analyze_stsb_frozen_mean_trainval.py \
            --seed "$seed" \
            --model_path "$MODEL_PATH" \
            --stsb_data_path "$STSB_DATA_PATH" \
            --pilot_dir "$OUTPUT_ROOT" \
            2>&1 | tee "$run_dir/posthoc_multiseed.log"
    fi
done

python -u text_repro/summarize_stsb_frozen_mean_multiseed.py \
    --root "$OUTPUT_ROOT" --seeds "${seed_list[@]}"
