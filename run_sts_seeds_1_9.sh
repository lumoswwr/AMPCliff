#!/bin/bash

source /home/data/home/wwr_lumos/miniconda3/etc/profile.d/conda.sh
conda activate flag

export PYTHONPATH=/home/data/home/wwr_lumos
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

REPO=/home/data/home/wwr_lumos/AMPCliff
OUT="$REPO/outputs/text/stsbenchmark"
STATUS="$REPO/outputs/text/sts_all_status.tsv"

cd "$REPO" || exit 1

mkdir -p "$OUT"

if [ ! -f "$STATUS" ]; then
    printf "pooling\tseed\tstatus\tstart_time\tend_time\n" > "$STATUS"
fi

for POOLING in mean FLaG
do
    for SEED in 1 2 3 4 5 6 7 8 9
    do
        RUN="$OUT/$POOLING/seed_$SEED"
        mkdir -p "$RUN"

        if [ -f "$RUN/metrics.json" ]; then
            echo "[$(date)] $POOLING seed $SEED already complete, skipping."
            continue
        fi

        START="$(date '+%Y-%m-%d %H:%M:%S')"

        echo
        echo "================================================"
        echo "Starting $POOLING seed $SEED"
        echo "Time: $START"
        echo "================================================"

        python -u text_repro/train_sts.py \
          --pooling "$POOLING" \
          --seed "$SEED" \
          --epochs 3 \
          --batch_size 4 \
          --grad_accum 4 \
          --max_length 128 \
          > "$RUN/train.log" 2>&1

        CODE=$?
        END="$(date '+%Y-%m-%d %H:%M:%S')"

        if [ "$CODE" -eq 0 ]; then
            STATE="DONE"
            echo "[$(date)] $POOLING seed $SEED DONE"
        else
            STATE="FAILED"
            echo "[$(date)] $POOLING seed $SEED FAILED, code=$CODE"
        fi

        printf "%s\t%s\t%s\t%s\t%s\n" \
          "$POOLING" "$SEED" "$STATE" "$START" "$END" >> "$STATUS"
    done
done

echo
echo "All STS runs processed at $(date)"
