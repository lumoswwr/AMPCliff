#!/bin/bash

source /home/data/home/wwr_lumos/miniconda3/etc/profile.d/conda.sh
conda activate flag

export PYTHONPATH=/home/data/home/wwr_lumos
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

REPO=/home/data/home/wwr_lumos/AMPCliff
MODEL=/home/data/home/wwr_lumos/models/esm2_t6_8M_UR50D
BASE="$REPO/data/blosum62 average/diff_5-trd_0.9"
ROOT="$REPO/outputs/ablation-new-data/esm2_t6_FLaG_e_coli_diff5"

cd "$REPO" || exit 1

mkdir -p "$ROOT"

STATUS="$ROOT/status.tsv"

if [ ! -f "$STATUS" ]; then
    printf "seed\tstatus\tstart_time\tend_time\n" > "$STATUS"
fi

for SEED in 1 2 3 4 5 6 7 8 9
do
    RUN="$ROOT/seed_${SEED}"
    mkdir -p "$RUN"

    # 已经成功生成 test_result.csv 的 seed 自动跳过
    if find "$RUN" -maxdepth 1 -name "*test_result.csv" | grep -q .; then
        echo "[$(date)] seed $SEED already completed, skipping."
        continue
    fi

    START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

    echo
    echo "=================================================="
    echo "Starting seed $SEED"
    echo "Time: $START_TIME"
    echo "=================================================="

    python -u downstream_train.py \
      mode.ddp=false \
      mode.amp=false \
      logger.log=false \
      other.debug=false \
      train.random_seed="$SEED" \
      train.num_epoch=50 \
      train.eval_epoch=2 \
      train.batch_size=4 \
      train.num_workers=1 \
      data.regression.mode=fix \
      data.regression.dataset=e_coli \
      'data.regression.condition=["blosum62 average"]' \
      'data.diff=[5]' \
      data.threshold=0.9 \
      "data.regression.fix.train_file=$BASE/grampa_e_coli_7_25-train.csv" \
      "data.regression.fix.valid_file=$BASE/grampa_e_coli_7_25-valid.csv" \
      "data.regression.fix.test_file=$BASE/grampa_e_coli_7_25-test.csv" \
      model.config_dir="$MODEL" \
      model.regression.version=esm2_t6 \
      model.regression.pooling=FLaG \
      model.regression.check_point.load=false \
      hydra.run.dir="$RUN" \
      > "$RUN/train.log" 2>&1

    EXIT_CODE=$?
    END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

    if [ "$EXIT_CODE" -eq 0 ]; then
        STATUS_TEXT="DONE"
        echo "[$(date)] seed $SEED completed successfully."
    else
        STATUS_TEXT="FAILED"
        echo "[$(date)] seed $SEED FAILED with exit code $EXIT_CODE."
    fi

    printf "%s\t%s\t%s\t%s\n" \
        "$SEED" "$STATUS_TEXT" "$START_TIME" "$END_TIME" >> "$STATUS"
done

echo
echo "=================================================="
echo "All requested seeds processed."
echo "Finished at: $(date)"
echo "=================================================="
