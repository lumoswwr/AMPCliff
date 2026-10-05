#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"

SEEDS=(0 1 2 3 4)
FORCE="${FORCE:-0}"

COMMON_ARGS=(
  --epochs 3
  --max_length 512
  --batch_size 8
  --eval_batch_size 16
  --grad_accum 4
  --backbone_lr 1e-5
  --pool_lr 1e-3
  --pool_dropout 0.1
  --warmup_ratio 0.1
  --output_dir outputs/text/imdb/paper_protocol
)

run_one () {
  local seed="$1"
  local pooling="$2"
  local exp_name="$3"
  local metrics="outputs/text/imdb/paper_protocol/${exp_name}/seed_${seed}/metrics.json"

  if [[ "${FORCE}" != "1" && -f "${metrics}" ]]; then
    echo
    echo "[skip] seed=${seed} pooling=${pooling}"
    echo "       existing: ${metrics}"
    return
  fi

  echo
  echo "================================================================"
  echo "IMDB seed=${seed} pooling=${pooling}"
  echo "================================================================"

  python -u text_repro/train_imdb.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --pooling "${pooling}" \
    --experiment_name "${exp_name}"
}

for seed in "${SEEDS[@]}"; do
  run_one "${seed}" mean imdb_mean
  run_one "${seed}" FLaG imdb_flag
  run_one "${seed}" FLaG_AlignmentAnchor imdb_alignmentanchor
done

echo
echo "================================================================"
echo "IMDB seeds 0-4 complete. Summarizing..."
echo "================================================================"

python -u text_repro/summarize_imdb_5seeds.py \
  --base_dir outputs/text/imdb/paper_protocol \
  --seeds 0,1,2,3,4
