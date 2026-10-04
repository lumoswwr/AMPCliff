#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"

COMMON_ARGS=(
  --seed 0
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

echo
echo "===== IMDB: Mean ====="
python -u text_repro/train_imdb.py \
  "${COMMON_ARGS[@]}" \
  --pooling mean \
  --experiment_name imdb_mean

echo
echo "===== IMDB: Original FLaG ====="
python -u text_repro/train_imdb.py \
  "${COMMON_ARGS[@]}" \
  --pooling FLaG \
  --experiment_name imdb_flag

echo
echo "===== IMDB: Alignment-aware Anchor ====="
python -u text_repro/train_imdb.py \
  "${COMMON_ARGS[@]}" \
  --pooling FLaG_AlignmentAnchor \
  --experiment_name imdb_alignmentanchor

echo
echo "===== IMDB seed 0 complete ====="
for f in \
  outputs/text/imdb/paper_protocol/imdb_mean/seed_0/metrics.json \
  outputs/text/imdb/paper_protocol/imdb_flag/seed_0/metrics.json \
  outputs/text/imdb/paper_protocol/imdb_alignmentanchor/seed_0/metrics.json
do
  echo
  echo "--- ${f} ---"
  cat "${f}"
done
