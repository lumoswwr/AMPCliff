#!/usr/bin/env bash
set -euo pipefail

# 10-seed paper-setting comparison on STSBenchmark.
# Text FLaG setting from the paper:
#   dropout = 0.1
#   post_pool_norm = True
#
# Existing SUCCESS markers are skipped automatically, so seeds 0-2
# from the pilot run will not be retrained.

python text_repro/run_managed_sts.py \
  --experiment_name stsb_paper_flag_d01_norm1 \
  --pooling FLaG \
  --pool_dropout 0.1 \
  --post_pool_norm 1 \
  --seeds 0 1 2 3 4 5 6 7 8 9

python text_repro/run_managed_sts.py \
  --experiment_name stsb_paper_e12_d01_norm1 \
  --pooling STFT_FLaG \
  --stft_win_length 16 \
  --stft_hop_length 16 \
  --stft_window_type rect \
  --pool_dropout 0.1 \
  --post_pool_norm 1 \
  --seeds 0 1 2 3 4 5 6 7 8 9
