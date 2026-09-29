#!/usr/bin/env bash
set -euo pipefail

# Paper-setting text FLaG controls on STSBenchmark.
# Only dropout/post-pool norm differ from the existing STSB runs.
#
# Shared setting:
#   dropout = 0.1
#   post_pool_norm = True
#   seeds = 0 1 2
#
# E12 remains win=16, hop=16, rectangular, no centering, no overlap.

python text_repro/run_managed_sts.py \
  --experiment_name stsb_paper_flag_d01_norm1 \
  --pooling FLaG \
  --pool_dropout 0.1 \
  --post_pool_norm 1 \
  --seeds 0 1 2

python text_repro/run_managed_sts.py \
  --experiment_name stsb_paper_e12_d01_norm1 \
  --pooling STFT_FLaG \
  --stft_win_length 16 \
  --stft_hop_length 16 \
  --stft_window_type rect \
  --pool_dropout 0.1 \
  --post_pool_norm 1 \
  --seeds 0 1 2
