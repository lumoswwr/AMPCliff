#!/usr/bin/env bash
set -e

python text_repro/train_imdb.py \
  --pooling mean \
  --experiment_name imdb_mean

python text_repro/train_imdb.py \
  --pooling FLaG \
  --experiment_name imdb_flag

python text_repro/train_imdb.py \
  --pooling FLaG_AlignmentAnchor \
  --experiment_name imdb_alignmentanchor
