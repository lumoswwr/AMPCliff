#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

python -u text_repro/plot_text_layer_band_knockout_paperstyle.py \
  --input_dir "${INPUT_DIR:-outputs/text/layer_band_knockout_frozen_seed0}"
