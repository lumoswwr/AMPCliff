#!/usr/bin/env bash
set -euo pipefail

# Proposal 2: attention-derived frequency-wise gate.
#
# Keep the original FLaG architecture, but turn the latent-attention
# distribution over frequency tokens into an additional scalar gate per bin:
#
#   a_k = mean_latent attention(k), normalized over k
#   r_k = K * a_k
#   g_k = 2*r_k / (1+r_k)
#   X'_k = OriginalFeatureGate(X_k) * g_k
#
# Uniform attention -> g_k=1 for every k, exactly recovering the original
# FLaG gating path. g_k can approach 0, so bins can now be truly suppressed.
#
# Development protocol:
#   STSB  : seed 0, 3 epochs, UNFROZEN RoBERTa
#   Sprint: seed 0, 10 epochs, FROZEN RoBERTa

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PARENT_DIR="$(dirname "${REPO_ROOT}")"
export PYTHONPATH="${PARENT_DIR}:${REPO_ROOT}/text_repro${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-0}"
SKIP_STSB="${SKIP_STSB:-0}"
SKIP_SPRINT="${SKIP_SPRINT:-0}"

echo "============================================================"
echo "Sanity check | attention-derived frequency gate"
echo "============================================================"
python -u text_repro/check_attention_frequency_gate.py

if [[ "${SKIP_STSB}" != "1" ]]; then
  echo "============================================================"
  echo "STSB | UNFROZEN | FLaG_AttnFreqGate | seed ${SEED}"
  echo "============================================================"

  python -u text_repro/train_sts.py \
    --pooling FLaG_AttnFreqGate \
    --experiment_name stsb_unfrozen_flag_attnfreqgate \
    --seed "${SEED}" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_length 128 \
    --backbone_lr 1e-5 \
    --pool_lr 1e-3 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --pool_dropout 0.0 \
    --post_pool_norm 1
fi

if [[ "${SKIP_SPRINT}" != "1" ]]; then
  echo "============================================================"
  echo "Sprint | FROZEN | FLaG_AttnFreqGate | seed ${SEED}"
  echo "============================================================"

  python -u text_repro/train_sprint.py \
    --pooling FLaG_AttnFreqGate \
    --experiment_name sprint_frozen_flag_attnfreqgate \
    --seed "${SEED}" \
    --epochs 10 \
    --batch_size 32 \
    --eval_batch_size 64 \
    --max_length 128 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --checkpoint_metric accuracy
fi

python -u text_repro/summarize_attention_frequency_gate.py \
  --seed "${SEED}"
