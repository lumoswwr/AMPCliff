# FLaG gate counterfactual probe

**Goal:** Determine whether the trained FLaG relies on the DC channel gate,
the AC gate, per-channel anisotropy, per-input adaptability, or the output projection.
This is a checkpoint-only post-hoc intervention, not a training comparison.

All checkpoints must be from the **time_pool=mean, post_pool_norm=False, dropout=0**
initialization / angle-audit experiment, not the published Max/LN setting.

## Key mathematical distinction

- Residual sigmoid: g=1+sigmoid(logit), multiplier between 1 and 2.
  The largest possible angle from positive channel-wise rescaling of a
  vector is arccos(2*sqrt(2)/3) ~ 19.47 degrees.
- Centered sigmoid: g=2*sigmoid(logit), multiplier between 0 and 2.
  The maximum possible angle approaches 90 degrees.
- Thus a B2 angle near 80 degrees is not automatically an optimization failure.

## Counterfactuals (all reuse the same encoder, learned cosine head and validation samples)

- full: original checkpoint forward. The probe requires numerical agreement
  with the actual model forward and original full-validation metric.
- dc_identity: keep learned modulation on AC but disable gate at k=0.
- ac_identity: keep learned modulation on DC but disable gate on k>0.
- gate_identity: disable all spectral gating, retain trained projection.
- scalar_gate: replace gate by its per-sample mean over all channels.
- bias_only: use only final gate-linear bias, a **static** channel gate.
- feature_only: use only the sample-dependent W2*h, minus final bias.
- no_projection: retain learned gate but bypass output linear projection.
- raw_mean: direct masked Mean of the **same checkpoint encoder**.

The DC interventions are diagnostic (frequency-local alterations to the
learned shared channel gate), not independently trained architectures.
Do not equate the raw_mean result with a separately trained Mean baseline.

## Run on V100 server

From AMPCliff branch FLaG-STFT-mechanism:

    cd /home/data/home/wwr_lumos/AMPCliff
    git pull origin FLaG-STFT-mechanism
    conda activate flag

Quick smoke only, not scientifically reliable Sprint AP:

    TASKS=stsb METHODS=FLaG_B2 MAX_PAIRS=64 bash text_repro/run_flag_gate_counterfactual.sh

Full validation probe (recommended, no retraining):

    TASKS="stsb sprint" METHODS="FLaG FLaG_B2" MAX_PAIRS=0 bash text_repro/run_flag_gate_counterfactual.sh

More optional checkpoint comparisons:

    METHODS="FLaG_A1 FLaG_B1" bash text_repro/run_flag_gate_counterfactual.sh

Results are in outputs/text/flag_gate_counterfactual_seed0/,
including each method probe_validation.json and
counterfactual_summary_validation.md.

## Interpretations

1. If dc_identity improves STS-B but harms Sprint, suppressing DC rotations
   may be task-specific rather than universally correct.
2. If ac_identity barely changes results, the learned non-DC gate may be
   dispensable **when final time pooling is masked Mean**. Note that masked
   averages on a padded FFT window can retain AC contributions.
3. If scalar_gate retains performance, gate-induced directional reweighting
   may not be necessary for this trained checkpoint.
4. If bias_only approximates full better than feature_only, the gate may
   behave more as a static channel mask than a contextual attention gate.
   This warrants a separately trained static-mask control later.
5. Large individual-vector angles do not imply poor pairwise cosine
   similarity. The no_projection intervention helps test that distinction.

Use STS-B Spearman and Sprint average precision as the primary task
metrics. Fixed-threshold Sprint F1 can move because intervention shifts
score calibration, even if ranking is relatively stable.

**Caveat:** Results describe reliance of an already trained checkpoint.
A post-hoc removal is a distribution shift; it does not prove the altered
architecture would behave the same if retrained.
