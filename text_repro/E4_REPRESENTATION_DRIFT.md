# E4: Which block moves the STSB embeddings farther from trained Mean?

## Motivation

E3b already established that jointly trained Gate+Latent and Projection both
move per-pair cosine similarities and validation ranks. That experiment DOES
NOT tell us which block rotates the individual normalized sentence embeddings
farther away from the original Mean vectors, nor whether a larger vector
rotation helps more individual pair predictions.

E4 therefore measures **direct representation-space** distances, avoiding an
E3b rerun with only renamed metrics.

## Run all five seeds in ONE invocation

From the existing server repo:

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:${PYTHONPATH:-}"

    python -m py_compile text_repro/e4_five_seed_representation_drift.py
    python -u text_repro/e4_five_seed_representation_drift.py --check-only
    python -u text_repro/e4_five_seed_representation_drift.py

The script processes seeds **0,1,2,3,4** in a single invocation. No --seed
flag and no separate seed-by-seed follow-up are needed. It reuses each
already-trained Mean reference and performs exactly ten E0 Adam updates per
seed for identical Step10 weights. It DOES NOT train five new experiments.

The source E0 Step0/1/10 full predictions must match exactly within the E2
numerical tolerance, with checks against original MSE/Spearman and online
minibatch losses. If an E3b result exists, validate all four same-Head
counterfactual val MSE/Spearman/Pearson values against that archived file.

## New measured quantities

For every validation sentence occurrence (1500 STSB pairs, 3000 sides per
seed), encode frozen RoBERTa once and calculate four normalized embeddings:

- mean: original trained Mean of frozen RoBERTa tokens
- gate_latent_only: trained Step10 FLaG_B2 **pre-Projection** gated mean
- projection_only: Step10 affine Projection applied to original Mean
- full_step10: trained Step10 Projection applied to trained gated mean

The three comparisons against Mean give angular deviation in **degrees**.
Both conditional effects are also measured: Projection after Gate and Gate
after Projection. Angles are computed by stable 2*atan2(||u-v||,||u+v||).

For each variant also record pair-cosine drift vs Mean, validation Spearman,
mean rank shift, and fraction of individual validation pair errors improved
or worsened. ALL comparisons use the SAME trained Step10 positive-affine
Head; it is not re-fitted to each counterfactual.

Important: angle magnitude and Spearman damage are different quantities.
A large change in one sentence's embedding angle need not imply large
change in its relative ranking to other pairs. Angular differences for
the two parameter blocks are conditional on the other block being at its
initial value and do not add up to the full-model displacement.

## Output

Exactly one fixed JSON, exclusive (never overwrite):

    outputs/text/stsb_mean_equiv_e4_representation_drift/
      e4_five_seed_step10_representation_drift.json

It holds per-seed angular distributions and metrics plus a five-seed
mean/std summary. Original E0/E1/E2/E3a/E3b/E3c files remain unchanged.
No checkpoints or extra timestamp folders are written.

## Scientific limits

- This measures the frozen-encoder STSB Step10 point, NOT full 540-step
  dynamics and NOT task-dependence across AMP/Sprint.
- Angle describes representation displacement, not directly a useful
  or harmful semantic change.
- Same-Head per-pair MSE comparison describes validation improvement under
  a particular trained Head, not generalization to an untouched test set.
- The Mean reference had been validation-selected. Avoid claiming an
  independent test result.
- These are inference-time functional counterfactuals at one jointly trained
  parameter trajectory, not independently trained single-block models.
