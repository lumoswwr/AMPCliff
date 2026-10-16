# E5: Why does joint Gate + Projection deviation not reliably help?

## Research objective (primary)

The target question is NOT "which block deviates farther from Mean",
which E4 already studied. Instead, test why parameter changes that can
separately improve validation MSE do not reliably yield better predictions
when combined, under the exact E0 trained-Mean-equivalent STSB protocol.

E4 already found a positive five-seed mean validation MSE interaction.
However, "positive interaction" is NOT proof of optimizer interference.

## Exact per-pair mathematical decomposition

At the same trained Step10 model, use one fixed Step10 positive-affine
Head and the four E4 inference-time representations:

- M: frozen RoBERTa trained-Mean output.
- G: trained Gate+Latent output before final Projection.
- P: trained Projection applied to original Mean.
- GP: both jointly trained modules applied in the original order.

Define r=M-y, dG=G-M, dP=P-M and k=GP-G-P+M.
For squared loss L(Z)=(Z-y)^2:

    I = L(GP)-L(G)-L(P)+L(M)
      = 2*dG*dP + 2*(r+dG+dP)*k + k^2

The first term comes purely from the SQUARED loss. It is present even
when the two changes add perfectly at prediction level (k=0).
The second term is the functional-nonadditivity correction.

E5 computes this identity for every pair and ASSERTS that the
decomposition closes numerically. It also compares the hypothetical
prediction M+dG+dP with the actual joint score GP.

Further statistics include:
- fraction dG and dP have the same / opposite sign
- fraction each update moves toward the target at first order
- categories in which both individually improve but joint worsens
- distribution and upper tail of k
- share of positive MSE damage concentrated in top 5% of pairs
- Spearman, Pearson, MSE of all four variants, with the same Head

## Train versus validation is central

Unlike E4, compute **both full STSB TRAIN and validation** for each seed.
The train metric is full deterministic no-shuffle evaluation using
batch 32 and dynamic padding; it is NOT E0's online minibatch loss.
Validation uses the original E0 1500-pair development split batch 64.

If nonadditivity/alignment differ systematically on train vs validation,
the difference is evidence about the STSB data-split dependence at Step10,
but not enough to establish generalization mechanisms across other tasks.

## Reproducibility and scope

All five seeds 0,1,2,3,4 are processed in a single command.
The script first validates their existing E0 archives and E4 results.
It reconstructs each seed's original trained Mean checkpoint, replays
exactly ten original joint Adam minibatches, and demands exact original
E0 Step0/Step1/Step10 validation per-pair score/cosine agreement.
It additionally validates ALL four validation counterfactual metrics
against the previously completed E4 JSON.

No independent model training, no full three-epoch retraining, no
optimizer ablations, and no checkpoint writing. All counterfactual
results are computed by read-only forwards at one joint Step10 state.

## Server commands

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:\${PYTHONPATH:-}"

    python -m py_compile text_repro/e5_joint_gain_mechanism.py
    python -u text_repro/e5_joint_gain_mechanism.py --check-only
    python -u text_repro/e5_joint_gain_mechanism.py

No --seed flag; it ALWAYS runs all five seeds.

## ONE fixed output file, exclusive/no overwrite

    outputs/text/stsb_mean_equiv_e5_joint_gain/
        e5_five_seed_step10_joint_gain_decomposition.json

Read only: E0, E1, E2, E3a, E3b, E3c, E4 and all Mean checkpoint files.
One output JSON contains both full-train and validation per-seed
results and a five-seed average/std summary.

## Interpretation, not pre-commitment

- Alignment 2*dG*dP dominates: the apparent MSE "conflict" may largely
  reflect arithmetic reinforcement of prediction shifts under squared
  loss, not direct evidence of interacting modules.
- Functional k correction dominates: output composition itself adds
  a meaningful extra effect. Its *source* remains to be investigated,
  e.g. learned affine map, normalization, sentence-dependent Gate.
- Large errors concentrated on few pairs: investigate whose predictions
  degrade, before changing the training procedure.
- Differences between train and validation: ask whether their residuals,
  predictive directions or tail behavior differ, but remember Mean was
  selected using the same validation set.

This is a local Step10 MECHANISM diagnosis, not yet an explanation of
why FLaG helps AMP/Sprint and harms STSB or full training. Those are
the longer-term targets; do not overinterpret this single checkpoint.
