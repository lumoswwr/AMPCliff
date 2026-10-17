# E8: exact structural attribution of FLaG_B2 ranking damage

## What question E8 actually answers

After 38 earlier experiments and E0-E7, we KNOW that STSB Spearman can
fall from a trained Mean-equivalent initialization; E7 ruled out many
claimed MSE improvements as mere better score calibration. That does
NOT identify WHICH additional computation changes the correct ordering.

E8 tests this distinct, **function-level** causal question using exact
B2 algebra, not another independently trained model.

It applies ONLY to the E0 B2 protocol: frozen RoBERTa trained-Mean
checkpoint; exact E0 Step10 joint Adam weights; centered sigmoid Gate,
real FFT along tokens, no frequency-specific gate/window/DC mask;
irFFT, valid Masked Mean, no post-pool LayerNorm, affine Projection and
positive-affine score Head.

## Exact identity and checked implementation

Let X be the masked RoBERTa hidden sequence (zeros at padded tokens);
m its valid-token mean; T its actual batch-padded FFT length; and
r=valid-masked-mean(X[(-t mod T)]) its circular-reflection mean.
Let a(X), b(X) be the REAL / IMAG halves of 2*sigmoid(Gate logits).

B2 pre-projection output:

    g(X) = a(X)*m + 0.5*(a(X)-b(X))*(r-m)

B2 final output:

    z(X) = W g(X) + c

Cosine is computed after L2-normalizing the final embedding; the
Step10 positive-affine Head is shared across all configurations.

Each TRAIN and validation sentence is passed through the real model.
The algebraic pre-Projection and post-Projection vectors must match
native model forward within 2e-5. The experiment STOPS if not.
Extra synthetic unit test checks odd/even FFT lengths at 1e-12.

## Twenty exhaustive factor controls

Gate *source*:
- identity: a=b=1 (exact original Mean representation)
- static: per-channel mean of TRAIN sentence-occurrence a,b,
  estimated from the whole TRAIN split only
- dynamic: actual Gate a(X),b(X) for each example

Other independently toggled computation terms:
- reflection off/on: omit/include 0.5*(a-b)*(r-m)
- Projection W: identity/trained matrix
- Projection c: zero/trained bias

Identity Gate renders the reflection term zero; 4 identity controls.
Static/dynamic each have 2^3=8 controls. Total **20** unique outputs.

Key direct comparisons at the **same trained checkpoint**:
- dynamic full vs static full: extra dependence on each example's
  Gate coefficients (also includes dynamic-padding dependence)
- dynamic no-reflection vs dynamic full: imaginary/real asymmetry
  and circular reflection / padding effects
- dynamic with W removed vs full: Projection weight matrix effect
- dynamic with c removed vs full: Projection bias effect
- identity Gate with W,c trained: Projection-only baseline
- static Gate with no reflection and no W/c: channel-only
  fixed feature recalibration, separate from new example information

Metrics for all controls: train + validation Spearman, Pearson, MSE.
On validation, E7's exact all-example-pairs rank audit is performed
for prespecified key controls, including corrected/broken orders
relative to Mean AND rank restorations relative to full B2.
The count of rank corrections is a Kendall-like statistic, not
mathematically equivalent to Spearman.

## Interaction-aware allocation

One-component-at-a-time removals are CONDITIONAL on all remaining
model operations. Their effects need not sum up.

Additionally, E8 computes and verifies the **four-factor Shapley
allocation of actual Spearman differences**:
- real-part Gate amplification (identity -> static OR dynamic Gate)
- real/imag reflection term (needs nonidentity Gate to matter)
- Projection weight matrix
- Projection bias

The 4 factors use the full 16-subset factorial (reflection-on but Gate
off is identical to reflection-off). Shapley contributions must add
EXACTLY to full-minus-Mean Spearman, for static and dynamic separately.
These are interacting inference-time function terms; Shapley does
NOT prove that independently retraining each term would cause the
same benefit/damage, nor single out a unique physical cause.

## Six decisions worth reporting, NOT predetermined results

1. If static full is approximately as harmful as dynamic full,
   sentence-dependent frequency Gate is unnecessary to reproduce
   most of the STSB damage at this checkpoint. Beware: static Gate
   can still encode training-distribution frequencies; this is not
   a proof "no frequency information is used".
2. If static full is considerably better than dynamic full,
   sentence-specific Gate changes *may* be harmful. Measure Gate
   variation carefully, including batch dynamic-padding effects.
3. If reflection removal restores rankings substantially, the
   distinct real/imag multipliers plus circular masking matter.
4. If Projection W or c removal has a large restoration effect,
   the damaging geometry is not simply "extra FFT information".
5. If both static and dynamic Gate harm similarly, evaluate
   whether covariance geometry W*diag(a) approximates a
   non-orthogonal change of the cosine metric.
6. If results differ sharply across seeds or attribution is
   dominated by interactions, we MUST NOT claim a single root
   cause. Discuss task/optimization and test other settings
   rather than stack more STSB-only micro-ablations.

Always interpret Spearman effects primarily; shared Step10 Head
means MSE here remains susceptible to calibration.

## Inputs and reproducibility

- Existing E0 reference Mean checkpoints / Step0,Step1,Step10 probes
  for all five seeds 0-4.
- Existing E7 cached full TRAIN + VALIDATION pairwise scores in NPZ.
- Model is recovered by exact replay of E0's first 10 Adam updates,
  and checked against E0 and E7 on every seed before accepting outputs.
- Train/validation E8 extraction uses same dynamic right-padding and
  batch64 as E7, preserving exact comparability. TRAIN static Gate
  is estimated from **TRAIN only**, no development leakage.
- Frozen encoder and reference selected using the same STSB validation
  set, so NOT a clean external test. Five seeds share data.

## How to run (one invocation, all five seeds)

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:$PYTHONPATH"

    python -m py_compile text_repro/e8_structural_gate_projection_attribution.py
    python -u text_repro/e8_structural_gate_projection_attribution.py --self-test
    python -u text_repro/e8_structural_gate_projection_attribution.py --check-only
    python -u text_repro/e8_structural_gate_projection_attribution.py

Expected exclusive outputs:

    outputs/text/stsb_mean_equiv_e8_structural_causality/
        e8_five_seed_structural_attribution.json
        e8_five_seed_structural_attribution.npz

The NPZ stores all 20 variant scores and original labels for both splits
and all five seeds, so later fine-grained rank audits will be CPU-only.
No overwrite, checkpoints untouched. The program caches each split's
Mean / reversal / a / b vectors temporarily, computes 20 variants
using the GPU *without* repeating the RoBERTa forward for variants,
and discards caches after each seed.

## Important stopping boundary

If exact reconstruction or E7 prediction checks fail, DO NOT
interpret E8 numbers; identify the protocol mismatch first.

If E8 passes, we will have identified the *functional* source of
damage at Step10. To say "why FLaG is worse than Mean on other tasks"
would require testing the mechanism in original-protocol models and
on Sprint/AMP with properly matched baselines and untouched data.
E8 itself cannot establish that broader generalization claim.
