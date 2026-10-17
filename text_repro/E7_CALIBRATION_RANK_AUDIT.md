# E7: Calibration-vs-ordering audit (one five-seed decision run)

This E7 is part of the **new E0–E7 trained-Mean-equivalent STSB series**,
not the historical "E7" in the separate 38-experiment report.

## Why this particular experiment is not redundant

The completed `FLaG_38项实验概览_最终版.md` already covers
STFT/global comparisons, padding reflection, DC dominance, Gate saturation,
the lack of material input-to-Gate variability after training, Mean+Projection
on Sprint, and Alignment-aware Mean/FLaG fusion. DO NOT re-run those.

The new E0–E6 trajectory established exact trained-Mean-equivalent
initialization and a Step10 MSE-amplitude overshoot. It did **not** isolate
whether that MSE gain was primarily from score calibration rather than better
relative ordering. The current B2 Head is updated jointly; the original Mean
score under this Step10 Head is not guaranteed to be optimally calibrated.

**Scientific question**: Is the observed MSE gain after shrinking the FLaG
deviation evidence of useful learned semantic geometry, or can the same gain
be obtained by calibrating the original Mean scores, which CANNOT improve
Spearman at all?

## Comparisons

Predeclared outputs at the exact original E0 Step10 checkpoint:
- Trained Mean representation
- E6 interpolation with (alpha,beta)=(0.25,0.25)
- E6 interpolation with (alpha,beta)=(0.50,0.50)
- Full Step10 FLaG_B2

For each path:
1. Original shared Step10 Head as E6.
2. Fit ONLY the two scalar coefficients of an **increasing affine score
   mapping** on full TRAIN, closed-form OLS. Apply unchanged to VALIDATION.
   Reject non-positive slopes. Spearman must be EXACTLY invariant.
3. Decompose per-split MSE into squared mean bias, scale mismatch,
   and Pearson-dependent error:
      MSE = (mean_score-mean_target)^2
          + (std_score-std_target)^2
          + 2*std_score*std_target*(1-Pearson).
4. For all pairs of STSB examples with unequal targets, count strict
   rank comparisons corrected (Mean wrong => candidate right), broken
   (Mean right => candidate wrong), and stratify by target-label gap.
   This is Kendall-like pairwise concordance, **NOT** Spearman.
5. Save all raw grid predictions and labels from E6 to one compressed
   offline cache; next diagnostic can be computed without GPU replay.

Run EXACT E0 first ten Adam updates and verify original Step0/1/10
predictions, E4 endpoints, E6 train/validation metrics for each seed.
No new learned Pooling parameters or model training. All five seeds
0,1,2,3,4 are processed automatically in ONE invocation.
The full TRAIN pairwise comparison uses exact blockwise computation so
it does not allocate a giant n-by-n dense ranking matrix.

## Falsifiable decision rules

- If calibrated original Mean is as good as or better in validation MSE
  than the best E6 partial model, E6's MSE gain cannot be credited to
  extra FLaG representation quality without stronger evidence.
- If the partial model still beats **calibrated Mean** in validation MSE,
  inspect the Pearson component and the corrected-vs-broken ordering
  comparisons. Additional predictive *association* may exist without
  a Spearman gain, but this does not yet prove causal semantic content.
- If rank breaks outweigh corrections or corrected comparisons are
  concentrated at small target gaps, the raw rank effect may account
  for the Spearman loss. Never equate Kendall count with Spearman.
- This test does not prove anything across tasks or fix the optimizer.
  The historical 38-experiment report cautions that trained Gate can
  saturate and Mean+Projection can match FLaG on Sprint. The next
  training change, IF warranted, must compare ranking-aware objective
  against the matched Mean+Projection and full FLaG baselines across
  more than STSB.

## GPU command

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:$PYTHONPATH"
    python -m py_compile text_repro/e7_calibration_rank_audit.py
    python -u text_repro/e7_calibration_rank_audit.py --check-only
    python -u text_repro/e7_calibration_rank_audit.py

One fixed report and one reusable score cache, never overwritten:

    outputs/text/stsb_mean_equiv_e7_calibration_rank/
      e7_five_seed_calibration_ranking_audit.json
      e7_five_seed_calibration_ranking_audit.npz

## Interpretation cautions

- Positive affine calibration leaves ranks unchanged and therefore
  cannot improve Spearman. Its MSE gains measure score calibration.
- The pretrained Mean checkpoint was selected on STSB validation
  Spearman. Comparisons on this same split are exploratory.
- The five seeds share the same train/validation examples.
- Ten-step frozen-RoBERTa changes do not characterize original FLaG
  full training or the conditions behind different task performance.
- Stop STSB-only micro-ablation after this analysis. Discuss the
  original 38 experiments plus E0-E7 before implementing further
  architecture or training variants.
