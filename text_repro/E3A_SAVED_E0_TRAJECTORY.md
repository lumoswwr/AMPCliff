# E3a: Analyze saved E0 validation trajectory (read-only)

## Goal

Explain WHEN, **among the recorded E0 validation probes**, B2's STSB
validation MSE and validation Spearman:
- initially improve or decline;
- cross their step-0 trained Mean baseline; and
- decouple from one another.

Compare the original B2 validation output with the **same frozen encoder's
Mean pooling but the B2 HEAD at that step**. This is a per-checkpoint
counterfactual using E0's saved predictions, not a multi-epoch retraining
ablation.

Study rank motion versus the step-0 cosine similarities, paired squared-error
improvement/worsening counts, and mean absolute cosine change. No train split
data, test split, GPU, model initialization, optimization, or new checkpoints.

## What E0 actually saved

Existing script:
- E0 summary: outputs/text/stsb_frozen_trained_mean/seed_N/summary.json
- E0 trajectory: outputs/text/stsb_frozen_trained_mean/seed_N/trajectory.jsonl
- E0 per-pair VALIDATION predictions:
  outputs/text/stsb_frozen_trained_mean/seed_N/validation_pairs_step_K.npz

Default E0 probes are step 0, 1, 10, 50, and each epoch end. Check the actual
saved steps rather than presuming all are present.

Each NPZ contains labels, cos_full, cos_same_encoder_mean, score_full,
score_same_encoder_mean. E3a recalculates all validation MSE, Spearman,
Pearson and refuses to continue if they differ from E0's recorded metrics.
The script also cross-checks the first/final Spearman with E0 summary.

**No full TRAIN-set predictions were recorded at every probe.** E3a cannot
infer each checkpoint's full train MSE or Spearman from the last minibatch MSE.

## Server commands

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag

    python -m py_compile text_repro/e3a_analyze_saved_e0_trajectory.py

    # Source-integrity check and terminal summary; creates NO E3a output:
    python -u text_repro/e3a_analyze_saved_e0_trajectory.py --seed 0 --check-only

    # Produce the full E3a result and two charts:
    python -u text_repro/e3a_analyze_saved_e0_trajectory.py --seed 0

Only repeat another seed later if seed 0 is useful. No seed requires retraining.

## Files and output safeguards

Predictable, fixed locations, with no random suffix or timestamp:

    outputs/text/stsb_mean_equiv_e3_trajectory/seed_0/
      e3_trajectory_analysis.json
      e3_val_mse.png
      e3_val_spearman.png

Neither --check-only nor syntax-only checks create an E3a result folder.
A second full run REJECTS existing output files rather than overwriting them.
E0, E1, and E2 outputs are never touched.

The JSON reports:
- observed_probe_steps (all actual saved check times)
- observations[].validation (full B2 metrics)
- observations[].mean_pool_with_current_head_validation
- observations[].delta_vs_step0 / delta_from_previous_probe
- observations[].pooling_mse_contribution_vs_mean_with_same_head
- observations[].mean_absolute_pair_cos_change_from_step0
- observations[].pair_rank_movement_from_step0
- observations[].per_pair_squared_error_vs_step0
- events.first_observed_validation_mse_above_step0
- events.first_observed_validation_spearman_below_step0
- events.best_observed_validation_mse_step and
  events.best_observed_validation_spearman_step

## Scientific limitations

1. These are **sparse probes**, not continuous records. If an observation
   changes between steps 10 and 50, the actual crossover could be anywhere
   in the interval, or could cross and recross between unobserved steps.
2. Head-only same-encoder Mean can change MSE (calibration) but not sample
   ranking / Spearman when the affine head slope remains strictly positive.
   Per-step MSE differences from B2 must be interpreted as a counterfactual
   at that step, **not** a causal decomposition of the entire training run.
3. This validation split was used to choose the reference Mean; it is not
   an untouched test set.
4. Comparing saved outputs cannot identify which B2 parameter group is
   responsible for later behavior. A subsequent E3b would need checkpoints,
   saved gradients, or a carefully controlled replay, and must be justified
   by the E3a observations.
