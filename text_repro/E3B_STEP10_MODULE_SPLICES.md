# E3b: Locate the Step-10 B2 ranking change (same-run parameter splices)

## Why this experiment?

E3a found that for **all five seeds**, validation Spearman has declined by
E0 Step 10 relative to the trained Mean initialization. E1 found nonzero
training gradients with positive train/validation MSE gradient alignment at
initialization. E2 showed that BOTH Gate-output and Projection can change
rankings on the first actual Adam update, but did not explain Step 10.

Question: at the original Step-10 B2 point, which **trained parameter blocks**
carry the change in validation cosine-similarity rankings?

This is *not* a claim that Gate alone is an FFT-specific mechanism or that
Projection would have the same behavior if separately trained. No hyperparameter
sweep, new architecture, or independently fitted single-block model.

## Exact E0 replay first

- Load the exact E0 seed reference Mean checkpoint; transfer trained frozen
  RoBERTa and original affine Head to FLaG_B2 at Gate=1, Projection=I.
- Recreate the SAME seeded shuffled STSB train loader (batch 32, original
  dynamic padding), E0 optimizer Adam lr=0.001 weight decay=0 and MSE.
- Update the same joint trainable parameters for **exactly ten** minibatches.
- At Steps 0, 1, and 10 require per-pair predictions to match E0's saved
  validation_pairs_step_0.npz, step_1.npz, step_10.npz. Also check MSE and
  Spearman against E0 trajectory. Stop on failed replay before analyzing
  parameter splices.
- A smoke --check-only mode checks source files, shapes and Step-0 predictions.
  It performs zero training updates and writes NO E3b files or directories.

## Actual parameter-block definitions

1. Gate+Latent: **ALL Pooling state** except time_out_proj.weight/bias.
   This includes freq_gate input/output layers AND trainable latents,
   attention, FFN and normalization (including any unchanged unused layers).
   These upstream parameters CAN change after Step 1; isolating only the Gate
   final layer at Step 10 would be incorrect.
2. Projection: time_out_proj.weight/bias.
3. Head: log_scale and bias.

A single jointly trained B2 yields Step-0 and Step-10 values for each block.
E3b builds all **8 combinations** of the three blocks:

    step0_mean_equivalent     initial gate + initial projection + initial head
    head_only                 initial gate + initial projection + Step10 head
    gate_latent_only          Step10 gate + initial projection + initial head
    projection_only           initial gate + Step10 projection + initial head
    pool_only                 Step10 gate + Step10 projection + initial head
    gate_latent_plus_head     Step10 gate + initial projection + Step10 head
    projection_plus_head      initial gate + Step10 projection + Step10 head
    full_step10               Step10 gate + Step10 projection + Step10 head

Measure full STSB TRAIN and VALIDATION MSE/Spearman/Pearson, validation cosine
changes and ranking shifts. Compare Pooling contrasts **under a shared Step10
Head** for fair MSE comparisons, alongside head-independent Spearman effects.
The script also reports the difference-in-differences interaction as a
*descriptive* quantity, not evidence of a specific architectural synergy.

## Server commands

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:$PYTHONPATH"

    python -m py_compile text_repro/e3b_replay_step10_module_splices.py

    # Input and Step-0 sanity only, NO new E3b outputs:
    python -u text_repro/e3b_replay_step10_module_splices.py --seed 0 --check-only

    # Real replay of TEN Adam updates, plus eight counterfactuals:
    python -u text_repro/e3b_replay_step10_module_splices.py --seed 0

Default start: **seed 0 only**. Check mechanism identification before
repeating seeds 1-4. The code never re-fits trained Mean.

## Fixed output, no overwrite

    outputs/text/stsb_mean_equiv_e3b_step10/seed_0/e3b_step10_counterfactual.json

One clearly named result file. No timestamp/random directories, and exclusive
creation: an existing result will NEVER be overwritten. E0, E1, E2 and E3a
outputs are strictly read-only.

Send the final console table and the E3b JSON file.

## Interpretation and falsification

- If only Step10 Gate+Latent splices reproduce the Spearman reduction,
  investigate Gate-path adaptation (NOT yet specifically Fourier selectivity).
- If only Projection splices reproduce it, investigate the learnable cosine
  metric as the leading explanation.
- If both alone cause some reduction, inspect **conditional** effects at a
  fixed Step10 Head and compare Pool-only vs individual changes.
- If neither alone reproduces the joint effect, their interaction may be
  important, but non-additivity of Spearman alone is not sufficient.
- If replay FAILS at Step 1 or 10, STOP. Do not attribute the resulting
  spliced metrics to the original E0 phenomenon.
- Comparisons do not tell us how separate Gate-only/Projection-only training
  would converge. Step10 is an early local observation, not a full-epoch
  explanation or causal generalization conclusion.

The same E0 validation split originally selected the Mean checkpoint, so
these are diagnostics on existing validation data, NOT an independent test.
