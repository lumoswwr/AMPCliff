# E2: Exact Adam Step 1 and module-splice counterfactuals

## Why this experiment?

E1 seed 0 shows nonzero extra-parameter gradients and *positive* full
train/validation MSE gradient alignment for Gate-output and Projection,
but high minibatch gradient variability. E1 does **not** show that Adam's
actual first update improves validation, nor can it identify how the modules
change the prediction function.

E2 tests the local mechanism with just ONE real Adam step. It deliberately
does not retrain Mean, change the architecture, or perform multi-epoch sweeps.

## Study protocol

1. Read the ORIGINAL E0 seed summary and trained Mean checkpoint.
2. Construct the identical frozen-RoBERTa B2 at Gate=1 and Projection=I.
3. Require the step-0 predictions to reproduce the already-saved E0
   validation_pairs_step_0.npz.
4. Retrieve the SAME first shuffled 32-pair minibatch as E0.
5. Apply ONE joint Adam update (lr=1e-3, weight_decay=0) to the original
   trainable B2 pool + affine head. Do not save the resulting checkpoint.
6. Before interpreting anything, require that the updated model reproduces
   E0's saved validation_pairs_step_1.npz by pair-level prediction comparison,
   plus the step-1 trajectory's Spearman/MSE.
7. Compare six **parameter splices**, all derived from the SAME JOINT update:

       step0_mean_equivalent    no updated parameters
       head_only                only Step-1 affine-head parameters
       gate_only                only Step-1 Gate final-layer parameters
       projection_only          only Step-1 Projection parameters
       pool_only                Gate final layer + Projection Step-1 parameters
       full_step1               all Step-1 trainable parameters

   The upstream latent/Gate layers should be unchanged after this first step
   because the zero-initialized Gate output layer blocks their gradients.

8. Evaluate full STSB TRAIN and VALIDATION MSE, Pearson, Spearman, cosine
   changes; measure how the Gate multiplier departs from one on the *original
   first minibatch*. Report module parameter displacements.

These are **same-update counterfactual splices**, not standalone
Gate-only/Projection-only optimizer training trajectories. Joint effects
can be nonlinear and need not equal sums of isolated effects.

The reference validation split was already used for Mean checkpoint selection.
Results are mechanism diagnostics, not an untouched independent test.

## GPU server commands

Run from the AMPCliff checkout root.

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:$PYTHONPATH"

    # Compile-only syntax check; creates only Python's normal __pycache__:
    python -m py_compile text_repro/e2_adam_step1_counterfactual.py

    # Verify E0 source files and initial predictions; writes NO results
    # and creates NO new E2 output folders:
    python -u text_repro/e2_adam_step1_counterfactual.py --seed 0 --check-only

    # Full E2 one-step comparison:
    python -u text_repro/e2_adam_step1_counterfactual.py --seed 0

The **one and only E2 result path** for seed 0 is fixed:

    outputs/text/stsb_mean_equiv_e2_step1/seed_0/e2_step1.json

The script uses exclusive file creation and refuses to replace any earlier E2
result. No random suffix, timestamp folder, or confusing smoke-test output.

Print a compact human-readable summary and the exact result path.
You may directly read the result file using:

    cat outputs/text/stsb_mean_equiv_e2_step1/seed_0/e2_step1.json

Stop the research interpretation if reproduction fails:
- Missing original E0 step-0/step-1 validation predictions,
- mismatched model/dataset/seed/settings,
- initial Mean equivalence fails,
- per-pair E0 Step-1 predictions differ beyond 2e-4, or
- E0 recorded Step-1 metrics differ beyond tolerance.

## Reading E2

- e0_reproduction: step-0/step-1 E0 matched prediction differences.
- first_train_minibatch: first-batch hash and its pre-update MSE.
- gate_output_displacement / projection_displacement / head_displacement:
  parameter changes from the same joint Adam step. Do not equate magnitudes
  of different parameter groups with functional causal importance.
- variants: full TRAIN/VALIDATION metrics, differences versus the step-0
  baseline, cosine changes and sampled gate-multiplier deviations.
- --check-only produces **no JSON and no new output directory**.

If Step-1 is healthy but later validation drops, the better next experiment
is focused on multi-step dynamics. If Step-1 already hurts, use the splice
results to identify whether Gate, Projection, or their interaction is implicated.

## E1 output naming update

For future E1 runs, smoke tests now write NOTHING, and full E1 runs write
one fixed path:

    outputs/text/stsb_mean_equiv_e1_gradients/seed_0/e1_gradient_diagnostic.json

Both previously created, timestamp-named E1 output folders are preserved
untouched. To find the old full result (versus smoke) run:

    grep -l '"full_dataset_result": true' outputs/text/stsb_mean_equiv_e1_gradients/seed_0/run_*/e1_gradient_diagnostic.json

No old results are deleted or overwritten.
