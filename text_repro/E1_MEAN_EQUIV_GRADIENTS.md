# E1: trained-Mean-equivalent B2 gradients (read-only)

## Questions

At optimizer step 0 from a strong STSB Mean:

1. Do Gate-output and identity-Projection parameters have nonzero full-TRAIN MSE gradients?
2. Are full TRAIN and VALIDATION gradients aligned or conflicting?
3. How variable are the actual batch-32 gradients compared with full TRAIN?

E1 is a gradient diagnosis, **not** proof of the final three-epoch degradation mechanism.

## Protocol

- Reuse the E0 seed-specific summary and exactly its trained Mean checkpoint.
- Build B2 with the existing helper, freeze RoBERTa and confirm initial pool
  and prediction identity.
- Reuse the existing STSB train/validation splits, tokenization and dynamic
  padding (TRAIN batch 32; VALIDATION batch 64).
- Objective MSE with labels divided by 5. Weight minibatch gradients by
  number of examples to obtain true full-dataset gradient for this loader.
- Inspect zero-gradient barrier on Gate upstream / Attention / FFN, and
  separately report Gate output, Projection, and affine Head gradients.
- Sample up to 16 minibatches from the seeded TRAIN pass for noise diagnostics.
- **No optimizer, no parameter updates, no old output overwrites, no test set.**
  All trainable parameters must remain exactly unchanged.

## Run from the GPU server

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:$PYTHONPATH"

    # Syntax check:
    python -m py_compile text_repro/e1_diagnose_mean_start_gradients.py

    # Optional smoke check only (NOT research data):
    python -u text_repro/e1_diagnose_mean_start_gradients.py --seed 0 --train_limit 32 --val_limit 64 --sample_minibatches 1

    # Full E1 scientific run, seed 0:
    python -u text_repro/e1_diagnose_mean_start_gradients.py --seed 0 --sample_minibatches 16

Only after an interpretable seed-0 result, consider repeating seeds 1–4 using
already-saved references, without retraining anything.

Each run writes one JSON into a newly generated unique directory:

    outputs/text/stsb_mean_equiv_e1_gradients/seed_0/run_YYYYMMDDTHHMMSSZ_RANDOM/e1_gradient_diagnostic.json

The terminal prints its actual path after "Saved NEW result".

## Interpret cautiously

- initial_identity: B2 and Mean should match to numerical precision.
- initial_upstream_gradient_sanity_first_train_batch: gradients of Gate
  upstream / latent layers should be zero or disconnected at step 0; otherwise
  stop to inspect implementation.
- train_vs_validation_gradients: train_norm, validation_norm, cos_train_val and
  sgd_first_order_validation_mse_coefficient, per module and overall.
  Positive cosine indicates local train/val agreement for infinitesimal
  plain SGD, while negative indicates conflict. Neither predicts Adam or
  Spearman changes by itself.
- minibatch_gradients: sampled batch cosines versus the full gradient and RMS
  deviation/full norm. Ratios are undefined if the full gradient is tiny.
  Descriptive only, not statistical uncertainty or proof of causation.

Do not compare raw gradient norms across differently sized modules as causal
importance. The validation split was used for Mean checkpoint selection,
so this is **not** an untouched test-set finding.
