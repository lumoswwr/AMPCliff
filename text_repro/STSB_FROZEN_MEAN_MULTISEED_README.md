# STSB Mean -> frozen RoBERTa -> FLaG_B2: five seeds

## Why

Test whether the seed-0 observation is repeatable:

- start from a **strong, STSB-fine-tuned Mean** (different Mean RoBERTa for each seed);
- verify B2 starts equal to that Mean;
- freeze RoBERTa and train B2 pooling plus the cosine affine scoring head
  for 3 epochs with batch32 and Adam LR=1e-3;
- compare *final* B2 and original trained Mean on the **same** STSB validation
  split (official validation: 1500 sentence pairs).

Previously seed0 finished with original Mean val Spearman 0.881437,
final frozen B2 val Spearman 0.871316, and B2 better train fitting but
worse validation fitting. This multi-seed run tests repeatability, not
the cause of the gap.

## One command

```bash
cd ~/AMPCliff
git switch FLaG-STFT-mechanism
git pull
conda activate flag
bash text_repro/run_stsb_frozen_mean_multiseed.sh
```

Default seeds are **0 1 2 3 4**. The runner performs one run at a time,
never concurrent GPU jobs, and **reuses** seed0's already-existing
`summary.json` and `posthoc_train_val_comparison.json` without
rerunning or overwriting those files.

Each new seed:
1. Reuses the same-seed trained Mean checkpoint when available, otherwise
   refits Mean with STSB RoBERTa unfrozen for 3 epochs, picking highest
   validation Spearman. It does not share Mean weights between seeds.
2. Verifies B2 equals this seed's pretrained Mean (pool vector and score
   numerical tolerance) BEFORE frozen training.
3. Freezes the RoBERTa, trains B2 pooling + scale/bias for 3 epochs,
   and records trajectory/checkpoints under `seed_N`.
4. Runs the read-only **full train and full validation** evaluation using
   the last B2 checkpoint, including best separate positive-affine
   cosine score heads **fitted on TRAIN only**.
5. After every requested seed succeeds, writes cross-seed CSV and JSON.

No STSB test set is loaded. Analysis of all seeds uses the
**last epoch after 3 epochs**, never validation-picked B2 best epoch,
as the PRIMARY endpoint. The best post-training probe is reported
*only* as exploratory context.

To avoid silently omitting lower-performing reference seeds, all
requested seeds are required for aggregation; do not delete, cherry-pick,
or replace a seed because its score is surprising. The script's predeclared
minimum acceptable Mean reference Spearman is **0.84**, a permissive
sanity check that does **not change training parameters**. If a reference
fails even that, the script stops visibly.

## Results

```text
outputs/text/stsb_frozen_trained_mean/
  seed_0/
    summary.json                       # EXISTING, reused
    posthoc_train_val_comparison.json  # EXISTING, reused
  seed_1/
    mean_reference_best.pt
    summary.json
    trajectory.jsonl
    last_pool_head.pt
    posthoc_train_val_comparison.json
    training_multiseed.log
    posthoc_multiseed.log
  ...
  seed_4/
  multiseed_comparison.csv
  multiseed_summary.json
```

Send `multiseed_summary.json` or paste the runner's final table to
continue interpretation.

Key columns:
- `final_val_spearman_delta`: B2 final STSB val Spearman - SAME-SEED
  Mean reference val Spearman. Primary pre-specified outcome.
- `train_spearman_delta`: B2 last checkpoint vs Mean on train.
- `train_mse_delta_fit_head`: B2 minus Mean training MSE where each pool's
  positive-affine score head is fitted analytically on the full TRAIN split.
  Negative means B2 fits train better.
- `val_mse_delta_fit_head`: same fitted heads evaluated on validation.
  Positive means B2 has worse held-out MSE.
- `best_posttraining_val_spearman_delta`: exploratory best >0 step,
  not equivalent to final-epoch metric and susceptible to selection.

`mean ± sample SD` is descriptive across five seeds, not a confidence
interval. The same official validation set selects each seed's Mean
reference and scores the new B2 adaptation. Thus the comparison is
exploratory and **not an independent test-set generalization result**.

## Resuming and customization

```bash
# Exactly resume the default set (skips complete seed folders)
bash text_repro/run_stsb_frozen_mean_multiseed.sh

# Another set, e.g. extend later to 0..9 without retraining existing seeds
SEEDS="0 1 2 3 4 5 6 7 8 9" bash text_repro/run_stsb_frozen_mean_multiseed.sh

# Summarize only existing completed seeds; throws error on any missing seed
python text_repro/summarize_stsb_frozen_mean_multiseed.py --seeds 0 1 2 3 4
```

The optional `FORCE=1` reruns/overwrites the selected seed's previous
result and post-hoc comparison. **Do not set FORCE=1 in the first run.**

B2 is a controlled **Mean-equivalent** FLaG variant
(mean time pooling, no post-pool LayerNorm, centered gate and identity
out projection), not the original FLaG architecture used to generate
the paper's original text/image results.
