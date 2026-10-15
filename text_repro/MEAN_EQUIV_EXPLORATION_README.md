# Mean-equivalent FLaG exploratory pilot (STSB and Sprint)

This script investigates two descriptive questions before proposing any mechanism:
1. Starting from the *same Mean prediction function*, how do independently
   trained Mean and FLaG_B2 behave during each epoch?
2. At each FLaG_B2 checkpoint, how different is the performance of the FULL
   FLaG pooling from replacing only the pooling output by the raw Mean of
   the *same encoder* (same RoBERTa, same score scale/bias)?

This is B2, **not original max-pooling FLaG**. B2 uses Global FFT, masked
Mean time pooling, no post-pool LayerNorm, no pooling dropout, Gate=1 and
Projection=I at initialization.

Original FLaG paper: Mean is better on STSB; full FLaG is better than
*raw Mean* on Sprint. A trained Mean + projection is a different comparator.

## Run on GPU server

```bash
cd ~/AMPCliff
git pull
conda activate flag

# 1. Small smoke test, stored in a SEPARATE folder.
QUICK=1 bash text_repro/run_mean_equiv_exploration.sh

# 2. Complete seed-0 exploration, full train/validation.
bash text_repro/run_mean_equiv_exploration.sh

# 3. Optional: only one task, additional seeds, or checkpoints.
TASKS=stsb bash text_repro/run_mean_equiv_exploration.sh
TASKS=sprint bash text_repro/run_mean_equiv_exploration.sh
SEEDS="1 2" bash text_repro/run_mean_equiv_exploration.sh
SAVE_CHECKPOINTS=1 bash text_repro/run_mean_equiv_exploration.sh
```

The smoke test (64 train / 256 val / 1 epoch) only checks that code runs;
never interpret its metrics as scientific evidence.

Training protocols: STSB fine-tunes RoBERTa for 3 epochs; Sprint freezes
RoBERTa and trains for 10 epochs. Both use batch32, Adam, weight decay=0,
no scheduler, pooling LR=1e-3, STSB backbone LR=1e-5, and identical
data splits / seed / shuffled minibatch order for each paired method.

The script only evaluates **validation** and does not access official test.
Best epochs are selected by validation Spearman (STSB) or validation AP
(Sprint), not Sprint accuracy.

## Outputs

Under `outputs/text/mean_equiv_exploration/{stsb,sprint}/`:

- `mean/seed_0/trajectory.jsonl`: independently trained raw Mean.
- `flag_b2/seed_0/trajectory.jsonl`: trained B2, plus same-encoder Mean.
- `*/seed_0/val_pairs_epoch_N.npz`: per-pair label, cosine similarity,
  score, and same-encoder Mean comparison; includes initialization N=0.
- `seed_0_comparison.json`: compact paired epoch-by-epoch table.
- `*/seed_0/complete.json`: configuration and best epoch.

Optional `SAVE_CHECKPOINTS=1` saves `best_model.pt` and
`last_model.pt` for *each* method and task; these are large files.

## Interpretation

Comparing `flag_b2` to `flag_checkpoint_mean` asks whether *changing
the pooling output at a fixed trained checkpoint* helps or hurts the task
metric. Comparing to `independently_trained_mean` evaluates the two
training trajectories, which may use different learned encoders for STSB.

No finding in this pilot alone identifies which component (Gate, FFT,
Projection) caused a difference, or *why* optimization learned it. We
should inspect the results together before planning a causal intervention.
