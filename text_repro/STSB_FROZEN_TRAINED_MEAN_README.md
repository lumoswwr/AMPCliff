# STSB: frozen RoBERTa starting from a well-trained Mean checkpoint

## Research question

The first exploratory run started B2 from **pretrained** Mean at about
0.65 STSB validation Spearman. This experiment instead starts from the
**task-fine-tuned Mean**, previous seed 0 about **0.8814** Spearman.

Transfer its exact RoBERTa + prediction-head weights into FLaG_B2, make
B2 output numerically identical to trained Mean, freeze RoBERTa, and
train ONLY extra B2 pool parameters plus the same score-head parameters.

Question: does B2 keep, improve or lose a **previously strong Mean** solution?

B2 is a controlled Mean-equivalent variant, not the original FLaG architecture.

## Run

```bash
git switch FLaG-STFT-mechanism
git pull
conda activate flag

# Tiny isolated smoke test, NOT research data:
QUICK=1 bash text_repro/run_stsb_frozen_mean_start.sh

# Full seed-0 experiment:
bash text_repro/run_stsb_frozen_mean_start.sh
```

Mean checkpoint lookup (first existing path):
1. `MEAN_CHECKPOINT=/some/path/best_model.pt` explicit user choice
2. `outputs/text/mean_equiv_exploration/stsb/mean/seed_0/best_model.pt`
   (only present if prior experiment had SAVE_CHECKPOINTS=1)
3. `outputs/text/stsb_frozen_trained_mean/seed_0/mean_reference_best.pt`
   from an earlier run of this experiment
4. Otherwise **train Mean from scratch** using the earlier STSB protocol:
   train full RoBERTa + Mean for 3 epochs, select best validation Spearman,
   save checkpoint. Full run stops if reference Spearman below 0.86.

At initialization the script asserts B2/Mean pooled-vector and
prediction equality. The RoBERTa weights and score head are transferred
from the SAME checkpoint. B2 has gate=1, identity projection, mean
time-pooling, post_pool_norm=False, and pooling dropout=0.

Then freeze RoBERTa; train FLaG pool + cosine-score head for 3 epochs,
Adam LR=1e-3, weight_decay=0, batch32. Evaluate STSB validation at
step 0, 1, 10, 50, and each epoch end. No test set accessed.

## Files

```text
outputs/text/stsb_frozen_trained_mean/seed_0/
  mean_reference_best.pt                   # if ref fitted in this run
  mean_reference_training_history.json     # if ref fitted in this run
  mean_reference_validation.json
  mean_reference_val_pairs.npz
  trajectory.jsonl
  validation_pairs_step_0.npz
  validation_pairs_step_1.npz
  validation_pairs_step_10.npz
  validation_pairs_step_50.npz
  ...
  best_pool_head.pt
  last_pool_head.pt
  summary.json
```

The B2 pool/head checkpoint does not duplicate the frozen RoBERTa.
Keep the referenced Mean checkpoint for reconstructing B2 later.

`full_flag_b2.spearman`: STSB validation Spearman at a B2 checkpoint.
`flag_minus_original_mean_spearman`: difference vs strong trained Mean.
`same_encoder_mean_with_current_head`: bypass B2 but keep the current
score head. Its Spearman should remain fixed even as its MSE changes.
`last_minibatch_train_mse`: just the most recent minibatch MSE.

## Meaning and limits

If the score goes down and stays down, then we have evidence of B2
learning a harmful extra transformation starting at a strong Mean.
If performance matches/improves, we should focus next on why joint
RoBERTa + FLaG training differed from separate Mean training. A
short-lived early drop does not imply final failure.

No causal mechanism about Gate, FFT, Projection or gradients is assumed.
No repository notes or prior experiment files are changed.
