# STSB standard RankNet comparison

## Purpose
Test whether replacing the MSE objective with a standard pairwise ranking objective changes the optimization behavior of Mean, FLaG-Zero, and original-protocol FLaG.

## Conditions
- Models: `mean`, `flag_zero` (the Mean-equivalent `FLaG_B2` control), and `flag` (original-protocol FLaG).
- Objectives: MSE and standard RankNet pairwise logistic loss.
- Seeds: 0, 1, 2, 3, 4.
- Training: STSB official train/validation/test splits, 3 epochs, batch sizes 32/64, max length 128, Adam, encoder LR 1e-5, pooling/head LR 1e-3, no weight decay.
- Checkpoint selection: validation Spearman. Test is evaluated only after checkpoint selection.

## RankNet loss
For every minibatch, treat each STSB sentence pair as one ranked item. Compare all unordered pairs of examples whose labels differ; ignore ties. If labels are (y_i,y_j) and cosine scores are (s_i,s_j), minimize:
`softplus(-sign(y_i-y_j) * (s_i-s_j))`
averaged over valid pairs. Use cosine similarity directly as the ranking score, so the affine MSE head cannot reduce the ranking loss merely by inflating its scale.

## Interpretation guardrails
- This is a standard baseline, not a new ranking method.
- Spearman is the primary metric. MSE is secondary.
- Raw MSE from the RankNet model is not directly comparable to MSE-trained outputs because RankNet does not train the affine score head. Therefore also report a positive-slope affine calibration fitted on the training split and evaluated on test.
- Results from FLaG-Zero are mechanism-control results; FLaG-Zero is not the original FLaG architecture.
- Do not select models or hyperparameters using test metrics.

## Output
Stable output root: `outputs/text/stsb_ltr_ranknet_standard/`
Per-run results are stored under model/objective/seed directories. Completed runs are never overwritten.