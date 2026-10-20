# STSB classic Learning-to-Rank loss suite

## Research question

On STSB, does changing the training objective from pointwise MSE to standard pairwise/listwise ranking losses make the extra capacity in FLaG-Zero or original FLaG produce a reliable improvement in ranking?

This is a controlled baseline comparison, not a proposal for a new ranking loss.

## Conditions

- Architectures: mean, flag_zero (Mean-equivalent FLaG_B2 control), and flag (original-protocol FLaG).
- Seeds: 0–4.
- Existing completed mse and ranknet outputs are reused; the runner skips completed runs and never overwrites per-run results.
- New loss conditions: ranknet_scaled, pairwise_hinge, listnet, listmle, and lambdarank.
- Training: existing STSB split, 3 epochs, batch size 32, eval batch size 64, max length 128, Adam, encoder LR 1e-5, pooling/head LR 1e-3, no weight decay.
- Checkpoint selection: validation Spearman. Test metrics are evaluated only after checkpoint selection.
- Primary metric: Spearman. Also report train/validation Spearman, Pearson, and train-fitted positive-affine calibrated test MSE.

## Losses

1. **MSE**: pointwise regression baseline, already completed.
2. **RankNet (historical)**: pairwise logistic loss on raw cosine. This exactly preserves the existing experiment and does not train the affine score scale.
3. **Scaled RankNet**: same pairwise logistic form, but uses score = exp(log_scale) * cosine + bias. This checks whether the raw-cosine score scale constrained the earlier RankNet baseline.
4. **Pairwise hinge / SVMRank-style**: max(0, 1 - sign(y_i-y_j)*(s_i-s_j)), skipping tied-label pairs.
5. **ListNet**: top-one probability cross-entropy between softmax relevance labels and softmax predicted scores.
6. **ListMLE**: Plackett-Luce permutation likelihood, sorting each minibatch by descending relevance. STSB has ties; tied items retain their randomized order in the shuffled minibatch.
7. **LambdaRank-style**: pairwise logistic loss weighted by the absolute change in NDCG from swapping a pair, using graded gains 2^label - 1.

All pairwise losses skip exact label ties. Pairwise losses use the trainable affine score except the historical raw-cosine RankNet condition.

## Important protocol limitation

STSB is not a query-document ranking dataset and does not provide natural query groups. For ListNet, ListMLE, and LambdaRank, each randomly sampled minibatch is treated as a list. These are therefore stochastic global-ranking surrogates, not canonical query-grouped LTR. Results should be interpreted with this limitation in mind.

The historical RankNet result remains in its original output folder and is not rerun. The additional scaled RankNet condition is a distinct controlled variant, not an overwrite or repetition of the old condition.

## Outputs

Stable root: outputs/text/stsb_ltr_ranknet_standard/

Each run: <model>/<loss>/seed_<seed>/

The runner updates summary.md, summary.csv, and paired_differences.csv after all conditions are complete. Paired differences include loss-vs-MSE comparisons for each architecture and architecture-vs-Mean comparisons under each loss.
