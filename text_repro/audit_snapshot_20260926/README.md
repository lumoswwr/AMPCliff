# FLaG / STFT-FLaG Codex Audit Snapshot

Purpose:
Independent code/results audit of the FLaG/STFT-FLaG mechanism study.

Included:
- experiment configs
- final metrics
- training histories
- mechanism-probe summaries
- training logs
- project mechanism summary
- environment metadata

Intentionally excluded:
- model checkpoints (*.pt)
- pretrained models
- raw datasets
- test_predictions.csv
- pair_results.csv
- caches and temporary files

Important experimental distinction:

STSB:
- FLaG and E12 are matched on non-operator settings
- dropout=0.0
- post_pool_norm=True
- global FFT vs frozen local STFT is the main operator comparison

Sprint:
- original published-setting FLaG:
  dropout=0.1, post_pool_norm=False
- frozen E12:
  dropout=0.0, post_pool_norm=True
- matched global and dropout x post-norm controls were therefore added

Key audit questions:
1. Are FLaG and E12 comparisons actually matched where claimed?
2. Are checkpoint-selection and validation/test boundaries correct?
3. Is there any test-set leakage?
4. Are the global/local 2x2 counterfactuals mathematically and computationally valid?
5. Do manual/native sanity checks really isolate gate observation vs reconstruction?
6. Is the circular-reflection derivation consistent with the implementation?
7. Are reported mean/std and paired effects computed correctly?
8. Are any conclusions stronger than the evidence supports?
