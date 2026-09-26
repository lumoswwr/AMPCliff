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


## Authoritative result scope

Use the following as the authoritative formal result directories.

Sprint formal / frozen comparisons:
- sprint/experiments/sprint_frozen_mean
- sprint/experiments/sprint_frozen_flag
- sprint/experiments/sprint_frozen_e12
- sprint/experiments/sprint_e12matched_global
- sprint/experiments/sprint_flag_postnorm
- sprint/experiments/sprint_flag_nodropout

Sprint mechanism probe:
- sprint/probes/global_local_2x2

STSB formal baselines:
- stsb/experiments/mean
- stsb/experiments/FLaG

STSB formal mechanism / ablation experiments:
- stsb/experiments/experiments/E1_flag_hann
- stsb/experiments/experiments/E2_stft_flag_w16_h8_rect
- stsb/experiments/experiments/E3_stft_flag_w8_h4_rect
- stsb/experiments/experiments/E4_stft_flag_w8_h4_framepos
- stsb/experiments/experiments/E5_stft_flag_w8_h4_hann_centered
- stsb/experiments/experiments/E11_stft_flag_w8_h8_rect
- stsb/experiments/experiments/E12_stft_flag_w16_h16_rect
- stsb/experiments/experiments/E13_flag_fixedfft128

STSB mechanism probes:
- stsb/probes/P1_gate_reconstruction_2x2
- stsb/probes/P2_global_local_2x2
- stsb/probes/E7_token_knockout
- stsb/probes/E8_local_dc_diagnostic

## Debug / smoke warning

Do NOT treat directories beginning with `_smoke_` as formal results.

Some files under `sprint/logs/` are historical protocol-debug runs from before
the Sprint scorer / validation-threshold protocol was frozen. In particular,
early all-negative Sprint runs and monotone-head debugging runs are retained
only so the audit can reconstruct how protocol bugs were diagnosed.

For numerical claims, prefer the formal experiment `metrics.json`,
`history.csv`, probe `summary.json`, and `STFT_MECHANISM_SUMMARY.md`.
Use raw logs mainly to verify execution traces.

Important Sprint protocol history:
- the initial free Linear(1,1) cosine head could learn a negative slope and
  invert ranking; those runs are invalid/debug only;
- the final scorer is monotone:
  logit = softplus(raw_scale) * cosine + bias;
- checkpoint selection uses best validation accuracy after validation-only
  threshold calibration;
- F1 thresholds are also selected on validation only and then frozen for test;
- no test-set threshold optimization should be considered part of the final
  protocol.

Important interpretation warning:
- frozen E12 vs published-setting Sprint FLaG is a valid frozen-configuration
  comparison;
- it is NOT a clean causal local-STFT comparison because dropout and
  post_pool_norm differ;
- use FLaG_E12Match and the dropout x post-norm controls for causal
  interpretation of Sprint performance differences.
