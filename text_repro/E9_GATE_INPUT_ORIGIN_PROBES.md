# E9: tracing the INPUT ORIGIN of ineffective Gate updates

This is **not** another "dynamic Gate vs static Gate" comparison.
That question was already addressed in E8; the user is asking why the
additional Gate input information has harmful or weak ordering value,
and where that information originates in the computation.

## Research question and exact causal separation

Frozen trained-Mean RoBERTa sentence hidden states -> FFT ->
latent attention -> gate logits -> (real and imaginary channel gates) ->
reconstructed pooled vectors -> Projection -> same Step10 Head.

E9 modifies ONLY the frequency tokens supplied to latent attention
and the Gate generator; the original hidden states, token mask,
DC/non-DC coefficients used by the reconstruction branch, masked Mean,
reflected masked Mean, Projection weights/bias and Head NEVER change.

The real Gate uses [a,b]=2*sigmoid(Gate logits); the reconstructed output
for each intervention is checked against the exact E8 formula:

    z = W*[a*m + 0.5*(a-b)*(r-m)] + c

The E7 original Mean and original B2 scores MUST match to 1e-5.

Five original seed 0-4 E0 Step10 model states are recovered with exact
ten-Adam-minibatch replay and Step0/1/10 E0 score comparisons. No new
learning or model weights are saved. All five seeds run in ONE command.

## THREE source-localization tests (not dozens of arbitrary ablations)

### 1. Which spectral source matters: DC or non-DC?

In an unnormalized rFFT of masked token hidden states:

    F_0 = sum_{t valid} h_t = valid_length * masked_Mean

So the DC feature contains a sentence-length multiplier even if Mean
semantics are unchanged. The other frequency tokens contain additional
order/frequency statistics and can depend on FFT length and padding.

Two interventions each independently REPLACE the entire original DC
or original non-DC spectrum supplied to Gate attention, using a
different sentence as a donor. The donor is chosen LABEL-BLIND within
the same side of the same eval minibatch, and must have EXACTLY equal
valid-token length. FFT T is thus also exactly matched.
Singleton-length groups do not get replaced. Matching coverage is
always reported.

- dc_swap_0, dc_swap_1: replace DC, preserve own non-DC.
- ac_swap_0, ac_swap_1: replace non-DC, preserve own DC.
- all_swap_0, all_swap_1: swap all bands, negative control that MUST
  reproduce the donor Gate to numerical tolerance.
- dc_only, ac_only: zero the other group, *out-of-distribution
  sensitivity tests only*. Never call ablation damage proof of noise.

For each model score compare:
(a) original full and original Gate-only path;
(b) match-length DC and AC swaps;
(c) original Mean.

If AC donor swaps reliably *improve* validation ordering, even though
the original AC came from the correct sentence, that suggests the
original AC-conditioned adjustment is task-unhelpful at this checkpoint
under the donor swap. This remains a post-hoc function intervention,
not a verdict that AC contains no semantic information.
If swaps are harmful, the original source likely contains some useful
conditional info, although total learned Gate can still be inferior to
Mean. Report both average and per-seed results.

### 2. Does Gate react to a provably irrelevant change: FFT padding?

Retain original hidden tokens and mask and all downstream embedding
factors. Append 16 or 32 additional ALL-ZERO token positions ONLY to
the FFT input of latent attention -> Gate:

    rFFT(x, T) versus rFFT([x, zeros], T+16 or T+32)

This does not change the sentence, encoded valid hidden states,
original Mean, or reconstruction branch. Any Gate and prediction change
is therefore sensitivity to an irrelevant computational choice, not
new semantics. This alone does not establish how much harm it causes.

### 3. Does Gate know which bin is which?

The present B2 Latent Attention receives frequency tokens without
explicit frequency-index positional encoding, and the Gate is shared
across all frequency bins.

A uniform permutation of all frequency tokens MUST leave Gate output
unchanged to numerical tolerance (inference mode). This is an
architecture identity, tested at runtime. It implies the Gate
aggregator is a symmetric SET function of spectral tokens. It can
still use their coefficient distributions and phase; it cannot
associate a coefficient with a particular bin index as such.

This test is mechanistically informative but **cannot on its own
prove that position-blindness caused STSB loss**.

## What is recorded

- Full TRAIN and STSB development scores, Spearman/Pearson/MSE per source,
  across five seeds.
- Same for Gate-only input-channel scaling path, to factor out W and
  circular reflection already localized by E8.
- Validation corrected-versus-broken pairwise ordering comparisons
  relative to Mean and native full B2 (same E7 definition).
- Actual Gate average absolute changes under each intervention.
- Exact-length donor matching rate in each eval minibatch.
- Runtime original-Gate, full-B2, E0 and E7 consistency checks.
- Raw label and score matrix in a small compressed NPZ for further
  CPU-only probing, avoiding new GPU replay.

**Interpretation boundary:**
A decrease caused by replacing correct sentence bands with donor bands
suggests band-specific *useful conditional signal*, regardless of
whether the original Gate's net effect is negative relative to Mean.
An improvement suggests harmful source-conditioned behavior but does
not prove intrinsic nuisance without replication.
Zero-masking DC/AC strongly shifts the source distribution and is
a sensitivity probe only.
Five seeds share one STSB data split and a validation-selected Mean
reference. Do not report cross-task generalizations.

## Run on the original server

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:$PYTHONPATH"

    python -m py_compile text_repro/e9_gate_input_origin_probes.py
    python -u text_repro/e9_gate_input_origin_probes.py --self-test
    python -u text_repro/e9_gate_input_origin_probes.py --check-only
    python -u text_repro/e9_gate_input_origin_probes.py

One command = all five seeds, only ten original Adam updates per seed.
The main probe can take longer than E8 because it recomputes trained
latent-attention Gate for each controlled spectral input; original
RoBERTa hidden states are computed only once per sentence batch.

## Output (exclusive; no old results overwritten)

    outputs/text/stsb_mean_equiv_e9_gate_origin/
        e9_five_seed_gate_frequency_source.json
        e9_five_seed_gate_frequency_source.npz

A result should be interpreted only if exact replay, original-score
agreement, donor all-swap and permutation invariance checks pass.
If results are inconsistent across seeds or do not localize a source,
we must state that rather than add probes endlessly.
