# FLaG / STFT-FLaG STSB Mechanism Study

## 1. Scope

This branch contains reproduction and exploratory mechanism experiments built on:

- FLaG: Frequency-Domain Latent-attention Gated Pooling
- RoBERTa-base on STSBenchmark
- AMP E. coli reproduction with ESM2-8M
- Local STFT variants of FLaG
- Mechanistic probes for frequency knockout, token knockout, padding dependence,
  complex-gate reflection, and global/local reconstruction decomposition

The STSB mechanism experiments are exploratory. Multiple variants were developed
using the same benchmark, so later results should not be treated as an untouched
confirmatory test.

---

# 2. Environment

Repository:

    /home/data/home/wwr_lumos/AMPCliff

Conda environment:

    flag

RoBERTa-base:

    /home/data/home/wwr_lumos/models/roberta-base

STSBenchmark:

    /home/data/home/wwr_lumos/AMPCliff/data/text/stsbenchmark

Typical environment:

    conda activate flag

    export PYTHONPATH=/home/data/home/wwr_lumos/AMPCliff:/home/data/home/wwr_lumos:$PYTHONPATH
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1

Server GPU is shared. Check `nvidia-smi` before starting long jobs.
Do not kill unrelated processes.

---

# 3. Original FLaG architecture

Given token representations

    X in R^(T x D)

global FLaG applies an rFFT along the token dimension:

    X_hat = rFFT(X)

Real and imaginary components are concatenated:

    F = [Re(X_hat) || Im(X_hat)] in R^(F x 2D)

Learned latent queries attend to the frequency tokens:

    A = MHA(Q, F, F)

The latent representation is processed by residual/FFN blocks and summarized.
A channel-wise sigmoid gate is produced:

    g in (0,1)^(2D)

With residual gating:

    F_tilde = F * (1 + g)

The real and imaginary components are recombined, followed by iFFT, masked time
pooling, optional post-pooling LayerNorm, and a final projection.

Text experiments use:

- num_latents = 8
- num_heads = 4
- dropout = 0
- masked max pooling
- post_pool_norm = True
- max_length = 128

---

# 4. Reproduction results

## 4.1 AMP E. coli

ESM2-8M + FLaG, 10 seeds:

- RMSE: 0.564854 ± 0.011198
- Spearman: 0.549663 ± 0.017300
- Recall@50: 17.5 ± 2.592725

Runner:

    run_flag_ecoli_seeds_1_9.sh

---

## 4.2 STSBenchmark

Protocol:

- RoBERTa-base
- max_length = 128
- batch_size = 4
- grad_accum = 4
- effective batch = 16 sentence pairs
- 3 epochs
- backbone LR = 1e-5
- pooling LR = 1e-3
- cosine similarity
- gold score divided by 5
- MSE loss
- checkpoint selected by validation Spearman

10-seed reproduction:

Mean pooling:

- Spearman: 0.851201 ± 0.002000
- Pearson: 0.856674 ± 0.001832

FLaG:

- Spearman: 0.841559 ± 0.003538
- Pearson: 0.838504 ± 0.003444

Thus Mean pooling remains substantially stronger than FLaG on STSB.

For mechanism experiments, seeds 0,1,2 of FLaG have:

- Spearman: 0.841128 ± 0.002177
- Pearson: 0.836765 ± 0.001782

---

# 5. STFT variants

## E1: Global Hann

Global Hann window before global FFT.

- Spearman: 0.837019 ± 0.002763
- Pearson: 0.836682 ± 0.000282

Hann alone did not improve FLaG.

---

## E2: STFT 16/8 rectangular

win = 16
hop = 8
rectangular
overlap

- Spearman: 0.842204 ± 0.001824
- Pearson: 0.837783 ± 0.003514

---

## E3: STFT 8/4 rectangular

win = 8
hop = 4
rectangular
overlap

Seeds:

- seed0: 0.843555 / 0.840130
- seed1: 0.843653 / 0.837288
- seed2: 0.841793 / 0.838637

Mean:

- Spearman: 0.843000 ± 0.001047
- Pearson: 0.838685 ± 0.001421

Paired Spearman improvement vs FLaG:

- +0.002003
- +0.000591
- +0.003024

3/3 positive.

---

## E4: E3 + learned frame position

- Spearman: 0.842282 ± 0.001834

Compared with E3:

- mean delta Spearman: -0.000718

No evidence that this frame-position implementation explains the gain.

---

## E5: E3 + periodic Hann, centered

- Spearman: 0.843027 ± 0.000593
- Pearson: 0.838083 ± 0.002720

Essentially tied with E3.

---

## E11: STFT 8/8, no overlap

- Spearman: 0.842166 ± 0.001423
- Pearson: 0.837608 ± 0.001154

---

## E12: STFT 16/16, no overlap

This is the current frozen STFT candidate.

- win = 16
- hop = 16
- rectangular
- no overlap

Seeds:

- seed0: Spearman 0.842811, Pearson 0.839843
- seed1: Spearman 0.843653, Pearson 0.837246
- seed2: Spearman 0.843792, Pearson 0.841494

Mean:

- Spearman: 0.843419 ± 0.000531
- Pearson: 0.839528 ± 0.002141

Paired Spearman improvement vs FLaG:

- seed0: +0.001259
- seed1: +0.000591
- seed2: +0.005023

3/3 positive.

E12 is currently the highest-mean FLaG/STFT variant among the tested
three-seed variants, but it is NOT better than Mean pooling on STSB.

It is preferred as the frozen STFT candidate because it is also structurally
simple: no overlap, no Hann, no explicit frame position.

---

# 6. Window / overlap analysis

2x2 mean Spearman:

                   overlap       no overlap
    win = 8        0.843000      0.842166
    win = 16       0.842204      0.843419

Overlap effect:

- win8: +0.000834
- win16: -0.001215

Therefore overlap is not a generally beneficial mechanism.

For effective sequence length <=16, E12 has exactly one valid local frame.
This was independently audited and verified.

Thus the short-sequence gain does not require multi-frame overlap at inference.

---

# 7. Frequency and token probes

## E6: Final-layer DCT frequency knockout

DCT-II is applied to final RoBERTa hidden states.
Eight frequency bands are defined.

The lowest-frequency/DC band B0 causes by far the largest perturbation.

STFT E3 showed greater B0 sensitivity than FLaG in some aggregate statistics,
but this behavior varied across seeds.

Therefore:

- B0 sensitivity is a real observation.
- It should not be interpreted as a stable causal explanation of E3 gain.

---

## E7: Token knockout

Individual final-layer content-token hidden states are zeroed.

Mean positional response standard deviation:

- E3: approximately 0.001955
- FLaG: approximately 0.001956

No meaningful separation was observed.

Thus there is no evidence that the STFT gain comes from greater token-position
selectivity under this diagnostic.

---

# 8. Padding invariance

Global FLaG zeroes padding before FFT but performs FFT across the current batch
tensor length T.

Therefore changing external right-padding changes:

- FFT length
- frequency sampling grid
- frequency token count
- inverse reconstruction geometry

STFT uses fixed local FFT windows and per-sample frame masks.

Padding-invariance probe:

Global FLaG:

- measurable prediction drift under extra right-zero padding
- typical mean drift around 0.005 to 0.006

STFT:

- effectively invariant to additional right-zero padding for the same hidden states

This establishes an architectural difference, but not a performance cause.

---

# 9. E13: training-time fixed global FFT length

E13 uses global FLaG with:

    fixed_fft_length = 128

during both training and inference.

It remains a global transform and introduces no local windows.

Results:

seed0:
- Spearman 0.841107
- Pearson 0.838949

seed1:
- Spearman 0.841470
- Pearson 0.835893

seed2:
- Spearman 0.842991
- Pearson 0.838542

Mean:

- Spearman: 0.841856 ± 0.001000
- Pearson: 0.837795 ± 0.001660

Paired Spearman delta vs FLaG:

- -0.000445
- -0.001592
- +0.004222

Mean:

    +0.000728 ± 0.003079

Only 1/3 seeds positive.

E13 remains below:

- E3 by approximately 0.001144 mean Spearman
- E12 by approximately 0.001563 mean Spearman

Conclusion:

Training with a fixed global FFT length does not consistently reproduce the
STFT improvement.

---

# 10. Circular reflection induced by Re/Im asymmetric gating

A key architectural result was derived and empirically verified.

Suppose one hidden channel uses multiplier a for the real Fourier component and
multiplier b for the imaginary Fourier component.

Then inverse FFT can be written as:

    y_t =
        (a+b)/2 * x_t
        +
        (a-b)/2 * x_{(-t) mod N}

Thus independent Re/Im gating produces:

1. a direct channel scaling term
2. a circular time-reflection mixing term

The reflection depends explicitly on FFT length N.

This explains why changing padding / transform length changes global FLaG output.

---

# 11. Gate-length x reconstruction-length decomposition

Using a fixed trained FLaG checkpoint, a 2x2 counterfactual was evaluated.

Ng = FFT length used only to compute the latent gate
Nr = FFT length used for reconstruction

Modes:

- TT: gate at current T, reconstruct at T
- T128: gate at T, reconstruct at 128
- 128T: gate at 128, reconstruct at T
- 128128: gate at 128, reconstruct at 128

Across seeds, changing gate observation T -> 128 produced almost no change:

- gate cosine approximately 0.99999999
- prediction drift approximately 1e-7

Changing reconstruction T -> 128 produced:

- mean absolute prediction drift approximately 0.005 to 0.006

Therefore global FLaG padding-length sensitivity is dominated by reconstruction,
not by latent attention responding to a different Fourier grid.

---

# 12. Symmetric Re/Im gate counterfactual

The original gate multipliers are:

    a = 1 + g_R
    b = 1 + g_I

A symmetric counterfactual uses:

    shared = (a+b)/2

for both real and imaginary components.

Then:

    a = b

and the reflection coefficient becomes exactly zero.

Across three seeds:

- mean |SYM - TT| prediction drift:
  0.005576 ± 0.000761

- mean |T128 - TT|:
  0.005520 ± 0.000805

- mean |SYM - T128|:
  0.000295 ± 0.000003

Residual / T128 drift:

    0.0542 ± 0.0083

Correlation between pair-level SYM and T128 perturbations:

    0.99853 ± 0.00046

Same-sign pair fraction:

    0.9391 ± 0.0021

Thus removal of the circular reflection reproduces most of the fixed-128
reconstruction perturbation.

However SYM changes validation Spearman only on the order of 1e-4 and is not
consistently beneficial.

Therefore:

    circular reflection explains padding sensitivity,
    but does not currently explain the STFT performance gain.

---

# 13. Global/local gate x reconstruction decomposition

A second counterfactual decomposition was performed.

Notation:

First letter = source used to compute the gate.
Second letter = reconstruction operator.

- GG = global gate + global reconstruction
- LG = local gate + global reconstruction
- GL = global gate + local reconstruction
- LL = local gate + local reconstruction

This was evaluated using both:

- FLaG-trained weights
- E12-trained weights

Sanity checks were exact:

    manual GG == native global
    manual LL == native local
    source model == source operator

For FLaG-trained weights, mean three-seed prediction effects:

- local gate observation effect:
  approximately 2.86e-7

- local reconstruction effect:
  approximately 0.005770

For E12-trained weights:

- local gate observation effect:
  approximately 2.14e-7

- local reconstruction effect:
  approximately 0.005970

Thus reconstruction effects are roughly four orders of magnitude larger.

The local and global spectra generate essentially identical sample-level gates.

The architectural difference between global FLaG and E12 therefore comes
almost entirely from the local inverse reconstruction / temporal support, not
from a different gate being generated by local frequency tokens.

---

# 14. Training-regime x inference-operator analysis

Three-seed Spearman effects:

FLaG-trained weights:

    local operator - global operator
    -0.000166 ± 0.000187

E12-trained weights:

    local operator - global operator
    +0.000158 ± 0.000267

Training effect under global operator:

    +0.000182 ± 0.000263

Training effect under local operator:

    +0.000506 ± 0.000684

Training x operator interaction:

    +0.000324 ± 0.000439

Pearson interaction:

    +0.000089 ± 0.000481

The directions are compatible with the hypothesis that models trained under
local reconstruction adapt to that operator.

However the interaction is small and variable across only three seeds.

Therefore operator-specific training adaptation is a plausible hypothesis,
not a confirmed mechanism.

---

# 15. Current evidence hierarchy

## Strong architectural findings

1. Global FLaG depends on external FFT/padding length.
2. STFT variants are effectively invariant to additional right-zero padding.
3. Re/Im asymmetric gating induces a circular time-reflection term.
4. This reflection explains most of global FLaG's padding-length output sensitivity.
5. Global and local spectra produce almost identical learned sample-level gates
   in the tested checkpoints.
6. Global-to-local forward differences are dominated by reconstruction/operator
   changes rather than gate observation changes.

## Supported empirical findings

1. E3 and E12 both improve mean STSB Spearman over FLaG for seeds 0-2.
2. E12 is currently the strongest tested FLaG/STFT variant by mean Spearman.
3. E12 is simpler than E3 because it uses no overlap.
4. Fixed global FFT length does not consistently recover the STFT improvement.

## Not established

1. Local reconstruction is not proven to causally improve STSB performance.
2. Operator-specific training adaptation is not yet robustly established.
3. DC sensitivity is not a stable causal explanation.
4. Explicit positional selectivity is not established as the source of gain.
5. E12 is not the overall best STSB pooling method; Mean pooling remains better.

---

# 16. Why E12 is frozen

Current frozen candidate:

    STFT-FLaG
    win_length = 16
    hop_length = 16
    rectangular window
    no centering
    no overlap
    no frame positional encoding
    num_latents = 8
    num_heads = 4
    masked max pooling
    post_pool_norm = True

Reasons:

1. Highest mean Spearman among tested FLaG/STFT variants.
2. 3/3 paired Spearman improvements over FLaG for seeds 0-2.
3. No overlap.
4. No Hann window.
5. No additional positional parameters.
6. Mechanistically easier to interpret.
7. Avoids further STSB hyperparameter mining.

---

# 17. Next planned experiment

Do NOT continue sweeping STFT hyperparameters on STSB.

The next step is cross-task validation of the frozen E12 configuration.

Preferred next task:

    SprintDuplicateQuestions

Compare only:

1. Mean pooling
2. Original FLaG
3. Frozen E12 STFT-FLaG

Do not retune win/hop on the new task before evaluating the frozen E12 setting.

Questions:

1. Does E12 also improve over FLaG on a task where original FLaG was stronger?
2. Is the local-reconstruction effect task-dependent?
3. Does E12 preserve FLaG's strengths outside STSB?

If the frozen E12 configuration shows a useful signal on another task, then
consider a larger confirmatory seed evaluation.

---

# 18. Important caveats

- STSB Mean pooling remains clearly stronger than both FLaG and E12.
- Most mechanism experiments were developed after inspecting earlier STSB
  results. Treat them as exploratory.
- Three seeds are insufficient for strong statistical claims about small
  performance differences.
- Do not claim that E12's gain is caused by local reconstruction.
- The strongest current statement is that local reconstruction is the dominant
  architectural source of the representation difference between global and
  local variants.
