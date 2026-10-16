# E3c: Real/imaginary Gate symmetry and padding sensitivity

## Mechanism being tested

The exact STSB FLaG_B2 Gate learns 2D channel multipliers shared by every
Fourier frequency bin, not a separate multiplier for each frequency.
The real and imaginary multipliers can differ (a != b).

For a real-valued x and FFT length T, scaling the real coefficients by a
and the imaginary coefficients by b gives

    irfft(a*Re(rfft(x)) + i*b*Im(rfft(x)))[t]
       = (a+b)/2 * x[t] + (a-b)/2 * x[-t mod T]

If a = b, the operation is equivalent to channelwise scaling in the time
domain. That is still not necessarily Mean pooling: a and b can vary by
sentence and feature channel, and Projection is learned. Exactly a=b=1
and Projection=identity reproduces original Mean.

With right padding and masked mean, a != b may make the output sensitive
to FFT length through circular time reversal.

## E3c protocol

Replay E0's exact first 10 Adam steps at seed 0 and require Steps 0, 1
and 10 to reproduce archived E0 per-pair cosine and predictions, plus
recorded metrics and minibatch losses. No independent training.

At the verified Step10 model use INFERENCE-ONLY final-Gate-output hooks,
without editing any weights:

1. original_gate: learned a,b.
2. tied_real_gate: set imaginary b=a, preserving learned real a and DC.
3. identity_gate: set a=b=1, retaining learned Step10 Projection.
4. mean_current_head: bypass B2 pool/Projection, retain Step10 affine Head.
5. padded_fixed_gate: append 32 right zero hidden tokens and zero mask
   AFTER RoBERTa and reuse original unpadded a,b.
6. padded_recomputed_gate: same padding but recompute a,b from longer FFT.
7. padded_tied_fixed_gate: same padding but b=a using unpadded real a.
   This should match unpadded tied-real scores up to float noise.

All variants use the SAME original RoBERTa hidden token representations
and the SAME Step10 affine Head. Padding is appended AFTER the frozen
RoBERTa, excluding encoder differences from the comparison.

Report Spearman/MSE/Pearson, cosine/rank changes, a/b asymmetry statistics,
and the exact symmetric-Gate fixed-logit padding invariant discrepancy.

## Commands

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:$PYTHONPATH"

    python -m py_compile text_repro/e3c_gate_symmetry_padding.py
    python -u text_repro/e3c_gate_symmetry_padding.py --seed 0 --check-only
    python -u text_repro/e3c_gate_symmetry_padding.py --seed 0

First test seed 0 only.

## Fixed output and safeguards

    outputs/text/stsb_mean_equiv_e3c_gate_symmetry/seed_0/
      e3c_step10_gate_symmetry.json

No modifications to E0/E1/E2/E3a/E3b or trained Mean checkpoint.
Check-only writes NO result. Full output uses exclusive creation,
cannot overwrite prior results and creates no timestamp folders.

## Interpretation

If tying b=a recovers Spearman, real/imag asymmetry contributes at
Step10. This does NOT prove the training optimizer should learn equality:
training optimizes MSE rather than validation Spearman.

Padding with FIXED a,b isolates FFT/iFFT + masked mean effects,
and padding with RECOMPUTED a,b also tests how FFT length alters the
learned Gate's input. This is a one-checkpoint controlled forward
intervention, not a new optimizer trajectory or untouched test split.
