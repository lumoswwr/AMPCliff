# E6: One decision experiment for useful direction vs excess amplitude

## Main scientific question

Why does the trained-Mean-containing FLaG_B2 struggle to turn extra
representation freedom into stable task benefit?

This is a **decision experiment**, not another attempt to rank Gate
and Projection by embedding displacement.

E5 found two Step10 module-related prediction changes often move in
the same direction. Positive squared-error interaction largely comes
from 2*dGate*dProjection, rather than functional nonadditivity, which
partially offsets that effect.

## Predeclared hypothesis

If the full model's deviation direction is useful but the amplitude is
excessive, a smaller shift chosen using TRAIN ONLY should beat the full
B2 on validation. For a *genuinely useful* deviation, it must ALSO beat
the original trained Mean, not merely recover toward Mean.

All five seeds (0-4) run in ONE command. Exactly replay E0 ten-step
Adam training from pretrained Mean, frozen RoBERTa, common Step10 Head.
Require replay checks against E0 and four endpoint cross-checks against
E4. No further model training or checkpoint saving.

## Two mechanistically different controls

### Raw-embedding interpolation

Let m be the unnormalized pretrained Mean, g be the trained
Gate+Latent preprojection output, and P(z)=Wz+b trained Projection:

    z_alpha=m+alpha*(g-m)
    u_alpha_beta=z_alpha+beta*(P(z_alpha)-z_alpha)
    e_alpha_beta=L2_normalize(u_alpha_beta)

Compute cosine of each sentence pair under SAME Step10 Head.
(0,0)=Mean; (1,0)=Gate-only; (0,1)=Projection-only; (1,1)=full B2.

Predeclared grid: alpha,beta each in {0,.25,.5,.75,1}.
Additional diagonal point (.125,.125) tests early joint attenuation.

Choose TRAIN-only best MSE and best Spearman in both 2D and diagonal
sets, then evaluate those fixed choices on validation, comparing each
to BOTH Mean and full B2. Validation is never used for selection.

### Pure prediction-space amplitude control

M is Mean score, GP is full score under the same Head; d=GP-M.
For score_lambda=M+lambda*d and label y:

    L(lambda)=mean((M-y+lambda*d)^2)
    dL/dlambda at 0 = 2*mean((M-y)*d)
    lambda_star_unconstrained = -mean((M-y)*d)/mean(d*d)

Find the optimum on TRAIN analytically, clip to [0,1], and evaluate on
validation WITHOUT re-selecting it. No grid search is needed.

For the TRAIN unconstrained optimum:
- <=0: original mean-to-full score direction is locally harmful to MSE
- between 0 and 1: finite-amplitude overshoot ALONG this score direction
- >=1: no overshoot on the examined interval

Validation optimum is reported descriptively ONLY, never used as a
selection criterion.

Both train and validation inference use batch64/dynamic padding.
E5 evaluated TRAIN with batch32, so E5 and E6 TRAIN metrics need not be
bitwise identical. E0 optimizer replay still uses original batch32.

## Decision rule / stopping condition

1. Beating full B2 without beating Mean is only damage recovery.
2. Train-selected choices must beat Mean consistently on validation to
   support recoverable useful deviation; independent holdout and
   cross-task confirmation are needed for a general conclusion.
3. If geometry interpolation helps but output scaling does not,
   investigate representational geometry; if score shrinkage suffices,
   investigate calibration before architectural changes.
4. If the experiment fails to produce a robust Mean-beating result,
   STOP further STSB-only mechanism micro-ablations and review E0-E6
   with the senior student/supervisor to develop a cross-task plan.

The pretrained Mean was validation-selected, so this is not a
completely untouched external evaluation. Five seeds are not five
independent task datasets.

## Server commands

    cd ~/AMPCliff
    git switch FLaG-STFT-mechanism
    git pull --ff-only
    conda activate flag
    export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:$PYTHONPATH"

    python -m py_compile text_repro/e6_direction_vs_amplitude.py
    python -u text_repro/e6_direction_vs_amplitude.py --check-only
    python -u text_repro/e6_direction_vs_amplitude.py

No --seed flag. All seeds in one invocation.

## One exclusive output

    outputs/text/stsb_mean_equiv_e6_overshoot/
        e6_five_seed_direction_vs_amplitude.json

All train/validation grid values, train-only choices, analytic score
optima, replay/cross-checks, and per-seed summary are in this JSON.
Existing E0-E5 outputs are never modified or overwritten.
