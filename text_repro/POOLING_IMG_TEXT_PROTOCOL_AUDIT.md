# Pooling-img-text protocol audit for AMPCliff text experiments

**Audit basis**: `lumoswwr/Pooling-img-text`, public `main` tree SHA
`6f634253609c8639d98dc287f3612cbfc006ab7c` (checked Oct 15, 2026).
Current local reimplementation: `lumoswwr/AMPCliff`,
`FLaG-STFT-mechanism`.

**Scope**: source-code protocol comparison, not an independent rerun of
the original publication and not a claim of reproducing published test scores.
Do not conflate experiment variants with original FLaG.

## Strongly supported by source code

| Item | Original STSB / STS E2E | Original Sprint Pair | Local reproduction / B2 | Match? |
|---|---|---|---|---|
| RoBERTa | base, unfrozen | base, frozen | same for each; new trained-Mean B2 STSB deliberately frozen | Base protocols yes; frozen STSB pilot intentionally different |
| Epochs | 3 | 10 | 3 / 10 | Yes |
| Train batch size | 32 | 32 | 32 | Yes |
| Eval batch size | 64 | 64 | 64 | Yes |
| Token max_length | 128 | 128 | 128 | Yes |
| Optimizer | Adam | Adam | Adam | Yes |
| Backbone LR | 1e-5 | frozen | same | Yes |
| Pool/head LR | 1e-3 | 1e-3 | same | Yes |
| Weight decay / scheduler | 0 / null | 0 / null | 0 / none | Yes |
| Loss | MSE; docs say target=score/5 | BCEWithLogits | MSE with score/5; BCEWithLogits | Objective types match; original label normalization needs unavailable data-loader code to independently verify |
| Pair encoder/head | RoBERTa bi-encoder, L2 normalize, positive affine cosine | same as logit | same | Yes |
| STSB model selection | validation Spearman | n/a | validation Spearman | Yes |
| Sprint model selection | n/a | **validation accuracy** default | exploratory runs report AP; explicit checkpoint metric varies by script | **Not fully matched** |
| Source evaluation | validation during training; training script also evaluates final model on test | same | exploration uses validation only | Not directly comparable to original test results |

Primary files:
- Original: [STSB YAML](https://github.com/lumoswwr/Pooling-img-text/blob/main/conf/mteb_sts_e2e.yaml)
- Original: [Sprint YAML](https://github.com/lumoswwr/Pooling-img-text/blob/main/conf/mteb_pair_classification.yaml)
- Original: [STSB grid script](https://github.com/lumoswwr/Pooling-img-text/blob/main/scripts/run_mteb_sts_e2e_10seeds.sh)
- Original: [Sprint grid script](https://github.com/lumoswwr/Pooling-img-text/blob/main/scripts/run_mteb_sprint_dup_10seeds.sh)
- Original: [Training implementation](https://github.com/lumoswwr/Pooling-img-text/blob/main/scripts/train.py)
- Local: [Author reproduction](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_author_text_protocol.py)
- Local: [Exploration](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/explore_mean_equiv_trajectory.py)

## Critical differences: original FLaG vs Mean-equivalent B2

Source original [FFT pooler](https://github.com/lumoswwr/Pooling-img-text/blob/main/src/pooling/llm_pooling_dropin.py)
and original STSB/Sprint YAML:
- `time_pool=max`, not Mean.
- post-pool LayerNorm enabled (`fft_latent_gate_use_pool_norm` defaults true).
- residual gate `1+sigmoid(gate_logits)`, not B2's
  `2*sigmoid(gate_logits)`.
- learned final output projection initialized normally, not explicitly
  identity at start.
- Gate final layer is not zero initialized in original.

Local [B2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_author_text_protocol.py)
deliberately sets:
- `time_pool=mean`, `post_pool_norm=False`,
  `gate_parameterization=centered_sigmoid`,
  zero-initialized Gate output and identity-initialized projection.
- Therefore B2 can start numerically identical to Mean. **It is a
  mechanism-control variant, not original FLaG replication**.
- Original FLaG adaptation in AMPCliff is another implementation, with
  expanded experiment options; parameter matching alone is NOT a bitwise
  reproduction of the original class.

## Important unresolved issues

**Original public source does not include dataset/collator implementation
in the checked tree.** The original training script imports
`src.data.factory.DatasetFactory`, but `src/data/factory.py`
is not present in that GitHub tree. Consequently source alone currently
cannot prove exact padding strategy, tokenization batching, sentence-A/B
padding symmetry, or exact seeded split implementation.

The local `PairCollator` explicitly tokenizes `s1+s2` together, with
`padding=True`, `truncation=True`, `max_length=128`.
This dynamically pads both sentence sides to the maximum token length
across the combined batch. **Batch size 32/64 is not a padding length**.

STSB official train+dev sizes agree with local 5749/1500. Original documentation
describes using official STSB dev for validation. Sprint documentation
describes adaptation from HF `validation` (90/10 internal split), but
without `src/data/factory.py`, the exact split generator and local
data version remain to be verified against author artifacts.
Local exploration concatenates locally stored train+validation then
re-splits at each seed; compare dataset identity and indices before
claiming *exact* matched splits.

The shell grid scripts document eight poolings in some comments, while their default POOLINGS lists contain five entries; an explicit environment override can change the set. The claimed exact original run therefore also needs the resolved CLI/Hydra config, not just comments.

The original training script records checkpoints based on validation
Spearman for STSB and validation accuracy by default for Sprint; it
also evaluates the model at the end of training on test, while a separate
evaluation script may reload the best checkpoint. Do not conflate these
test protocols with the B2 validation-only exploratory results.

## Image protocol (not replicated in this text-only pilot)

The original [image Hydra config](https://github.com/lumoswwr/Pooling-img-text/blob/main/conf/config.yaml)
defaults to CIFAR10, ResNet18, training batch128 / eval256, 250 epochs,
SGD lr0.1, momentum0.9, weight_decay=5e-4, cosine schedule with warmup,
and validation split 10%. Original [image grid script](https://github.com/lumoswwr/Pooling-img-text/blob/main/scripts/run_cifar_pooling_grid_top5_10seeds.sh)
runs CIFAR10/100 with ResNet18 and FLaG pool override
`d_model=512,num_heads=4,num_latents=8,time_pool=max,residual=true,dropout=0`.
The local STSB/Sprint scripts are **not** an image-paper reproduction.
Image dataset transforms, seed-specific indices, and exact test
checkpoint protocol require a separate check if image reproduction is requested.

## Next decisive cheap test

Run:
```bash
cd ~/AMPCliff
git switch FLaG-STFT-mechanism
git pull
conda activate flag
export PYTHONPATH="$(dirname "$PWD"):$PWD/text_repro:${PYTHONPATH:-}"
python -m py_compile text_repro/check_stsb_padding_sensitivity.py
python -u text_repro/check_stsb_padding_sensitivity.py
```
This performs **no training**. It uses the previously saved **LAST** frozen
B2 checkpoint and trained Mean reference, and compares:
1) cached identical encoder hidden states with different amounts of
   literal zero padding only at the pooling stage;
2) same STSB validation subset (default 256 pairs) evaluated at dynamic
   batch sizes 1/32/64 vs fixed 128 length and batch64.

Its output `outputs/text/stsb_frozen_trained_mean/seed_0/padding_sensitivity.json`
separates isolated pooler effects from end-to-end effects. Mean is a
negative control. With default 256 pairs, any Spearman is subset-only.
Pass `--n_pairs 0` to evaluate all 1500 validation pairs.

This check can identify padding sensitivity but cannot by itself prove
that all or most observed generalization differences are caused by padding.
