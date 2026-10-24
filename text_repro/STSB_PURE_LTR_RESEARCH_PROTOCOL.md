# FLaG / STSB：恢复研究主线，比较真正的纯排序 Loss

## 核心科学问题

原始机制研究的现象是在既定训练配置下，**FLaG 的训练 MSE 可能更低，但验证 Spearman 不及 Mean**。我们要解释“额外表达能力为何改善拟合、却不改善排序泛化”，以及改成纯排序目标能否改变三个池化模型的相对表现。

### 明确区分三个训练目标

1. **MSE：** `L = MSE(affine_score, y/5)`。
2. **Hybrid：** `L = MSE(affine_score, y/5) + 0.05×SoftSpearman(affine_score, y)`。
3. **Pure ranking（本脚本）：** `L = 1 - Pearson(soft_rank(cosine,T), exact_midrank(y))`。不包含任何 MSE 梯度。

之前完成的 MSE / Hybrid 的五种子实验不能回答“纯 LTR 直接替代 MSE 后表现如何”。Soft Spearman 属于直接针对名次相关性的可微排序 surrogate，而不是传统查询分组的 RankNet / LambdaRank。STSB 不包含 query/document 分组，不能宣称实现了标准 query-grouped LTR。

## 本轮固定协议

- 新入口：`text_repro/run_stsb_pure_softspearman.py`。
- 3 模型：`mean`，`flag_zero`（FLaG_B2，训练前检查 Mean 函数等价），`flag`（原始 FLaG）。
- 5 个 seeds：0–4，**总共 15 次全新纯排序训练**。
- 全部从同一种 RoBERTa-base 预训练编码器按 seed 初始化、端到端训练。FLaG-Zero 是*与未训练 Mean 在初始化时函数等价*，本阶段**没有**加载一个训练好的 Mean-MSE checkpoint。
- 训练目标只对 cosine 计算 SoftRank：使用 `T=0.05`，对真实标签 ties 使用正确的平均名次。未使用 train MSE 参与反向传播。
- `log_scale`、`bias` 固定不训练。排名不受正斜率仿射变换影响，因此这个操作可避免学习评分 scale 去隐性改变温度；`raw_cos_mse` 仅诊断，**不可直接与 MSE 模型的 fitted head MSE 横向比较**。
- Adam，encoder lr=1e-5，pool lr=1e-3，batch=32，max length 128，最多20 epochs，至少12 epochs，patience5，基于验证 Spearman 选择 best epoch。初始 epoch0 也是候选。
- **不读取 test**。保存 best checkpoint / history / metrics。选好 checkpoint 后单独计算完整 train Spearman 和 train-calibrated MSE 用于诊断（train 上拟合正斜率仿射映射）。
- 独立输出目录 `outputs/text/stsb_pure_softsp_cosine_v1`，拒绝覆盖非空残缺目录。已有参数完全匹配的完成运行会跳过。
- 汇总时只读取之前的 MSE 实验结果 `outputs/text/stsb_soft_spearman_full_3models_t005_w005` 作参照；不修改旧 checkpoint，不复用旧测试集结果选择超参数。

## 命令

```bash
cd ~/AMPCliff
git pull origin FLaG-STFT-mechanism
conda activate flag
python -m py_compile text_repro/run_stsb_pure_softspearman.py
python -u text_repro/run_stsb_pure_softspearman.py --dry_run
```

初步检查纯排序训练是否正常，可以只跑三个模型 seed 0（在正式目标目录内，后续自动跳过这些完整运行）：

```bash
python -u text_repro/run_stsb_pure_softspearman.py \
  --models mean flag_zero flag --seeds 0 \
  2>&1 | tee stsb_pure_softsp_seed0.log
```

若梯度正常、Validation Spearman 随训练上升，再跑完整协议：

```bash
python -u text_repro/run_stsb_pure_softspearman.py \
  --models mean flag_zero flag --seeds 0 1 2 3 4 \
  2>&1 | tee stsb_pure_softsp_15runs.log
```

**将程序结尾的** `[PURE_LTR_VALIDATION_SUMMARY]` **和** `[PURE_LTR_PAIRED_VS_MSE]` **两段输出发回即可。**

## 科学结论判读

优先看**三模型在同一个纯排序 Loss 下**的验证 Spearman，随后检查每种模型从 MSE 到 Pure LTR 的配对变化。

- 若 Pure LTR 下 FLaG 更接近或超过 Mean，说明排序目标有可能让额外自由度得到更有用的更新，还需额外独立数据检验；
- 若 Pure LTR 下 FLaG 仍差于 Mean，重点检查 Gate / Projection 的梯度、偏离速度、训练拟合与验证排序的分歧；
- 若纯排序损失本身难以收敛，要从优化难度和排序代理温度入手排查，不得自动将 Hybrid 的成绩算作 Pure LTR。

## 两个需要分开做的后续协议

本轮是 **end-to-end 的架构对比**。原始“FLaG 已包含 Mean，却产生无效偏离”的因果机制研究，仍需另做**冻结 encoder、从训练好的高性能 Mean checkpoint 出发**的严格 Mean 等价实验。不能混淆这两种初始化。

此外，此项目先前已多次查看 STSB 测试集结果，本轮应以验证集进行开发；如需完全独立的外部泛化证据，应采用其他未用于选择方法的数据集或固定好的外部测试协议。
