# STSB Soft Spearman 正式三模型 × 五种子验证集实验

## 依据与研究问题

来自独立 seed 0 小实验，使用训练中联合优化的 minibatch soft midrank Spearman loss：
- Mean MSE validation Spearman = **0.887250**；
- MSE + 0.01 × Soft Spearman = **0.887653**（+0.000403）；
- MSE + 0.05 × Soft Spearman = **0.888585**（+0.001334）。

这是 seed0 上观察到的微小提升，尚不能宣称方法有跨种子收益，更不能宣称测试集收益。本次预先固定 T=0.05、辅助损失权重=0.05，不再以测试集调参。

## 固定实验协议

新入口：`text_repro/run_stsb_soft_spearman_full.py`。

- Models：`mean`、`flag_zero`（FLaG_B2，初始 Mean 等价性数值检查）、`flag`（原始 FLaG）。
- 每模型 Objectives：`mse_scratch`、`mse_softsp_w005`。
- Seeds：0、1、2、3、4。**3 models × 2 objectives × 5 seeds = 30 runs**。
- 每组都从对应 seed 的 RoBERTa-base 预训练编码器重新训练；FLaG-Zero **不**从已训练 Mean checkpoint 热启动。
- Data：STSB train / validation（正式训练脚本**完全不读取 test**）。
- Frozen encoder：否（端到端），Adam，encoder LR=1e-5、pool/head LR=1e-3；batch 32、max length 128。
- Training budget：最多 20 epochs、至少 12 epochs、验证集 5 轮不提升则早停，checkpoint 以 validation Spearman 选 best epoch，epoch 0 也在候选内。
- Combined objective：`MSE(affine_score, label/5) + 0.05 × (1 - corr(soft_ranks(affine_score, T=0.05), midranks(label)))`。
- Soft rank 在每个训练 minibatch 内计算，使用原先已验证的 soft-midrank 函数。目标标签并列的真实秩采用 average ranks，未声称这是全数据精确 Spearman。
- 每组 seed 使用固定初始化 seed 与独立 shuffle generator。保持与 MSE 完全相同的数据、架构和训练预算。

## 安全与复现

- 全部新结果写入：`outputs/text/stsb_soft_spearman_full_3models_t005_w005/`；不会覆盖 LTR-v1/v2、warm-start、小样本 Soft Spearman。
- 已经完成且协议参数相符的单组会跳过；不完整非空目录会拒绝覆盖，需要人工检查。
- 记录完整 `history.csv`、`metrics.json`、`best_model.pt`。30组齐全之后，写出 `summary_val.csv` 和 `paired_val_differences.csv`。
- 配对差包括 (a) 同模型 Soft Spearman − MSE，(b) 该模型 Soft Spearman − Mean-MSE；每个 seed 保留单独差值。
- 对首个 Mean-MSE seed0 进行复现核对：参考前次 `0.887250`，若明显不一致，先定位协议/代码差异，避免继续浪费 GPU。
- 注意 0.0013 的单 seed 改善较小，应报告五个 seed 的配对均值、标准差、改善 seed 数。**不能根据 Test 选择权重或重新调整模型。**

## 一次性完整运行

```bash
cd ~/AMPCliff
git pull origin FLaG-STFT-mechanism
conda activate flag
python -m py_compile text_repro/run_stsb_soft_spearman_full.py
python -u text_repro/run_stsb_soft_spearman_full.py --dry_run
```

```bash
python -u text_repro/run_stsb_soft_spearman_full.py \
  --models mean flag_zero flag \
  --variants mse_scratch mse_softsp_w005 \
  --seeds 0 1 2 3 4 \
  2>&1 | tee stsb_soft_spearman_full.log
```

输出末尾有 `[FINAL_VALIDATION_SUMMARY]` 和 `[PAIRED_VALIDATION_DELTAS]`。在此阶段直接把这两段终端文字（或这两个汇总 CSV）发送进行分析即可。

## 后续 Test 评估

待此组固定协议的五种子验证集结果汇总后，冻结方法选择和模型选择规则，再编写**独立的只读最终 Test 评估**，按 validation 选好的 `best_model.pt` 一次性计算测试指标。避免继续使用测试集挑选训练策略。正式论文中同时披露 seed0 试验的超参数选择来源和验证集结果。
