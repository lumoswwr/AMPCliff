# STSB 排序目标定向排查：从高性能 Mean-MSE 出发

## 本阶段的唯一目标

检验：**在已训练 Mean-MSE 的高性能 checkpoint 上加入排序优化，能否进一步提高 Validation Spearman？**

本阶段不使用 test 做选择，不扩大损失算法搜索网格，不覆盖已有 v1/v2 训练结果，也不把“LTR 必然超过 MSE”当作实验先验。

参考 v2 五种子均值：
- Mean-MSE：Validation Spearman = 0.887813；Test Spearman = 0.859666。
- Mean-Hinge：Validation Spearman = 0.867925；Test Spearman = 0.832122。
- Mean-ListNet：Validation Spearman = 0.870815；Test Spearman = 0.831439。
这些数值的比较仅用来说明动机。**后续模型和超参数只用验证集选择。**

## 主线筛选

- **本轮退出：** raw-cosine RankNet、当前 LambdaRank-style（多 seed 的 train objective 降低，但 val Spearman 持续恶化）。
- **暂停：** Scaled RankNet（FLaG-Zero 的 seed 敏感性大）。
- **本轮主线：** Pairwise Hinge、ListNet。
- **待修复后再参与：** ListMLE。v2 在 STSB 相同标签时强制采用 minibatch 顺序，不是正确的并列标签建模。

## 四个控制组

每个 seed、每组模型均从 **相同的 v2 Mean-MSE 最佳 checkpoint** 开始，用一样的 batch shuffle seed、学习率、最大 epoch 和 patience：

| 组名 | 训练目标 | 唯一改变 |
| --- | --- | --- |
| `mse_continue` | MSE | 继续训练的对照组 |
| `hinge_only` | 纯 Hinge | 保留强 Mean-MSE 初始化，但去掉 MSE 约束，单独隔离 warm-start 影响 |
| `mse_hinge_all` | MSE + 0.05 × Hinge | 对所有非并列标签样本对施加排序约束 |
| `mse_hinge_gap` | MSE + 0.05 × Hinge | **仅对标签差值 ≥0.5 的 pair** 施加约束 |
| `mse_listnet` | MSE + 0.05 × ListNet KL | 将标签统一缩放到 [0,1]，与预测评分尺度一致 |

具体 Loss 保持拆分：`mse_hinge_gap` 的间隔仍为 1，只修改参与 pair 的标签差值阈值；`mse_listnet` 使用 `softmax(labels/5)` 和 `softmax(score)`，这是和 v2 原版 ListNet **不同的明确命名版本**。

**实验不声称 Hinge/Ranking loss 必然提高 Spearman。** 要对比的是 `mse_continue` 和其他分支，而不是将新方法的 8-epoch 训练与旧 Mean-MSE 的 3-epoch 训练混为一谈。

## 预训练 / checkpoint 控制

- **Mean：** 从 `outputs/text/stsb_ltr_v2_e20_min12_pat5/mean/mse/seed_N/best_model.pt` 恢复 encoder 与评分头。
- **FLaG-Zero：** 同样加载这个 Mean 模型的 encoder 与评分头，FLaG_B2 池化层独立进行恒等初始化，并用现有数值检查验证函数起点等价。
- 两者对应 **从已训练 Mean checkpoint 出发**，不同于 v2 从 RoBERTa-base 训练所有模型的 protocol。
- 每个实验把 **epoch 0 原始 checkpoint** 作为合法候选。若排序微调没有任何 validation 收益，`best_epoch` 应为 0，而不是强行报告变差后的模型。
- 每轮记录 `train_mse`、`train_rank`、512 个固定训练样本上的 Spearman、完整验证集 Spearman、scale、梯度范数与 pair 保留数量。
- **不读取、不评估 test**。只有新协议和方法选择确定后，另行做最终测试。

## 运行命令

先做语法检查和训练前验证：

```bash
cd ~/AMPCliff
git pull origin FLaG-STFT-mechanism
conda activate flag
python -m py_compile text_repro/stsb_ltr_warmstart_probe.py
python -u text_repro/stsb_ltr_warmstart_probe.py --dry_run
```

初次先跑 Mean、seed 0、四组（Hinge 主线）：

```bash
python -u text_repro/stsb_ltr_warmstart_probe.py \
  --model mean --seeds 0 \
  --variants mse_continue hinge_only mse_hinge_all mse_hinge_gap \
  2>&1 | tee stsb_ltr_warmstart_probe_s0.log
```

比较输出：`outputs/text/stsb_ltr_warmstart_probe_w005_gap05/mean/<variant>/seed_0/{history.csv,metrics.json,best_model.pt}`。

优先判断哪些分支的 Validation Spearman 超过来自同一个 MSE checkpoint 的 epoch 0 和 `mse_continue`。**只选出明确改善验证集的损失，再扩大到 seeds 0–4 和 FLaG-Zero**。

如果 Hinge 主线没有正向结果，再单独测试已保留的 ListNet：

```bash
python -u text_repro/stsb_ltr_warmstart_probe.py --model mean --seeds 0 --variants mse_listnet
```

注意：输出目录一旦存在有效文件，不会覆盖；改变 rank_weight、min_label_gap 或学习率时，请明确使用不同 `--output_root`，防止不同协议混在一起。

## 下一步判别

- **MSE + Hinge 超过 MSE continuation：** 排序信号能够帮助强 Mean，进一步验证跨 seed 稳定性。
- **只有 gap-filtered Hinge 有效：** 说明对微小标签差异强行施加 margin 可能是先前的重要问题。
- **MSE + ListNet KL 有效：** 值得单独研究概率分布尺度与温度。
- **所有方法都未能改善：** 检查预训练任务、Pair/Batch 构造、评分头约束和梯度对齐，而不是再机械地增加 epoch 或 Loss 搜索数量。
- **FLaG-Zero 弱于 Mean：** 继续检查新增 Gate/Projection 参数的训练更新；暂不能归因于“排序 Loss 本身无效”。

## 梯度诊断结果（Mean-MSE，seed 0，两个 batch，每批 16 条）

参考 `text_repro/diagnose_stsb_ltr_gradient_balance.py`，对同一次 forward 分别计算 MSE 和排序 Loss 的梯度（不更新参数，不访问 test）：

| Margin 策略 | violation rate (batch0 / 1) | Encoder grad cos with MSE (batch0 / 1) | 0.05×排序梯度范数 / MSE 梯度范数 |
| --- | --- | --- | --- |
| fixed 1 | 1.000 / 1.000 | 0.5343 / 0.4943 | 0.3555 / 0.3036 |
| fixed 0.1 | 0.209 / 0.229 | 0.3293 / 0.8196 | 0.0742 / 0.1097 |
| abs(label gap)/5 | 0.617 / 0.644 | 0.8569 / 0.8868 | 0.2976 / 0.2713 |
| fixed 1, label gap ≥0.5 | 1.000 / 1.000 | 0.5799 / 0.5610 | 0.4070 / 0.3637 |

**判断边界：** 原始 margin=1 使所有观察到的有效 pair 都存在间隔违例，提示与强 Mean-MSE 预测尺度不匹配；但是 Encoder 梯度与 MSE 并非整体方向相反（余弦为正）。两个 batch 不代表总体，需要验证集实验支持。

### 不覆盖旧数据的最小新实验

两组新的命名变体已经加入 `stsb_ltr_warmstart_probe.py`，分别为：

- `mse_hinge_margin01`：固定 margin=0.1，仍保留所有非同标签样本对。
- `mse_hinge_labelmargin`：margin=abs(label_i-label_j)/5，与 MSE 的标签归一化尺度一致。

两者都从**完全相同的高性能 Mean-MSE checkpoint** 出发，并使用相同的 MSE + 0.05×Hinge、训练轮次和学习率。原实验组和结果不改变。

```bash
cd ~/AMPCliff
git pull origin FLaG-STFT-mechanism
conda activate flag
python -m py_compile text_repro/stsb_ltr_warmstart_probe.py
python -u text_repro/stsb_ltr_warmstart_probe.py \
  --model mean --seeds 0 \
  --variants mse_hinge_margin01 mse_hinge_labelmargin \
  2>&1 | tee stsb_ltr_margin_probe_s0.log
```

检查两个新变体的 `metrics.json` 和 `history.csv`。若改善，不要直接以单 seed 宣布超越 Mean-MSE，应在提前固定协议后继续做五种子验证；最终 test 仍仅做一次验证。若仍无改善，先考虑更小的排序 loss 权重或从原始训练起点联合训练，而不是继续机械增大 margin/epoch。
