# STSB: 从预训练 RoBERTa 开始的 Mean-MSE + LTR 联合训练

## 动机：Warm-start LTR 负迁移

Mean-MSE v2 最优 checkpoint（seed 0）Validation Spearman = 0.887250。

从这一 checkpoint 微调、初始最佳轮次为 0：

| 方法 | 第三轮 Val SP | 最佳 Val SP / Epoch |
| --- | ---: | --- |
| MSE 继续训练 | 0.88603 | 0.887250 / 0 |
| 纯 Hinge，margin=1 | 0.84261 | 0.887250 / 0 |
| MSE+0.05 Hinge，margin=1 | 0.88476 | 0.887250 / 0 |
| MSE+0.05 Hinge，margin=0.1 | 0.88602 | 0.887250 / 0 |
| MSE+0.05 Hinge，margin=abs(label_i-label_j)/5 | 0.88567 | 0.887250 / 0 |

这说明 **从强 MSE checkpoint 继续排序微调未获得验证收益**，但尚不能证明从训练开始联合优化同样无效。

## 本轮设计

入口：`text_repro/probe_stsb_ltr_joint_from_scratch.py`。

四种损失独立从同一个 RoBERTa-base 预训练权重和相同 seed 开始：

- `mse_scratch`：MSE on `score` 与 `label/5`（严格匹配的基线）。
- `mse_hinge_margin01`：MSE + 0.05×固定 margin=0.1 Hinge，所有非并列 pair。
- `mse_hinge_labelmargin`：MSE + 0.05×标签差值 /5 的 Hinge，所有非并列 pair。
- `mse_listnet`：MSE + 0.05×归一化标签 `softmax(labels/5)` 对 `softmax(score)` 的 KL。

**固定训练协议：** Mean pooling、seed 0（先做初筛）、Adam、encoder lr=1e-5、head lr=1e-3、batch 32、max 20 epochs、至少 12 epochs、patience 5；按验证集 Spearman 选最佳 epoch。所有组的随机 seed、数据打乱方式和模型初始化一致。该脚本不会评估 test，且把 epoch 0 作为可选择的候选 checkpoint。

**诊断日志：** 训练 MSE、训练 Rank Loss、固定 512 条训练样本的 Spearman、完整验证 Spearman、scale、梯度范数和有效 pair 数。新输出位于 `outputs/text/stsb_ltr_joint_scratch_v1_w005/`，与 v1/v2 和 warm-start 所有结果隔离。

## 命令

```bash
cd ~/AMPCliff
git pull origin FLaG-STFT-mechanism
conda activate flag
python -m py_compile text_repro/probe_stsb_ltr_joint_from_scratch.py
python -u text_repro/probe_stsb_ltr_joint_from_scratch.py --dry_run

python -u text_repro/probe_stsb_ltr_joint_from_scratch.py \
  --model mean --seeds 0 \
  --variants mse_scratch mse_hinge_margin01 mse_hinge_labelmargin mse_listnet \
  2>&1 | tee stsb_ltr_joint_s0.log
```

## 结果解释与下一步

要报告每组相同 seed 下的最佳 Validation Spearman、对应 epoch、最后一轮 Train MSE 和 Train/Val Spearman，而不是只比较 Loss 数值的下降。

- 某 LTR 联合训练组超过 **匹配的 `mse_scratch`**：保持完整验证协议，之后扩大 5 seeds 确认，确认后再讨论 FLaG-Zero。
- 两者相当或 LTR 更低：不要宣称 LTR 训练失败；要进一步审视 label ties、pair sampling、rank surrogate 与 Spearman 指标之间的差异。
- 若同脚本 `mse_scratch` 与 v2 同 seed Mean-MSE 相差很大：先定位初始化、optimizer、数据和 batch 顺序的复现差异。
- 不因为 test 更好而回头选择权重；本阶段**不访问 test**。

注意：**LTR 并无普遍优于 MSE 的数学保证**，这轮检验的是“联合训练时机”能否带来提升。
