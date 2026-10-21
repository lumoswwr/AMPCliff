# STSB LTR-v2: 统一训练预算的完整重跑

## 为什么需要 v2

诊断实验采用 STSB 官方 train/validation 划分、seed=0、Pairwise Hinge、完整 5749 条训练数据，分别得到：

| Model | Best validation Spearman | Best epoch |
| --- | ---: | ---: |
| Mean | 0.86348653 | 11 |
| FLaG-Zero (FLaG_B2) | 0.82947686 | 11 |

两者最优 epoch 均超过旧协议的 3 个 epoch；这支持“3 epochs 可能不足”的假设，但不证明所有排序目标都已收敛，也不能证明 FLaG-Zero 优于 Mean。FLaG-Zero 的初始 Mean 等价数值检查通过（池化最大绝对误差 2.8610e-6、cosine 最大绝对误差 1.7881e-7）。

## 实验设置

- **入口：** `text_repro/run_stsb_ltr_v2.py`
- **输出：** `outputs/text/stsb_ltr_v2_e20_min12_pat5/`。和 v1 的 `outputs/text/stsb_ltr_ranknet_standard/` 完全隔离。
- **模型：** Mean、FLaG-Zero、原始 FLaG。
- **损失：** MSE（同预算对照）+ 六个 LTR：RankNet、Scaled RankNet、Pairwise Hinge、ListNet、ListMLE、LambdaRank-style。合计 3 × 7 × 5 = 105 组。
- **Seeds：** 0、1、2、3、4。
- **数据：** STSB train / validation / test；正式训练只使用 train，checkpoint 按 validation Spearman 选择。
- **训练：** Adam、encoder LR=1e-5、pool/head LR=1e-3、batch size=32、weight decay=0、max length=128。
- **Epoch 预算：** `--max_epochs 20 --min_epochs 12 --patience 5`。至少训练到第 12 轮，此后连续 5 轮 validation Spearman 不刷新最佳值时停止，训练绝不超过 20 轮。
- **Checkpoint：** 每个 epoch 检查 val Spearman，保存最佳 epoch 的 `best_model.pt`。**只在选完 checkpoint 后计算一次 test 指标**。
- **恢复与安全：** 已有完整 `metrics.json` 的运行检查训练预算参数后跳过；未完成但非空的输出目录不会自动覆盖，需人工检查。
- **监控：** 各 seed 写出 `history.csv`、`metrics.json`、`best_model.pt`。全部 105 组完成后才生成 `summary.csv`、`paired_differences.csv` 和 `summary.md`。
- **不变因素：** 所有 loss 的实现与 v1 保持一致。本实验**只改变训练轮数和早停标准**，便于定位原来 3 epochs 的影响。

## 重要限制

1. **MSE 必须作为匹配预算的对照。** 将 v2 LTR 和旧版 3-epoch MSE 对比，无法区分训练轮数和损失设计的影响。
2. **ListMLE 并列标签缺陷仍存在。** 为了不将训练轮数变化和 Loss 公式修订混在同一轮实验，本次有意保留旧实现：当 STSB 标签并列时，ListMLE 沿 minibatch 的随机顺序进行排列学习。它并非严格的 tie-aware ListMLE，后续需要独立修正版对照。
3. **STSB 不是 query-grouped 排序数据集。** ListNet、ListMLE、LambdaRank-style 在本仓库采用随机 minibatch 作为排序列表；不能当作标准查询分组 LTR benchmark 解读。
4. **RankNet（原始）沿用 raw cosine**，无法训练 affine score scale；Scaled RankNet 则允许训练尺度。两种都保留。
5. **即使 Spearman 没有超过 MSE，也不必然代表训练代码错误。** 各个排序 surrogate 并不等同于全数据 Spearman。
6. **最佳 epoch 贴近 20 轮上界**时，要进一步检查是否存在右截断，不能直接宣布模型收敛。

## 一次性运行

先检查命令与 Loss 数学函数（不启动训练）：

```bash
cd ~/AMPCliff
git pull origin FLaG-STFT-mechanism
conda activate flag
python -m py_compile text_repro/run_stsb_ltr_v2.py
python text_repro/run_stsb_ltr_v2.py --dry_run
```

正式跑全部 105 组：

```bash
python -u text_repro/run_stsb_ltr_v2.py \
  --output_root outputs/text/stsb_ltr_v2_e20_min12_pat5 \
  --max_epochs 20 \
  --min_epochs 12 \
  --patience 5 \
  2>&1 | tee stsb_ltr_v2_run.log
```

推荐在 tmux 会话执行，避免 SSH 掉线使训练中断。脚本顺序运行，不会自行占用多个 GPU。

如需先检查一个条件，**使用独立预试验目录**，避免将不完整的预试验或不同参数混入正式结果：

```bash
python -u text_repro/run_stsb_ltr_v2.py \
  --models mean --losses pairwise_hinge --seeds 0 \
  --output_root outputs/text/stsb_ltr_v2_smoke
```

## 对比重点

首先检查 105 组的 `best_epoch`、`epochs_ran`、训练损失曲线与 Validation Spearman，重点关注是否普遍在第 12 轮之后继续改善。再比较**同预算**下 LTR 相对 MSE、FLaG-Zero 相对 Mean、原始 FLaG 相对 Mean 的逐 seed 差值。不要先根据测试集选 Loss、epoch 或模型。
