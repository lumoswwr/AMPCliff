# STSB: 从 MSE + 经典 LTR 转向直接近似 Spearman 的小实验

## 当前依据（Mean，seed 0）

所有条件从相同的 RoBERTa-base pretrained weights 开始，训练协议一致，validation 选择 checkpoint：

| 方法 | Best Validation Spearman | Best Epoch | 最后一轮训练子样本 Spearman |
| --- | ---: | ---: | ---: |
| Mean-MSE | 0.887250 | 7 | 0.982730 |
| MSE + Hinge (margin 0.1) | 0.886799 | 7 | 0.983066 |
| MSE + label-aware Hinge | 0.886719 | 7 | 0.983009 |
| MSE + normalized ListNet | 0.886957 | 7 | 0.983042 |

增加经典排序目标有实际优化作用，但没有验证集增益。不能因为 LTR 理论目标与排名相关就假定其一定优于 MSE，也不能仅凭 Loss 下降认为代码有误。训练 Spearman 来自固定 512 条 train 子样本；验证 Spearman 来自完整验证集，切勿把两者当作同一数据集直接比较。

## 下一步的唯一变化：Tie-aware Soft Spearman

Spearman 定义为真实名次与预测名次的 Pearson 相关系数。真正的 rank 不可导。当前用 minibatch SoftRank 作为平滑代理：

- 预测 SoftRank：`0.5 + sum_j sigmoid((s_i-s_j)/T)`。
- 真实标签使用准确的 **average/mid rank** 处理并列标签，保证方向和并列语义与 scipy Spearman 相符。
- 排序正则为 `1 - Pearson(soft_prediction_rank, exact_label_midrank)`。
- `T=0.05`。模型评分输出尺度约在归一化 STSB [0,1] 范围，但实际不受硬限制。
- 完全同标签 batch 的排序项返回 0，跳过无定义的排序相关性。
- 这是 **batch 级可微近似**，不是全 STSB 的精确可导 Spearman。仍存在 batch 采样噪声，并不能保证超过 MSE。

三个对照组均从同一预训练 RoBERTa-base 初始化，以验证集选最优 checkpoint，不评估 test：

| Variant | Training objective |
| --- | --- |
| `mse_scratch` | MSE |
| `mse_softsp_w001` | MSE + 0.01×Soft Spearman |
| `mse_softsp_w005` | MSE + 0.05×Soft Spearman |

其余严格匹配前一轮联合训练：seed0、Adam、encoder lr=1e-5、head lr=1e-3、batch size32、最多20轮、至少12轮，patience5。保留 history 和独立 checkpoint，输出目录 `outputs/text/stsb_soft_spearman_v1_T005`，旧数据永不覆盖。

## 运行

```bash
cd ~/AMPCliff
git pull origin FLaG-STFT-mechanism
conda activate flag
python -m py_compile text_repro/probe_stsb_soft_spearman.py
python text_repro/probe_stsb_soft_spearman.py --dry_run

python -u text_repro/probe_stsb_soft_spearman.py \
  --model mean --seeds 0 \
  --variants mse_scratch mse_softsp_w001 mse_softsp_w005 \
  2>&1 | tee stsb_soft_spearman_s0.log
```

程序末尾会打印 `[FINAL_VALIDATION_COMPARISON]`；请同时保留每轮 Train Spearman / Validation Spearman / Train Rank Loss 的输出以检查轨迹。

**Success criterion**：同 seed 下最佳 val Spearman 超过同脚本 `mse_scratch`，然后预先固定超参数做 5-seed 验证；最终 Test 只能在确定方法之后作为一次性评测。单 seed 微小改善可能是噪声，不应直接用于论文优越性结论。

**Failed condition**：若 Soft Spearman 仍无验证集提升，下步要先分析验证集排序错序位置、训练 batch 的分组采样偏差和优化泛化，不继续无目的扫超参。
