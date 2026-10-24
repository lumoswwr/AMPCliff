# STS-B 纯 LTR：修正版 Pairwise / Listwise 对照方案

## 研究问题和已有数据

问题不是单纯寻找最佳 loss，而是：
1. 将 MSE 换成**真正纯排序损失**，Mean、FLaG-Zero、FLaG 的验证 Spearman 是否提升？
2. **固定排序损失**时，FLaG / FLaG-Zero 是否优于 Mean？
3. FLaG 是否更容易提升训练 Spearman，却无法转化为验证 Spearman？

已经存在、**不重新训练**的基线：
- MSE：`outputs/text/stsb_soft_spearman_full_3models_t005_w005`。
- Pure Soft Spearman：`outputs/text/stsb_pure_softsp_cosine_v1`。
- 旧 LTR-v2 用于历史探索，不加入修正后“同协议”对照表。

## 修正版训练方法

新方法在 `text_repro/stsb_classic_ltr_losses.py`，全部运行在 **raw cosine** 上，固定 cosine temperature `T=0.05`，不使用可训练的 score affine scale/bias，也**不使用 MSE 反向传播**。

| 名称 | 类型 | 损失与纠错点 |
| --- | --- | --- |
| `ranknet` | Pairwise | 非并列标签对的 logistic RankNet，输入 `(cos_i-cos_j)/T`，不再使用旧版未缩放 logits |
| `hinge_fixed` | Pairwise | 在 logits 差上 margin=1，等价原始 cosine 差 margin=0.05；不再使用旧版 margin=1 raw scores |
| `hinge_label` | Pairwise | 在 logits 差上 `margin=0.25+0.75×abs(label_i-label_j)/5` |
| `lambda_ndcg` | Pairwise / NDCG-weighted | 以 batch 内 ∆NDCG 给 RankNet 加权；gain=`2^(label/5)-1`，normalized；**不是**标准 query-grouped LambdaRank |
| `listnet_kl` | Listwise | `KL(softmax((label/5)/0.20) || softmax(cos/0.05))`，标签 tie 天然等权，移除常数 target entropy |
| `grouped_listmle` | Listwise | 对相同标签先分组，按 relevance 组顺序优化 groupwise Plackett-Luce surrogate；组内无强制先后，不是精确的 tie-marginalized ListMLE |

第七个纯排序方法是已跑完的 `pure_softsp_cosine`（minibatch soft rank correlation，tie midrank），纳入同一汇总对照。

**注意**：STSB 是独立句对评分数据，并无自然 queryID。这里把每个随机 minibatch 视作一张训练“list”。不能声称等同于有真实查询组的搜索排序任务。

## 固定协议

- Models：`mean` / `flag_zero`（初始 Mean 等价性检查）/ `flag`。
- Seeds：0,1,2,3,4。
- 新增：3 × 6 × 5 = **90 次纯 LTR 训练**。与既有 3 × 2 × 5 个 MSE/PureSoftSP 对照合并，最终 **120 个 model×loss×seed 条件**。
- Encoder：RoBERTa-base，解冻，encoder LR=1e-5，pool LR=1e-3，Adam(weight_decay=0)。
- Batch=32，max length=128，最多20 epochs，至少12 epochs，patience5。
- Checkpoint 只由 Validation Spearman 选择（epoch0也纳入）。**从不读取 Test**。
- 所有新目标独立输出；不会覆盖旧实验。参数一致且有 checkpoint + history 的已完成运行会跳过。
- 第一个训练 batch 验证 encoder/pool 梯度有限且非零（MeanPooling 无 trainable pool 参数，正常跳过 pool 要求）。
- 记录每轮 rank loss / 训练子集 Spearman / Validation Spearman / 非训练用途 cosine-MSE / 有效 pairs。
- 最佳 checkpoint 选好后读取 train 全集，计算 Train Spearman 和 Train-only 正斜率仿射校准 MSE。这些**仅作分析诊断**，没有回传梯度。

训练代码：`text_repro/run_stsb_classic_ltr_suite.py`。

## 命令

```bash
cd ~/AMPCliff
git pull origin FLaG-STFT-mechanism
conda activate flag
python -m py_compile text_repro/stsb_classic_ltr_losses.py text_repro/run_stsb_classic_ltr_suite.py
python -u text_repro/stsb_classic_ltr_losses.py
python -u text_repro/run_stsb_classic_ltr_suite.py --dry_run
```

建议第一批先用 seed0 在 3 模型 × 6 方法上做 smoke test：

```bash
python -u text_repro/run_stsb_classic_ltr_suite.py \
  --models mean flag_zero flag \
  --seeds 0 \
  2>&1 | tee stsb_classic_ltr_seed0.log
```

无异常后继续 5 seeds 全量：

```bash
python -u text_repro/run_stsb_classic_ltr_suite.py \
  --models mean flag_zero flag \
  --seeds 0 1 2 3 4 \
  2>&1 | tee stsb_classic_ltr_all.log
```

已完成的 seed0 不重复训练。单独刷新汇总时：

```bash
python -u text_repro/run_stsb_classic_ltr_suite.py --summary_only
```

## 自动汇总表

所有数据齐全后输出独立目录：

`outputs/text/stsb_classic_ltr_pure_cosine_v1`

- `validation_all_methods_per_seed.csv`：每个模型每种 loss 每个 seed 的 Validation Spearman、best epoch、选中模型的 Train Spearman 和 Train-calibrated MSE（缺失字段为 NaN）。
- `validation_all_methods_summary.csv`：各方法与模型的 5-seed Validation SP mean±sd。
- `validation_all_methods_paired.csv`：同 seed 配对（纯 LTR−各自 MSE；FLaG−Mean 同 loss；纯 LTR−Mean-MSE）。
- 终端输出：`[CLASSIC_LTR_FULL_VALIDATION]`、`[CLASSIC_LTR_PAIRED_VS_MSE]`、`[CLASSIC_LTR_ARCHITECTURE]`。

分析请先看 **每种 loss 下的 Mean / FLaG-Zero / FLaG 排名**，再看同架构更换训练目标的配对 ΔSpearman，最后分析最佳模型的 Train−Val 差距。不能仅根据 Train Loss（跨方法量纲不同）排名。

**科学限制**：因为旧实验已接触 STSB Test，继续使用 Validation 选择方法也会增加多重比较的选择偏差。这轮不要碰 Test；如需外部泛化确认，应在固定方法选择后使用其他未用于开发的数据集。

## Disk / Error note

新增 90 组将分别保存含 RoBERTa-base 的 `best_model.pt`，检查服务器磁盘容量 `df -h .`。实验中断若落下非空但没有 metrics.json 的目录，脚本会拒绝自动覆盖，必须先检查或另选输出目录，避免误伤旧结果。
