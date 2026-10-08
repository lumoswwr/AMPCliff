# FLaG 机制判断：从已训练 Gate 的静态化到可训练的简化对照

## 已经有的 seed0 数据（不要误说成新结果）

- Train-calibrated mean Gate 和 within-batch gate swap 在 STSB/Sprint validation/test 的
  Spearman/AP 到 6 位小数完全复现 Full。
- deep_mechanism 的 Gate 差异打印到 5 位小数显示 0.00000；这不是 bitwise 证明。
- Sprint test 原始 FLaG 有 2.214% 的 real 通道处于低 Sigmoid Gate 区间，
  这些通道占原始 DC 能量 93.720%；B2 的 3.776% 对应 95.488%。
- 原始 residual sigmoid gate 的低值是 g~1，而非 0；B2 则接近 g~0。
- 原始 FLaG/B2 的 Sigmoid 大部分通道处于两端饱和区间。
- Sprint test 原始 FLaG AP=0.848503；B2 AP=0.833946。
- 去掉 AC Gate 和把 real/imag Gate 设为相同，不会显著降低 Sprint 性能，
  但 STSB B2 的 AC 门控仍有可见影响；不可说所有非 DC 都无用。

## 为什么还需要这轮

实验支持“已经训练好的 checkpoint 在推理时几乎只需要一套静态 Gate”，
但不支持“可训练的静态池化一定能达到同样的效果”。
这轮通过三种参数显式缩减的训练控制补充因果证据。

### 实验 A：高精度校验（无需重新训练）

命令（新版本 deep probe 记录额外精度和跨句子通道方差）：

    TASKS="stsb sprint" METHODS="FLaG FLaG_B2" SPLIT=validation \
      TRAIN_PAIRS=512 EVAL_PAIRS=1024 bash text_repro/run_flag_deep_mechanism.sh

重点输出 JSON 或终端中：

- dynamic_gate_max_abs_delta_vs_train_mean
- dynamic_gate_bitwise_diff_fraction
- dynamic_gate_fraction_above_1e-7
- dynamic_gate_fraction_above_1e-5
- logits_between_sentence_channel_sd_mean
- gate_between_sentence_channel_sd_mean

分辨 logits 是随输入变化、但经过饱和 sigmoid 被压扁，
还是从输入到输出都几乎恒定；避免只看平均值的近似 0。

### 实验 B：从零训练不使用 FFT 的简化对照

三种可训练池化模式：

- StaticFLaG_ReIm_B2：可训练静态 2D gate，解析式用
  a*maskedMean + b*maskedReverseMean，不需要 FFT/attention/MLP。
- StaticFLaG_Diag_B2：可训练静态 D gate，直接用 gate*maskedMean。
- MeanProj_B2：没有 gate，但接一个可训练、初始为单位矩阵的
  D×D projection。对比判断 gate 的贡献是否只是 projection 学到的。

三种模式的 epoch0 输出都严格为 masked Mean（算子层面），
projection 可训练，与 B2 的投影初始化方式保持一致。
STSB 仍是 3 epoch RoBERTa 端到端训练，Sprint 仍是
冻结 RoBERTa 10 epoch，原数据集/划分、优化器、head 配置。

先只跑 seed0：

    SEEDS=0 TASKS="stsb sprint" \
      bash text_repro/run_flag_static_mechanism.sh 2>&1 | tee flag_static_seed0.log

结果：

    outputs/text/flag_static_mechanism/static_mechanism_summary.md

如果 seed0 与 B2 接近，再跑种子 1 2 3 4。为了形成配对比较，
同时重跑相同设置的 B2，不能只增加静态模型的种子：

    SEEDS="1 2 3 4" TASKS="stsb sprint" \
      METHODS="FLaG_B2 StaticFLaG_ReIm_B2 StaticFLaG_Diag_B2 MeanProj_B2" \
      bash text_repro/run_flag_static_mechanism.sh 2>&1 | tee flag_static_4seed.log

重新汇总全部 0..4 种子：

    python text_repro/summarize_flag_static_mechanism.py --seeds "0 1 2 3 4"

注意 B2 目前仅 seed0 的 matched init/angle checkpoint。多 seed 正式对照
仍需要同训练协议的 B2 种子 1..4；汇总会对缺失的 B2 标记 n=1。
切勿把 B2 n=1 与 static n=5 的均值说成完整配对多 seed 消融。

## 科学结论边界

1. 训练后交换 Gate 与静态化不变：说明**已训练实例**不依赖动态 Gate，
   但不能证明 latent attention 在训练时没有塑造出这套掩码。
2. 少数通道集中大部分 DC 能量：支持强各向异性，
   但不能把这些维度直接称为无用或噪声。
3. StaticReIm 接近 B2：说明该复杂路径不是达到性能的必要条件之一
   （至少在该任务/seed/协议下），但不同模型优化难度需要多 seed 检验。
4. StaticDiag 接近 StaticReIm：支持实虚差异/反转项非关键；
   不支持跨所有 pooling 算法的一般频域无用结论。
5. MeanProj 接近 StaticDiag：说明表观 Gate 收益可能主要来自 Projection，
   若出现显著差异，说明静态通道筛选仍提供额外信息。

## 学姐汇报

当前可以稳健地说：我们发现 FLaG 的共享频率通道 Gate 在 seed0 已训练
checkpoint 上有显著饱和及近乎输入不变的现象，且通道低值群
高度集中于原始 DC 的高能量维度；频谱实虚共享门控还存在精确的
时域等价公式。这不证明频域结构没有训练作用，因此我们补充
从零训练的静态 ReIm、静态 Diag、Mean+Projection 三组对照。

在完整对照跑完前，不要报告简化模型的性能或者准确率收益。
