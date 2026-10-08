# FLaG 机制深入实验（基于现有 checkpoint，无需重新训练）

## 为什么需要这一轮

已有的 seed-0 证据：
- FLaG 的最终转角来自 gate 与 time_out_proj，而不仅是 FFT。
- B2 从 Mean 严格初始化，但 epoch1 DC Gate 已大幅改变方向。
- Sprint test：FLaG AP=0.848503，B2 AP=0.833946。
- B2 关闭 DC Gate 后 AP=0.678850；关闭 AC Gate 后 AP=0.835823。
- FLaG/B2 的 sigmoid 输出大多处于饱和端（约 96%～98% 高于 0.95）。
这些是观察事实；“大转角不利性能”和“Gate 已变静态”尚未被证实。

## 关键数学结果

原始 Masked 输入 x（形状 [B,N,D]）在 FFT 前有右侧 padding 并置零。
当前配置的 Gate 对所有频率 bin 使用同一组 g_R 和 g_I，
将复频谱实部和虚部分别相乘。固定这两组 Gate 后，有精确等价式：

    y[t] = (g_R+g_I)/2 * x[t] + (g_R-g_I)/2 * x[(-t) mod N]

注意：N 是实际的 batch FFT 长度，而不是 token 有效长度。
如 g_R=g_I，y[t]=g_R*x[t]，FFT 路径退化为简单的通道缩放。
如果所有时间位置都是有效 token，那么时间 Mean = g_R*Mean(x)，
与 g_I 无关。若是 right-padding 且 N >= 2*L，反转后的 masked mean
等于第一个 token 向量除以 L，因此：

    pooled = (g_R+g_I)/2 * masked_mean(x) + (g_R-g_I)/2 * x[0]/L

上述公式只对“固定 Gate 的调制+iFFT+Masked Mean”成立。
完整 FLaG 仍通过 FFT tokens、latent attention、MLP 动态生成 Gate。
本目录的纯 torch 单测覆盖了偶数/奇数 N、右 padding、
没有 padding、DC 能量夹角恒等式等，避免只凭直觉解释。

## 本轮四个可被证伪的假设

### H1: 输入自适应 Gate 可能退化为近似固定的通道掩码
比较同一个已训练 checkpoint 在以下推理时干预：
- full：正常用当前句子生成 Gate；
- train_mean_gate：先从训练集独立采样的 512 对句子估计平均 Gate，测试时固定；
- within_batch_swap：句子 A 使用同一批里其他句子的 Gate；
- scalar_gate：当前样本 Gate 变成跨通道单标量；
- gate_identity：全 1 Gate。

还直接计算：动态 Gate 到训练集平均 Gate 的偏差、
门控二值选择模式翻转比例（阈值为原始 FLaG 1.5、B2 1.0）。
如果训练集静态 Gate 与原始 Full 的排序分数一致、换句子 Gate 也不变，
才支持“动态生成机制的实际贡献较小”，仅凭 sigmoid 饱和无法证明。

### H2: 少量受抑制的通道可能承载大量 DC 能量
把第 0 频率 bin 的实部作为 DC（等价于有效 token sum）。
分别记录 real 与 imag 部分受到抑制的通道比例（sigmoid<0.05），
以及被抑制 **real** 通道占原始 DC 的能量比例。
同样计算“Gate 最低 5% 的 real 通道”占 DC 能量的比例。

理论上：令 w_i=x_i²/Σx_j²，那么

    cos(theta_DC) = E_w[g_R] / sqrt(E_w[g_R²])

如果 B2 的 77° 夹角主要来自少数高能通道被压低，
将体现为 suppressed_real_dc_energy_fraction 较高。
注意以前的 4% 是 2D real+imag 全部维度统计，并非仅 DC-real。

### H3: 复杂 FFT 路径可能主要是通道缩放与 padding 反转项
- full：解析时域公式等于原始模型频域 forward，脚本硬校验。
- imag_equals_real：将虚部 Gate 强制与实部 Gate 相等，消掉反转项；
- no_pad_fixed_gate：以当前 Gate 固定的反事实，时间 Mean 等于 g_R * Mean；
- long_pad_fixed_gate：Gate 固定，强制考虑 N >= 2L 的解析极限。

对比分数排序变化和输出向量夹角，并检测 g_R/g_I 差异。
请勿把固定 Gate 的 padding 反事实解读为完整模型重新编码后的结果。

## 使用方法

    cd /home/data/home/wwr_lumos/AMPCliff
    git pull origin FLaG-STFT-mechanism
    conda activate flag

先做最短测试：

    TASKS=stsb METHODS=FLaG_B2 TRAIN_PAIRS=32 EVAL_PAIRS=64 \
      bash text_repro/run_flag_deep_mechanism.sh

正式验证集（先跑，只涉及两份 checkpoint，无需训练）：

    TASKS="stsb sprint" METHODS="FLaG FLaG_B2" SPLIT=validation \
      TRAIN_PAIRS=512 EVAL_PAIRS=0 \
      bash text_repro/run_flag_deep_mechanism.sh

随后完整 test（只有在验证集实验正常时才执行）：

    TASKS="stsb sprint" METHODS="FLaG FLaG_B2" SPLIT=test \
      TRAIN_PAIRS=512 EVAL_PAIRS=0 \
      bash text_repro/run_flag_deep_mechanism.sh

输出目录：outputs/text/flag_deep_mechanism_seed0/
生成 deep_validation.json / deep_test.json，以及对应的汇总 .md。

## 阅读结果时的原则

1. 先确保 `[identity]` FFT forward 与解析时域 forward 误差很小。
2. 对 validation full 模式有历史 metric 检查，误差 >0.002 将终止。
3. STSB 主要看 Spearman；Sprint 主要看 AP，而不是固定 0.5 阈值 F1。
4. 只对相同 checkpoint 内的反事实结论负责；
   推理时禁用模块会造成分布偏移，不能代表重新训练同一简化结构的结果。
5. 所有统计从 seed0 开始，只是机制诊断。论文级判断需多 seed，
   并在独立测试集与不同 batch/padding 约定下复核。
6. 不要把 train_mean_gate 当成额外可训练参数：它从训练集估计而来，
   只用于评估“现有 Gate 是否强依赖当前句子”。

## 如何向学姐汇报（当前已证实 vs 待测）

已证实：该 Gate 在所有频率 bin 上共享实虚门控系数，
频域调制与逆变换具有精确时域反转混合等价式。
已有实验表明 Gate 与 projection 都能显著影响表示，
而在 Sprint 上关 AC Gate 的影响很小，关 B2 的 DC Gate 影响很大。
尚待验证：静态掩码是否足以保留性能、被抑制通道是否
承载绝大多数 DC 能量，以及 padding 反转项造成多少任务差异。
