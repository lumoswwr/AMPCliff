# 配对种子实验：FLaG 的收益是否主要来自可训练 Projection？

## 动机

seed0 旧协议 test 指标：

| Method | STSB Spearman | Sprint AP |
| --- | ---: | ---: |
| FLaG-B2 | 0.845490 | 0.834217 |
| StaticReIm | 0.841483 | 0.820715 |
| StaticDiag | 0.840198 | 0.824215 |
| MeanProjection (identity init) | 0.841056 | 0.833765 |

注：旧 Sprint checkpoint 选模使用验证集 Accuracy；以下新协议改为验证集 AP，
新结果不应与上述数值直接混用。为了同条件对照，所有模型都重新训练。

原始 residual-sigmoid FLaG 在旧 seed0 Sprint test AP 大约 0.8485，
高于 B2/MeanProjection。因此必须比较 FLaG vs **随机初始化 Projection 的 Mean**，
否则随机初始化和 Gate 结构存在混杂。

## 四个主要模型

- FLaG: residual sigmoid gate，默认 random linear Projection，原始结构；
- FLaG_B2: centered sigmoid，zero-init gate final linear、identity Projection；
- MeanProj_B2: masked Mean + 可训练 identity-init Projection；
- MeanProjRand: masked Mean + 可训练 random-init Projection。

另有可选 StaticFLaG_ReIm_B2 与 StaticFLaG_Diag_B2，
但优先四个模型，防止浪费 GPU。

一个确定的数学事实：

    W (diag(g) m) + b = Wprime m + b, Wprime = W diag(g).

因此，静态对角 Gate + 不受约束可训练线性 Projection 不增加函数表达能力，
其效果差异可能来自初始化、优化过程或正则化，而非模型函数空间更大。

对冻结 RoBERTa 的 Sprint，上述 Mean+Projection
可以解释为一种可学习的余弦相似度度量（当偏置为 0，内积由 W^T W 诱导）。

## 协议

- STSB: 3 epochs，backbone 端到端微调，validation Spearman 选 checkpoint。
- Sprint: 10 epochs，backbone frozen，**validation AP** 选 checkpoint。
- 训练数据拆分、seed、time_pool=mean、no postpool LN、dropout 和学习率使用既有协议。
- 通过参数开关改变 Sprint 选模指标，旧脚本默认 validation Accuracy 不改变。
- 此处全部放到新的输出目录，保留先前所有模型及其结果。
- 汇总只会读取此目录、同一个 seed 的模型，并计算严格配对差值。

## 运行

    cd /home/data/home/wwr_lumos/AMPCliff
    git pull origin FLaG-STFT-mechanism
    conda activate flag

先测一个新对照，确认可以训练：

    SEEDS=0 TASKS=sprint METHODS=MeanProjRand \
      bash text_repro/run_flag_paired_ap_multiseed.sh

正式 Sprint 三个种子，配对四个主要模型：

    SEEDS="0 1 2" TASKS=sprint \
      METHODS="FLaG FLaG_B2 MeanProj_B2 MeanProjRand" \
      bash text_repro/run_flag_paired_ap_multiseed.sh 2>&1 | tee flag_paired_sprint_3seed.log

若结果稳定，再扩到 STSB 或 5 seed：

    SEEDS="0 1 2" TASKS=stsb \
      METHODS="FLaG FLaG_B2 MeanProj_B2 MeanProjRand" \
      bash text_repro/run_flag_paired_ap_multiseed.sh 2>&1 | tee flag_paired_stsb_3seed.log

    SEEDS="3 4" TASKS="sprint stsb" \
      METHODS="FLaG FLaG_B2 MeanProj_B2 MeanProjRand" \
      bash text_repro/run_flag_paired_ap_multiseed.sh 2>&1 | tee flag_paired_seeds34.log

单纯汇总全部五个 seed：

    python text_repro/summarize_flag_paired_multiseed.py \
      --seeds "0 1 2 3 4" --tasks "sprint stsb"

运行目录：outputs/text/flag_paired_ap_multiseed/
汇总：outputs/text/flag_paired_ap_multiseed/paired_multiseed_summary.md

## 解释时不能犯的错误

1. FLaG 与 MeanProjRand 的差异，才较直接检验原始 FLaG
   在随机 Projection 初始化条件下是否有额外价值。
2. B2 与 MeanProj_B2 的差异，较直接检验 Mean 起点上的频域/动态 Gate。
3. 目前 seed0 结果不能代表随机种子分布，报告配对均值、标准差、
   每个种子的方向与 checkpoint best epoch，不仅报告最好的一次。
4. 不能把旧“validation Accuracy 选模”与新“validation AP 选模”混合求均值。
5. STSB backbone 随池化方法共同微调，因此不能完全将收益归于 readout；
   Sprint backbone frozen，机制归因较为清晰。
6. 之前已多次查看同一个 test 集来调整研究方案，正式论文应视这些 test
   分析为探索性结果，采用预先固定实验方案或新任务/外部测试集做最终验证。

