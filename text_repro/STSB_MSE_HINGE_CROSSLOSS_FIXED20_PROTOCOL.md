# STSB: MSE 与 Hinge Fixed 双向 Loss 评估（固定 20 Epoch）

## 目的与六子图布局

对于 Mean、FLaG-Zero、FLaG，分别训练两套模型：

- **MSE 训练**：只对 MSE 执行 backward；同时记录 MSE 与 Hinge Fixed。
- **Hinge Fixed 训练**：只对 Hinge Fixed 执行 backward；同时记录 Hinge Fixed 与 MSE。

**2 × 3 六子图：**

|  | 单独训练方式 | 单独训练方式 | 双训练方式叠加（每张 6 条曲线） |
| --- | --- | --- | --- |
| **上排** | Train MSE → Eval MSE（3 Pooling） | Train MSE → Eval Hinge（3 Pooling） | 固定 **Eval MSE**；MSE/Hinge 两种训练方式 × 3 Pooling = 6 条 |
| **下排** | Train Hinge → Eval MSE（3 Pooling） | Train Hinge → Eval Hinge（3 Pooling） | 固定 **Eval Hinge**；MSE/Hinge 两种训练方式 × 3 Pooling = 6 条 |

颜色代表 Pooling（蓝 Mean / 橙 FLaG-Zero / 绿 FLaG），
线型代表训练目标（**实线 MSE / 虚线 Hinge Fixed**）。

## 协议

- STSB 官方 train / validation，**不读取 test**；
- RoBERTa-base Encoder 端到端微调；
- 3 Pooling × 2 训练目标 × seeds 0–4 = **30 次训练**；
- 每次训练完整 **20 Epoch**，无早停；
- 训练 batch 32，评估 batch 64，最大长度 128；
- Encoder lr = 1e-5，其余可训练参数 lr = 1e-3，Adam，weight decay 0；
- 每 Epoch 保留训练 Loss 和另一个交叉 Loss，均来自**同一次前向、同一训练 batch，optimizer.step() 之前**；交叉 Loss 的计算不参与梯度；
- 每轮计算 Validation Spearman，最佳 checkpoint 仍按该指标选择；
- **每一个 Epoch 都有完整 5 seeds 数据**，曲线为均值，阴影为 sample SD；
- 额外记录全程 raw-cosine MSE 作为校准差异的诊断。

### 重要：MSE Loss 的评分头口径

本实验**不改原 MSE/Hinge 的训练定义**：

- MSE 目标：\`MSE(score, labels/5)\`，
  其中 \`score=exp(log_scale)*cosine+bias\`，MSE 训练时 affine head 可学习。
- Hinge Fixed 目标：对 \`cosine / 0.05\` 计算 pairwise hinge，
  只比较标签不同的样本对，margin=1（logit 尺度）。Hinge 训练时
  affine head 冻结为 identity（scale=1, bias=0）。

故图中的 **Eval MSE** 始终使用上述 \`score\` 计算。
在 Hinge 训练模型上，它等价于 raw-cosine MSE；
在 MSE 训练模型上，它使用其学到的 affine head。
这两种模型的 **MSE 曲线不仅体现表示差异，也包含评分头校准差异**。
为避免误读，\`history.csv\` 和汇总 CSV 另存
\`raw_cosine_mse\`，可以在后续做不依赖 affine head 的对照。

尽管直接对比训练 Loss 有助于检查优化行为，**训练 Loss 更低并不直接证明
Test/Validation Spearman 更高**。不同定义的 MSE 与 Hinge 数值尺度也
不能互相比较。

## 服务器运行

\`\`\`bash
cd ~/AMPCliff
git pull origin FLaG-STFT-mechanism
conda activate flag

python -m py_compile text_repro/run_stsb_mse_hinge_crossloss_fixed20.py

# 初步检查公式、自检与参数；不训练
python -u text_repro/run_stsb_mse_hinge_crossloss_fixed20.py --dry_run

# 全部 30 次训练 + 自动绘制六子图，独立目录
mkdir -p outputs/text/stsb_mse_hinge_crossloss_fixed20_v1
set -o pipefail
python -u text_repro/run_stsb_mse_hinge_crossloss_fixed20.py \
  2>&1 | tee outputs/text/stsb_mse_hinge_crossloss_fixed20_v1/run.log
\`\`\`

仅验证所有训练记录，或重新绘图：

\`\`\`bash
python -u text_repro/run_stsb_mse_hinge_crossloss_fixed20.py --verify_only
python -u text_repro/run_stsb_mse_hinge_crossloss_fixed20.py --plot_only
\`\`\`

## 输出文件

\`\`\`text
outputs/text/stsb_mse_hinge_crossloss_fixed20_v1/
├── mean/
│   ├── mse/seed_0..4/{history.csv,metrics.json,best_model.pt}
│   └── hinge_fixed/seed_0..4/{history.csv,metrics.json,best_model.pt}
├── flag_zero/...
├── flag/...
├── crossloss_per_seed_epoch.csv
├── crossloss_5seed_mean_sd.csv
└── plots/
    └── mse_hinge_crossloss_six_panels.png
\`\`\`

- \`crossloss_per_seed_epoch.csv\`：30 运行 × 20 Epoch = 600 行；包含
  eval_mse、eval_hinge_fixed、raw_cosine_mse、Train sampled Spearman、
  Validation Spearman。
- \`crossloss_5seed_mean_sd.csv\`：各训练方法 × Pooling × Epoch 的
  五种子平均与 sample SD。
- \`mse_hinge_crossloss_six_panels.png\`：上述六子图。

## 失败与上传

脚本自动跳过已经完整结束的 runs。若中途断电，会拒绝覆盖
不完整 seed 目录，先备份并检查该目录，再单独移走失败目录重新运行。

只需上传图片与两张 CSV，**不要上传 RoBERTa checkpoint**：

\`\`\`bash
git add -f outputs/text/stsb_mse_hinge_crossloss_fixed20_v1/plots/
git add -f outputs/text/stsb_mse_hinge_crossloss_fixed20_v1/crossloss_per_seed_epoch.csv
git add -f outputs/text/stsb_mse_hinge_crossloss_fixed20_v1/crossloss_5seed_mean_sd.csv
git commit -m "Add STSB MSE Hinge cross-loss six-panel results"
git push origin FLaG-STFT-mechanism
\`\`\`

本次只向 AMPCliff 添加代码和协议说明，**不修改 FLaG-STFT-Notes**。
