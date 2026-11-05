# STSB 固定 20 Epoch 对齐实验（不使用 Early Stopping）

## 目的

为精确比较 Mean / FLaG-Zero / FLaG 的训练 Loss 与 Spearman 曲线，
重新运行第四阶段的 **3 模型 × 8 目标 × 5 seeds = 120 runs**。

- 每个 run **完整训练 Epoch 1–20**，绝不提前停止。
- 每一 Epoch 的均值和标准差都含 **同一组完整的 5 个 seeds**。
- 不更换 Loss、Encoder、batch、随机种子或学习率。
- 每轮仍计算 Validation Spearman，**仍以最佳 Validation Epoch 保存 best_model.pt**。
  完整训练到 Epoch 20 不意味着必须选择 Epoch 20 模型。
- **不评估 Test**；与此前早停实验的 Validation/Test 指标区分存放。

## 如何保证不提前停止

保留已验证的原始训练器，统一传入：

- \`--epochs 20\` / \`--max_epochs 20\`
- \`--min_epochs 20\`
- \`--patience 21\`

原代码停止条件为 \`epoch >= min_epochs and waiting >= patience\`。
因为 \`waiting <= epoch <= 20\`，第二个条件不可能满足；
因此 **这次等效于彻底关闭 Early Stopping**，无需改动原训练逻辑。
输出中的 \`epochs_ran\` 应严格为 **20**。

这不是从过去的 \`best_model.pt\` 继续训练。过去通常没有保存完整的
optimizer/scheduler/RNG 中间状态，不能准确续接被早停中断的训练。
新实验从同一预训练 RoBERTa-base 起点、同样的 seed 重新运行，
保留旧数据供对照。

## 一键运行（GPU 服务器）

在 \`AMPCliff\` 仓库根目录：

\`\`\`bash
git pull origin FLaG-STFT-mechanism
conda activate flag

# 三个训练器的 Loss 自检与参数预检；不训练
python -u text_repro/run_stsb_fixed20_no_earlystop.py --dry_run

# 所有结果写入一个新的、固定名称的独立目录
mkdir -p outputs/text/stsb_fixed20_no_earlystop_v1
set -o pipefail
python -u text_repro/run_stsb_fixed20_no_earlystop.py \
  2>&1 | tee outputs/text/stsb_fixed20_no_earlystop_v1/run.log
\`\`\`

程序依次执行：

1. MSE：3 模型 × 5 seeds = 15 runs。
2. Pure Soft Spearman：3 模型 × 5 seeds = 15 runs。
3. 六种经典 LTR：3 模型 × 6 Loss × 5 seeds = 90 runs。
4. 检查全部 120 条训练记录完整性，自动画图。

所有方法最终一起比较，没有混用过去早停结果。

## 检查与单独重画

\`\`\`bash
python -u text_repro/run_stsb_fixed20_no_earlystop.py --verify_only
python -u text_repro/run_stsb_fixed20_no_earlystop.py --plot_only
\`\`\`

验证通过会打印：

\`\`\`text
[FIXED20_VERIFIED] 120/120, each contains 20 train epochs, 5/5 seeds for each model/loss; Test untouched.
\`\`\`

绘图保存于：

\`\`\`text
outputs/text/stsb_fixed20_no_earlystop_v1/plots/
├── pooling_overlay_train_loss_fixed20.png
├── pooling_overlay_train_spearman_fixed20.png
├── pooling_overlay_validation_spearman_fixed20.png
├── per_seed_best_vs_epoch20.csv
└── summary_best_vs_epoch20.csv
\`\`\`

Loss 图的每个小图对应相同的训练目标，三条颜色分别代表 Mean、
FLaG-Zero 和 FLaG。Loss 类型之间定义和尺度不同，**只在同一 Loss
的小图内比较三种 Pooling**。阴影表示五种子的 sample SD。

训练 Spearman 曲线来自旧协议一致的 **固定 512 个训练样本抽样**，
不是完整训练集 Spearman；其曲线标注为 Sampled Train Spearman。
Validation 为 STSB 官方验证集指标。

## 中途失败和重跑

所有已完成的 runs 都有 \`metrics.json\`、\`history.csv\`、
\`best_model.pt\`，下次再次启动会在协议匹配时自动跳过。
如果在一个 seed 的训练中途被打断，原训练器会拒绝覆盖这个
**尚未完成**的目录。请先确认具体目录、备份必要日志，然后只处理
该单独未完成的 \`seed_N\` 目录，不要删除整批已完成的 runs。

完整 RoBERTa checkpoint 占用较大，**不要把所有 best_model.pt 推送到 GitHub**。
同步用于报告的图片和汇总表即可：

\`\`\`bash
git add -f outputs/text/stsb_fixed20_no_earlystop_v1/plots/
git commit -m "Add full-20-epoch STSB loss and Spearman plots"
git push origin FLaG-STFT-mechanism
\`\`\`

## 与报告的关系

这是一批新的 **fixed-20 / no-early-stop** 实验，其最佳 checkpoint
仍根据 Validation Spearman 选择。若将来需要测试集指标，
须单独用新 checkpoint 做相同的冻结 Test 评估。
在新一轮 Test 结果出来之前，**不要将旧报告的 Test 数值当作新实验的 Test 数值**。

这一阶段先不修改 \`FLaG-STFT-Notes\` 的第四阶段页面。
