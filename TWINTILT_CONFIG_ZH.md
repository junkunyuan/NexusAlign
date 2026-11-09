# TwinTilt 关键配置说明

本文说明 TwinTilt 训练时的采样与 group exact projection 参数。
配置文件：[configs/algorithm/twintilt.yaml](src/nexus_align/configs/algorithm/twintilt.yaml)。

```yaml
algorithm:
  solver: euler
  optimization_level: trajectory
  churn_weight: 0.04
  eta: 1.0
  u_scale: 0.1
  exact_projection_iterations: 5
  guidance_scale: 1.0
```

## 1. `optimization_level: trajectory`

同一个 group 的样本共享相同的**基础初始噪声**，再通过逐成员独立的
perturbation（随机扰动）生成不同样本。扰动后，各成员的实际输入会不同。

**扰动强度：** `churn_weight` 控制随机扰动的强度，默认是 `0.04`。扰动既包括初始化时的
扰动，也包括采样中间步骤的随机扰动；最后一步不额外加噪。

**零扰动：** 当 `churn_weight=0` 时，同组成员沿相同的采样轨迹生成，不再通过这些扰动
引入样本差异。不同 group 分别抽取自己的基础初始噪声。

## 2. `optimization_level: distribution`

同一个 group 的各个成员直接使用**不同的、独立抽取的初始噪声**，
跳过 trajectory 模式的初始 perturbation。

**中间扰动：** 此模式仍保留由 `churn_weight` 控制的中间步骤扰动。如果只希望随机性来自
独立初始噪声，配置为：

```yaml
algorithm:
  optimization_level: distribution
  churn_weight: 0
```

## 3. `eta`：奖励倾斜强度

`eta` 控制 group exact projection 中对高奖励样本的偏好强度：

\[
q = \operatorname{softmax}(\eta a)
\]

**奖励信号：** 其中，\(a\) 是送入投影求解器的组内奖励信号；PickScore 默认先经过组内
z-score，不裁剪。`eta` 越大，目标分布越偏向奖励高的成员；越小则越接近均匀分布。

**默认值：** 默认 `eta=1.0`。显式设为 `null` 时使用 `1 / group_size`。

## 4. `u_scale`：correction 的缩放系数

求解器先得到 correction \(u\)，再用 `u_scale` 缩放实际施加的位移：

\[
\Delta = \text{u\_scale}\cdot u,
\qquad
\text{target} = v_{\mathrm{rollout}} - \Delta
\]

**默认值：** 默认 `u_scale=0.1`。它控制求解完成后 correction 的幅度；`eta` 则控制
求解目标中的奖励倾斜。位移外部不再额外乘一次 `eta`。

## 5. `exact_projection_iterations`：数值求解迭代次数

表示 group exact projection 求数值解时执行的 Newton 更新次数，默认是 `5`。
每次更新内部保留 Armijo 回溯来选择步长。

**采样步数：** 这个参数控制投影求解器的迭代次数，图像采样步数由 `num_steps` 控制。

## 6. `guidance_scale`：CFG 强度

`guidance_scale` 控制 classifier-free guidance（CFG）的强度。
**我们不使用 CFG，因此设置为 `1.0`。** 在当前训练实现中，设为 `1.0`
会直接使用条件预测，不执行正负条件的 CFG 混合；不要用 `0` 表示关闭 CFG。

**评估配置：** 独立评估也默认使用 `evaluator.guidance_scale=1.0`。评估有独立的采样路径和参数，
不受训练的 `optimization_level`、`churn_weight` 控制；详见
[TwinTilt 训练与评估说明](TWINTILT.md#training-and-evaluation)。

## 7. `multi_reward_geneval_clip30`：四项奖励组合

该配置对应旧库的 `--reward-preset multi_reward_geneval_clip30`，默认使用
GenEval 数据及完整 metadata。四个评分器分别输出原始分数 \(P,H,C,G\)：

| 分量 | 模型／评分 | 校准后的分数 | 默认训练权重 |
|---|---|---|---|
| PickScore | PickScore v1 | \(P\times(1/26)\) | 1.0 |
| HPSv2 | 本地 HPSv2.1 | \(H\) | 1.0 |
| CLIPScore | CLIP-L/14 原始 logit | \(C\times(1/30)\) | 1.0 |
| GenEval | partial score | \(G\) | 1.0 |

**训练公式：** 固定缩放后，各分量分别做组内 z-score，再加权平均。
各分量和合计均不裁剪。对于分量 \(r_j\)：

\[
z_j=\frac{r_j-\operatorname{mean}(r_j)}
{\operatorname{std}_{population}(r_j)+10^{-4}},\qquad
a=\frac{\sum_j w_jz_j}{\sum_j w_j}.
\]

**计算范围：** 均值和标准差在每个完整 group 内计算。合并后的 \(a\) 直接送入 group exact
projection，不再整体 z-score 或整体裁剪，也不会再次把 PickScore 除以 26。
名称中的 `clip30` 指 CLIPScore 的校准除数 30，不表示裁剪。
这里不提供 `rank`、`diffusionnft` 合并模式。

**明确修改：** 与改名前迁移版本相比，单 PickScore 和 multi-reward 都移除了
奖励裁剪；multi-reward 从加权求和改成加权平均。默认四个权重为 1，平均的分母
为 4。`eta=1.0` 和 `u_scale=0.1` 保持，不额外补偿该尺度变化。
`trainer.reward_clip_max` 和 multi-reward 的 `reward.clip_max` 已删除，显式传入会
报错。梯度裁剪 `trainer.max_grad_norm=1.0` 保持。

**运行命令：** 在 NexusAlign 目录中执行：

```bash
EXP=multi_reward_geneval_clip30 NPROC_PER_NODE=2 \
  bash scripts/reproduce/twintilt_train.sh

# 调整训练权重，例如只改变 GenEval 对优势的贡献。
EXP=multi_reward_geneval_clip30 NPROC_PER_NODE=2 \
  bash scripts/reproduce/twintilt_train.sh reward.weights.geneval=2.0

EXP=multi_reward_geneval_clip30 EXP_NAME=<训练实验目录名> \
  bash scripts/reproduce/twintilt_eval.sh evaluator.new_run=true
```

**训练权重：** `reward.weights` 下有 `pickscore`、`hpsv2`、`clipscore`、`geneval` 四项。
权重可以为零或负数，但每项及权重和都必须有限，且权重和不能为零。
权重为零的分量仍会评分和记录，只是不参与训练优势的合成。

**日志指标：** 日志区分训练优势与报告分数：

- **`reward_multi_reward_geneval_clip30`：**固定报告
  \(P/26+H+C/30+G\)，不受训练权重修改影响。
- **`reward_pickscore`、`reward_hpsv2`、`reward_clipscore`、`reward_geneval`：**
  各评分器的原始分数均值。这里的 PickScore 日志是原始值；单 PickScore 任务的
  同名指标则是 `/26` 后的值。
- **`reward_<分量>_calibrated`：**校准后的分数均值。
- **`reward_<分量>_group_zscore`：**非零权重分量的组内优势均值。
- **`reward_training_reward`：**最终训练加权平均优势的均值，不代表图像质量总分。

**评估奖励：** 评估报告固定总分、四项原始分数、校准分数及 GenEval 准确率，不执行组内训练
归一化。HPSv2、PickScore 和 CLIPScore 在当前进程中使用本地模型；GenEval
复用现有奖励服务。启动脚本只清理自己启动的服务。

## 8. 四类任务的统一评估入口

统一使用 `scripts/reproduce/twintilt_eval.sh`，通过 `EXP` 选择测试数据和奖励。
`EXP_NAME` 填训练实验在 `logs/` 下的目录名，包含训练时生成的时间戳。
以下命令均在 NexusAlign 目录中执行，请将示例目录名替换为实际目录名。

```bash
# PickScore test：报告原始 PickScore。
EXP=pickscore EXP_NAME='twintilt_pickscore-实际时间戳' NPROC_PER_NODE=1 \
  bash scripts/reproduce/twintilt_eval.sh

# GenEval test：报告 partial score、准确率及分项指标。
EXP=geneval EXP_NAME='twintilt_geneval-实际时间戳' NPROC_PER_NODE=1 \
  bash scripts/reproduce/twintilt_eval.sh

# OCR test：使用 PaddleOCR 评分。
EXP=ocr EXP_NAME='twintilt_ocr-实际时间戳' NPROC_PER_NODE=1 \
  bash scripts/reproduce/twintilt_eval.sh

# GenEval test：报告四项奖励、校准总分及 GenEval 指标。
EXP=multi_reward_geneval_clip30 \
  EXP_NAME='twintilt_multi_reward_geneval_clip30-实际时间戳' NPROC_PER_NODE=1 \
  bash scripts/reproduce/twintilt_eval.sh
```

**指定 checkpoint：** 默认评估该实验 `checkpoints/` 下的所有 checkpoint，按全局 step 排序。
如需只评估一个 checkpoint，传入完整路径：

```bash
EXP=pickscore EXP_NAME='twintilt_pickscore-实际时间戳' \
  bash scripts/reproduce/twintilt_eval.sh \
  "evaluator.ckpts=[$PWD/logs/twintilt_pickscore-实际时间戳/checkpoints/epoch_1-step_0-total_step_1]"
```

**少量样本：** 少量样本验证可以同时缩小采样参数，并新建评估目录：

```bash
EXP=pickscore EXP_NAME='twintilt_pickscore-实际时间戳' NPROC_PER_NODE=1 \
  bash scripts/reproduce/twintilt_eval.sh \
  "evaluator.ckpts=[$PWD/logs/twintilt_pickscore-实际时间戳/checkpoints/epoch_1-step_0-total_step_1]" \
  evaluator.new_run=true evaluator.num_prompts=2 evaluator.num_repeats=1 \
  evaluator.num_steps=3 evaluator.resolution=256
```

**正式默认值：** 同样的参数适用于另外三项任务，只需替换 `EXP`、实验目录名和 checkpoint 路径。
默认正式评估仍是 student、40 步原生 Flow Euler、512 分辨率、CFG=1、T5 长度 256，
不使用 churn。`evaluator.global_seed` 控制逐图种子，每个 repeat 使用
`global_seed + repeat_index`。

**启动检查：** 脚本会在启动服务前检查父项目 `.venv/bin/python`、实际实验目录和
`config_train.yaml`。Hydra 的 `log_dir`、`exp_info` 等覆盖参数也参与目录解析。
所有评估参数继续透传；可用 `--cfg job --resolve` 查看组合后的配置，该操作不启动
模型或奖励服务。

**缓存复用：** 默认复用采样签名匹配的图片和已完成指标；如果采样设置不同，程序会明确报错，
请传入 `evaluator.new_run=true`。小规模验证结果与正式评估应使用不同的评估目录。

**输出位置：** 输出保存在 `logs/<实验目录>/eval/reward-twintilt-<时间戳>/`：包含
`config_eval.yaml`、汇总 JSON、曲线，以及各 checkpoint 的 `images/*.png`、
`samples.json` 和 `metrics.json`。自定义 `log_dir` 时，输出跟随该实验目录。

**奖励服务：** GenEval、OCR 和 multi-reward 会检查或启动所需服务；已有服务会被复用，退出时只
清理本脚本启动的服务。使用自行管理的远端服务时，设置 `START_REWARD_SERVER=0`
并传入对应的奖励 URL。零奖励是有效评分结果，短验证通过不代表图像质量达标。

## 9. `solver`：训练采样求解器

可选 `euler` 和 `sa`，默认 `euler`。`sa` 使用旧库的 AWM 2/2 PEC SA-Solver，
要求 `num_steps >= 2`；`num_steps` 仍表示模型调用次数及每个成员的训练时间点数。
SA 内部的 predictor/corrector 阶数属于采样数值积分，不是已删除的一阶／二阶 guidance。

```bash
# 默认 Euler；不传 solver 时也是此配置。
EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/twintilt_train.sh \
  algorithm.solver=euler

# SA，共享基础初始噪声。
EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/twintilt_train.sh \
  algorithm.solver=sa

# SA，独立初始噪声；保留默认 churn=0.04。
EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/twintilt_train.sh \
  algorithm.solver=sa algorithm.optimization_level=distribution
```

**时间网格：** 两种 solver 均支持两种 `optimization_level`。SA 按旧实现调整时间网格，并匹配
原 scheduler 网格下 churn 的新噪声强度；训练记录实际 predictor 输入和调用时间。
`churn_weight=0` 不加入中间噪声，但 SA 仍保留旧实现的随机数抽取顺序。
过大的 churn 若无法映射到 SA 时间网格会明确报错。普通续训不能切换 solver。

**独立评估：** 此参数只控制训练 rollout。独立评估继续使用原生 Flow Euler、默认 40 步且无 churn。

## 10. 环境、训练与普通续训

从 NexusAlign 仓库目录运行。正式入口为 `--config-name twintilt`，算法注册名为
`algorithm.name: twintilt`。脚本使用父目录的 `.venv/bin/python`，显式加载本仓库
`src`，不安装依赖或下载模型。

**资源目录：** `TWINTILT_RESOURCE_ROOT` 默认指向父项目目录，可另行指定。其下需要
`data/` 和 `models/`，SD3.5 Medium 位于
`models/stabilityai/stable-diffusion-3.5-medium`。不接受旧方法名称或环境变量回退。

**四类训练：** 默认使用 Euler、trajectory、每轮 48 groups、每组 24 个成员。
以下命令会启动正式训练；短验证参数见验证报告。

```bash
EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/twintilt_train.sh
EXP=geneval NPROC_PER_NODE=2 bash scripts/reproduce/twintilt_train.sh
EXP=ocr NPROC_PER_NODE=2 bash scripts/reproduce/twintilt_train.sh
EXP=multi_reward_geneval_clip30 NPROC_PER_NODE=2 \
  bash scripts/reproduce/twintilt_train.sh
```

**普通续训：** 提供 checkpoint 目录，并保留原实验的任务、GPU 数和训练语义参数。
非默认实验需要再次传入原有覆盖参数；可以增加总 epochs 或调整日志和保存频率。
optimizer 参数从 checkpoint 恢复，轮内缓存和各 rank 的数据流一同恢复。

```bash
EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/twintilt_train.sh \
  trainer.resume_path=/absolute/path/to/checkpoint
```

**保存设置：** `trainer.save_ckpt_every_n_epochs=1` 和
`trainer.save_ckpt_every_n_steps=1` 独立生效，均可设为 `null` 关闭；同时触发只保存
一次。默认全部保留，仅保存 student、rollout EMA、optimizer、scaler、runtime 和
训练状态，不保存 evaluation EMA。旧 CoFiTi checkpoint 不自动转换或继续训练。

**验证范围：** 当前版本的奖励公式、上游适配及短运行结果见
[TWINTILT_VALIDATION.md](TWINTILT_VALIDATION.md)。历史记录与本次验证分开列出。
