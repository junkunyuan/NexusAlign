# Flow-GRPO

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *Flow-GRPO: Training Flow Matching Models via Online RL* ([NeurIPS 2025](https://arxiv.org/abs/2505.05470))

**Summary:** Turn the ODE sampler of a flow matching model into an equivalent SDE. GRPO can then explore and optimize it with online rewards.

**Implementation:** We follow the official code [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo), with the GenEval reward from [yifan123/reward-server](https://github.com/yifan123/reward-server), except for:

- **GPUs and EMA:** We train on 64 GPUs instead of the official 32, with the same 1152 images per epoch. `train_sd3.py` steps EMA after every batch. So in every 8 updates, EMA steps once here and twice in the official setting. Its decay per 8 updates is thus 0.9 here and 0.81 there.
- **GenEval metric:** GenEval† is reward-server's partial score (`geneval`). DiffusionNFT's `evaluation.py` also reports it. The paper's Overall is the mean task accuracy, which our evaluation summaries also record (`<task>_accuracy`).
- **Infrastructure:** `torch.compile`, FlashAttention-3, fp16 frozen weights, and FSDP speed up training and change only the rounding. The code for logging, checkpointing, and the OCR server is our own.

## Results

**Table 2 of the paper:** SD3.5-M trained separately on the rewards of GenEval, OCR, and PickScore.

| Model | GenEval† | OCR† | PickScore† | Aesthetic | DeQA | ImgRwd | PickScore | UniRwd | ClipScore | HPSv2.1 | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M (paper) | 0.63 | 0.59 | 21.72 | 5.39 | 4.07 | 0.87 | 22.34 | 3.33 | - | - | - | - |
| SD3.5-M (ours) | 0.64 | 0.56 | 21.71 | 5.39 | 4.06 | 0.88 | 22.37 | 3.08 | 0.285 | 0.281 | 9.52 | [run](geneval-20261021_063947) |
| GenEval (paper) | **0.95** | - | - | 5.25 | 4.01 | 1.03 | 22.37 | 3.51 | - | - | - | - |
| GenEval (ours) | **0.94** | 0.68 | 21.89 | 5.41 | 4.14 | 1.10 | 22.65 | 3.25 | 0.293 | 0.284 | 10.51 | [run](geneval-20261021_063947) |
| OCR (paper) | - | **0.92** | - | 5.32 | 4.06 | 0.95 | 22.44 | 3.42 | - | - | - | - |
| OCR (ours) | 0.66 | **0.91** | 21.77 | 5.37 | 4.07 | 0.99 | 22.46 | 3.16 | 0.289 | 0.282 | 9.67 | [run](ocr-20261020_234557) |
| PickScore (paper) | - | - | **23.31** | 5.92 | 4.22 | 1.28 | 23.53 | 3.66 | - | - | - | - |
| PickScore (ours) | 0.64 | 0.66 | **23.25** | 5.88 | 4.21 | 1.27 | 23.46 | 3.38 | 0.284 | 0.315 | 12.60 | [run](pickscore-20261021_071622) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench.
- **Bold:** in-domain results.

## Command

### Training

Train SD3.5-M on one reward with its specific training data:
```bash
# EXP is geneval, ocr, or pickscore.
EXP=geneval bash scripts/reproduce/flow_grpo_train.sh
```

### Evaluation

Evaluate GenEval, OCR, and PickScore, each on its own test dataset.
```bash
for TASK in geneval ocr pickscore; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/flow_grpo_eval.sh
done
```

Evaluate the other metrics on DrawBench, with two reward models per evaluation to prevent OOM.
```bash
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/flow_grpo_eval.sh
done
```
