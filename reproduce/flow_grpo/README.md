# Flow-GRPO

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** [*Flow-GRPO: Training Flow Matching Models via Online RL* (NeurIPS 2025)](https://arxiv.org/abs/2505.05470)

**Official code:** [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo).

**Summary:** Turn the ODE sampler of a flow matching model into an equivalent SDE, so that GRPO can explore and optimize it with online rewards.

## Results

Table 2 of paper.

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

- † means scoring on their own test prompts, and the other on DrawBench.
- In-domain results are highlighted in bold.

## Command

### Training

Train SD3.5-M model.
```bash
# EXP is geneval, ocr, or pickscore.
EXP=geneval bash scripts/reproduce/flow_grpo_train.sh
```

### Evaluation

```bash
# GenEval, OCR, and PickScore, each on its own test prompts.
for TASK in geneval ocr pickscore; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/flow_grpo_eval.sh
done

# The other metrics on DrawBench, two reward models per eval to avoid OOM.
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/flow_grpo_eval.sh
done
```
