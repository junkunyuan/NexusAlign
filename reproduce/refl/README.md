# ReFL

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation* ([NeurIPS 2023](https://arxiv.org/abs/2304.05977))

**Summary:** Sample without gradients to a random late denoising step, and predict the clean image from it in one step. Then backpropagate a differentiable reward of that prediction into the generator.

**Implementation:** Because [worldbench/DiffusionOPSD](https://github.com/worldbench/DiffusionOPSD) trains ReFL on the newer SD3.5-M and reports results that we can compare with, we follow it instead of the official code [zai-org/ImageReward](https://github.com/zai-org/ImageReward), except for:

- **Optimizer:** We use learning rate 3e-4 and weight decay 1e-4, as DiffusionNFT does. The ReFL config of DiffusionOPSD uses 1e-5 and 1e-2, but its Table 1 and Fig. 7 match 3e-4. With 1e-5, our DrawBench PickScore reaches only 22.02.
- **Training prompts:** We use the 25,432 PickScore prompts of Flow-GRPO, as our DiffusionNFT and Flow-GRPO runs do, not the 25,415 Pick-a-Pic prompts of DiffusionOPSD. Thus the 512 prompts that standardize the reward also differ.
- **Reward resize:** The differentiable PickScore resizes the images with antialiased bicubic, as the PickScore processor does, not with bilinear. This changes the gradient direction, but not the training speed.

## Results

**Table 1 of [DiffusionOPSD](https://arxiv.org/abs/2608.24646):** SD3.5-M trained on PickScore.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (paper) | - | - | - | 20.51 | 0.237 | 0.204 | 5.13 | -0.58 | - | 3.30 | -6.47 | - |
| SD3.5-M w/o CFG (ours) | 0.24 | 0.13 | 19.82 | 20.54 | 0.238 | 0.207 | 5.17 | -0.58 | 2.03 | 3.37 | -1.01 | [run](pickscore-20261103_073350) |
| PickScore (paper) | - | - | - | **23.92** | - | - | - | - | - | - | - | - |
| PickScore (ours) | 0.27 | 0.38 | **24.07** | **23.93** | 0.259 | 0.292 | 6.16 | 1.02 | 3.20 | 3.73 | 10.96 | [run](pickscore-20261103_073350) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench.
- **Bold:** in-domain results.
- **Paper:** DiffusionOPSD trains one ReFL checkpoint per reward, so only its PickScore comes from this setting.

## Command

### Training

**Train SD3.5-M** on PickScore:
```bash
# EXP is pickscore, because ReFL needs a differentiable reward.
EXP=pickscore bash scripts/reproduce/refl_train.sh
```

### Evaluation

```bash
# GenEval, PickScore, and OCR, each on its own test prompts.
for TASK in geneval pickscore ocr; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/refl_eval.sh
done

# The other metrics on DrawBench, with two reward models per evaluation to prevent OOM.
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/refl_eval.sh
done
```
