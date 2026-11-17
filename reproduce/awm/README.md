# AWM

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models* ([ICML 2026](https://arxiv.org/abs/2509.25050))

**Summary:** Roll out a group of images per prompt, and train the flow matching loss of pretraining on their clean latents. Each image's loss gets the weight of its group-relative advantage, so the policy gradient needs no reverse-process likelihood. A velocity KL to an EMA of the policy keeps the updates small.

**Implementation:** We follow the official code [scxue/advantage_weighted_matching](https://github.com/scxue/advantage_weighted_matching), except for:

- **Paper and code:** Where the paper and the code differ, we follow the code. Its loss is a generalized Huber loss (power 0.5) with the weight t, and it adds a small KL to the base model with CFG 4.5. The images of one prompt share their training timesteps and noise.
- **Training length:** The code trains until it is stopped, and the paper gives only GPU hours. We train 750 (GenEval), 130 (OCR), 1800 (PickScore), and 500 (multi-reward) epochs, with one update per epoch.
- **Multi-reward:** We train on GenEval, PickScore, ClipScore, and HPSv2.1 on the GenEval prompts, with the GenEval recipe, as our DiffusionNFT and DiffusionART multi-reward runs do, so that the runs compare. The code has no multi-reward task. The training reward is the mean of the group z-scores of the four calibrated scores, as in DiffusionART. Flow-GRPO, which the code follows, normalizes their summed score instead.
- **GPUs:** We train on 128 GPUs instead of 32. 128 ranks cannot split the official batches into whole prompt groups, so we use 1920 images per update instead of 1728 (GenEval, OCR), and 768 instead of 576 (PickScore).
- **Evaluation:** We evaluate the EMA weights. The official code logs both the EMA and the trained weights.
- **Infrastructure:** `torch.compile`, fp16 frozen weights, FSDP, fp32 solver steps on fp16 latents, and one forward for both CFG branches speed up training and change only the rounding. The code for logging, checkpointing, and the reward servers is our own.

## Results

**Tables 1 and 2 of the paper:** SD3.5-M trained on GenEval, OCR, and PickScore.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 0.236 | 0.13 | 19.82 | 20.54 | 0.239 | 0.206 | 5.17 | -0.58 | 2.03 | 3.36 | -1.06 | [run](geneval-20261114_170932) |
| GenEval (paper) | **0.985** | - | - | - | - | - | - | - | - | - | - | - |
| GenEval (ours) | **0.969** | 0.37 | 21.41 | 22.37 | 0.284 | 0.280 | 5.44 | 0.96 | 3.19 | 4.28 | 10.92 | [run](geneval-20261114_170932) |
| OCR (paper) | - | **0.96** | - | - | - | - | - | - | - | - | - | - |
| OCR (ours) | 0.594 | **0.94** | 21.39 | 22.13 | 0.277 | 0.266 | 5.32 | 0.58 | 2.92 | 4.12 | 7.88 | [run](ocr-20261114_164607) |
| PickScore (paper) | - | - | **24.03** | - | - | - | - | - | - | - | - | - |
| PickScore (ours) | 0.508 | 0.65 | **23.87** | 24.10 | 0.272 | 0.315 | 6.24 | 1.37 | 3.42 | 4.25 | 12.77 | [run](pickscore-20261114_191649) |

**Multi-reward of DiffusionNFT and DiffusionART:** SD3.5-M trained on GenEval, PickScore, ClipScore, and HPSv2.1, on the GenEval prompts.

| Model | Multi† | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 2.047 | 0.24 | 0.13 | 19.82 | 20.54 | 0.239 | 0.206 | 5.17 | -0.58 | 2.03 | 3.36 | -1.06 | [run](multi_reward-20261116_230650) |
| Multi-reward (ours) | **3.135** | 0.84 | 0.38 | 22.01 | 23.12 | 0.282 | 0.317 | 5.71 | 1.26 | 3.20 | 4.45 | 13.25 | [run](multi_reward-20261116_230650) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench. Multi† is GenEval + PickScore / 26 + ClipScore logit / 30 + HPSv2.1 on the GenEval test prompts, as DiffusionART reports it.
- **Checkpoints:** the multi-reward row uses the final epoch 500, and its base row epoch 0. Epoch 400 is higher on 11 of the 12 metrics, all except DeQA (Multi† 3.174, GenEval† 0.87).
- **Bold:** in-domain results.
- **Paper:** Tables 1 and 2 give only the in-domain metric. The OCR and PickScore rows come from its longer runs.

## Command

### Training

Train SD3.5-M on one reward with its specific training data:
```bash
# EXP is geneval, ocr, pickscore, or multi_reward (on the GenEval prompts).
EXP=geneval bash scripts/reproduce/awm_train.sh
```

### Evaluation

```bash
# GenEval, PickScore, and OCR, each on its own test prompts.
for TASK in geneval pickscore ocr; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/awm_eval.sh
done

# The other metrics on DrawBench, with two reward models per evaluation to prevent OOM.
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/awm_eval.sh
done

# The multi-reward and its four components, on the GenEval test prompts.
EXP_NAME=<log_dir_name> DATA=geneval REWARDS=multi_reward \
  bash scripts/reproduce/awm_eval.sh
```
