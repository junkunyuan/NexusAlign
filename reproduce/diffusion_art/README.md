# DiffusionART

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models* ([arXiv 2026](https://arxiv.org/abs/2610.12572))

**Summary:** Roll out a group of images per prompt with an ODE sampler. The covariance of the rewards and the clean latents of each group gives one guidance for the group. Then regress the velocities at the rollout states to the old velocities plus this guidance, in two disjoint updates with a trust-region mask.

**Implementation:** We follow the official code [vanzll/DiffusionART](https://github.com/vanzll/DiffusionART), except for:

- **Paper and code:** Where the paper and the code differ, we follow the code. Each of its two updates gets one half of every prompt group. It trains 1000 updates.
- **Prompt sampling:** Each epoch draws its prompts at random, as our DiffusionNFT and Flow-GRPO runs do. The official code goes through the shuffled prompts without replacement.
- **GPUs:** We train on 64 or 128 GPUs instead of 8, with the same 1152 images per rollout (1024 for OCR).
- **Multi-reward:** HPSv2.1 resizes the pixels as float tensors, as DiffusionNFT does, not as PIL images with the transform of the HPSv2 package.
- **Qwen judge:** The judge task trains SD3.5-M on the PickScore prompts with the rollout of our PickScore run, and the judge scores the whole image. The TwinTilt code has this judge task, and we train SD3.5-M for 50 epochs, which give the 100 updates of our other SD3.5-M runs, so that the judge compares with them.
- **MiniMax-H3-Image:** The HPSv3 task trains the H3-Image DiT on its native six-step PDD sampler at 768 pixels, with 128 groups of 24 images for 50 epochs, which give 100 updates. FSDP shards the 33B model and the text encoder over the GPUs of each node, and the rollout and the training forward passes take four members at a time. A packed sequence holds one prompt, so one rank rolls out one whole group, each of the two updates trains one half of it, and `groups_per_epoch` must be a multiple of the GPU count.
- **Infrastructure:** `torch.compile`, fp16 frozen weights, and FSDP speed up training and change only the rounding. The copies of a group spread over the ranks, so the guidance sums over all ranks. On 128 GPUs, each rank trains 4 or 5 images of each update, with a loss scale that keeps the mean over all images of the update. The code for logging, checkpointing, and the reward servers is our own.

## Results

**Fig. 5 of the paper:** SD3.5-M trained on GenEval, OCR, and PickScore.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 0.24 | 0.13 | 19.82 | 20.54 | 0.239 | 0.206 | 5.17 | -0.58 | 2.02 | 3.36 | -1.06 | [run](geneval-20261104_032911) |
| GenEval (ours) | **0.88** | 0.46 | 19.77 | 21.18 | 0.279 | 0.229 | 4.75 | 0.33 | 2.51 | 2.84 | 1.77 | [run](geneval-20261104_032911) |
| OCR (ours) | 0.02 | **0.98** | 17.27 | 18.85 | 0.192 | 0.142 | 3.74 | -1.73 | 1.18 | 1.54 | -8.80 | [run](ocr-20261104_004111) |
| PickScore (ours) | 0.44 | 0.54 | **23.38** | 23.57 | 0.275 | 0.312 | 6.35 | 1.29 | 3.16 | 3.45 | 9.81 | [run](pickscore-20261104_021028) |

**Multi-reward of the official code:** SD3.5-M trained on GenEval, PickScore, ClipScore, and HPSv2.1, on the GenEval prompts.

| Model | Multi† | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 2.046 | 0.24 | 0.13 | 19.82 | 20.54 | 0.239 | 0.206 | 5.17 | -0.58 | 2.02 | 3.36 | -1.06 | [run](multi_reward-20261116_025704) |
| Multi-reward (ours) | **2.887** | 0.64 | 0.40 | 19.88 | 21.73 | 0.276 | 0.286 | 4.97 | 0.83 | 2.56 | 2.00 | 5.55 | [run](multi_reward-20261116_025704) |

**Qwen judge:** SD3.5-M trained on the holistic score of a vision-language judge, evaluated with CFG off.

| Model | Qwen | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 0.546 | 0.24 | 0.13 | 19.82 | 20.54 | 0.239 | 0.206 | 5.17 | -0.58 | 2.02 | 3.36 | -1.06 | [run](qwen_judge-20261117_071715) |
| Qwen judge (ours) | **0.778** | 0.60 | 0.34 | 21.57 | 22.38 | 0.276 | 0.283 | 5.71 | 0.98 | 3.15 | 3.82 | 8.98 | [run](qwen_judge-20261117_071715) |

**HPSv3 on MiniMax-H3-Image:** its native six-step sampler at 768 pixels without CFG, with the student weights.

| Model | #Iter | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniMax-H3-Image (ours) | - | 0.60 | 0.74 | 20.86 | 21.96 | 0.277 | 0.267 | 5.04 | 0.75 | 3.15 | 4.22 | 9.05 | [run](h3_hpsv3-20261116_161746) |
| HPSv3 (ours) | 100 | 0.59 | 0.64 | 20.99 | 22.39 | 0.247 | 0.309 | 6.17 | 0.93 | 3.15 | 4.27 | **13.77** | [run](h3_hpsv3-20261116_161746) |
| HPSv3 (ours) | 200 | 0.69 | 0.70 | 20.56 | 22.23 | 0.263 | 0.304 | 5.93 | 1.02 | 3.16 | 3.96 | **13.93** | [run](h3_hpsv3-20261117_012220) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench. Multi† is GenEval + PickScore / 26 + ClipScore logit / 30 + HPSv2.1 on the GenEval test prompts, as the official code reports it.
- **Checkpoints:** each row uses the final epoch 500, and the base rows use epoch 0. Each HPSv3 row uses its own final step. Step 40 is the best step of both HPSv3 runs, which agree there (PickScore† 21.77 and 21.80, HPSv3 13.65 and 13.66), because they over-optimize the reward after step 50. The 200-step run loses OCR† down to 0.47 at step 160 while its HPSv3 stays near 13.9, so the extra steps give nothing. The Qwen judge row uses its final step 100, and step 80 is better on 8 of the 12 metrics (Qwen 0.780, PickScore† 21.74, HPSv3 10.55).
- **Bold:** in-domain results.
- **Paper:** Fig. 5 gives only the training rewards of the single tasks, and the official code gives no multi-reward results. Our training rewards match these curves to our observation.

## Command

### Training

Train SD3.5-M on one reward with its specific training data, on 8, 16, 32, 64, or 128 GPUs:
```bash
# EXP is geneval, ocr, pickscore, multi_reward (on the GenEval prompts), qwen_judge,
# or h3_hpsv3 (MiniMax-H3-Image on 128 GPUs).
EXP=geneval bash scripts/reproduce/diffusion_art_train.sh
```

### Evaluation

```bash
# GenEval, PickScore, and OCR, each on its own test prompts.
for TASK in geneval pickscore ocr; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/diffusion_art_eval.sh
done

# The Qwen judge, on the PickScore test prompts.
EXP_NAME=<log_dir_name> DATA=pickscore REWARDS=qwen_judge \
  bash scripts/reproduce/diffusion_art_eval.sh

# The other metrics on DrawBench, with two reward models per evaluation to prevent OOM.
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/diffusion_art_eval.sh
done

# The multi-reward and its four components, on the GenEval test prompts.
EXP_NAME=<log_dir_name> DATA=geneval REWARDS=multi_reward \
  bash scripts/reproduce/diffusion_art_eval.sh
```
