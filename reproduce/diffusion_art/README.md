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
- **GPUs:** We train on 64 GPUs instead of 8, with the same 1152 images per rollout (1024 for OCR).
- **Infrastructure:** `torch.compile`, fp16 frozen weights, and FSDP speed up training and change only the rounding. The copies of a group spread over the ranks, so the guidance sums over all ranks. The code for logging, checkpointing, and the reward servers is our own.

## Results

**Fig. 5 of the paper:** SD3.5-M trained on GenEval, OCR, and PickScore.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 0.24 | 0.13 | 19.82 | 20.54 | 0.239 | 0.206 | 5.17 | -0.58 | 2.02 | 3.36 | -1.06 | [run](geneval-20261104_032911) |
| GenEval (ours) | **0.88** | 0.46 | 19.77 | 21.18 | 0.279 | 0.229 | 4.75 | 0.33 | 2.51 | 2.84 | 1.77 | [run](geneval-20261104_032911) |
| OCR (ours) | 0.02 | **0.98** | 17.27 | 18.85 | 0.192 | 0.142 | 3.74 | -1.73 | 1.18 | 1.54 | -8.80 | [run](ocr-20261104_004111) |
| PickScore (ours) | 0.44 | 0.54 | **23.38** | 23.57 | 0.275 | 0.312 | 6.35 | 1.29 | 3.16 | 3.45 | 9.81 | [run](pickscore-20261104_021028) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench.
- **Bold:** in-domain results.
- **Paper:** Fig. 5 gives only the training rewards. Our training rewards match these curves to our observation.

## Command

### Training

Train SD3.5-M on one reward with its specific training data:
```bash
# EXP is geneval, ocr, or pickscore.
EXP=geneval bash scripts/reproduce/diffusion_art_train.sh
```

### Evaluation

```bash
# GenEval, PickScore, and OCR, each on its own test prompts.
for TASK in geneval pickscore ocr; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/diffusion_art_eval.sh
done

# The other metrics on DrawBench, with two reward models per evaluation to prevent OOM.
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/diffusion_art_eval.sh
done
```
