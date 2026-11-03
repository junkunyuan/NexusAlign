# DiffusionNFT

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *DiffusionNFT: Online Diffusion Reinforcement with Forward Process* ([ICLR 2026](https://arxiv.org/abs/2509.16117))

**Summary:** Run online RL on the forward process. One flow-matching loss learns the positive and negative policies that the reward defines on the rollouts.

**Implementation:** We follow the official code [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT), with the GenEval reward from [yifan123/reward-server](https://github.com/yifan123/reward-server), except for:

- **Paper and code:** Where the paper and the code differ, we follow the code. It trains 9 of the 10 timesteps and uses a KL weight of 1e-4. For OCR, its old policy stays on-policy for 75 steps, then decays at a rate of 0.0075.
- **T5 length in evaluation:** The evaluation encodes the prompts with 128 T5 tokens, as the training and `eval_fn` do. `evaluation.py` of the official code uses the diffusers default of 256.
- **GPUs:** We train on 64 GPUs instead of 8, with the same 1152 images per update (576 for OCR).
- **Infrastructure:** `torch.compile`, fp16 frozen weights, FSDP, and the in-place old-policy swap speed up training and change only the rounding. The code for logging, checkpointing, and the reward servers is our own.

## Results

**Table 2 of the paper:** SD3.5-M trained on GenEval, OCR, and PickScore.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (paper) | 0.24 | 0.12 | - | 20.51 | 0.237 | 0.204 | 5.13 | -0.58 | 2.02 | - | - | - |
| SD3.5-M w/o CFG (ours) | 0.24 | 0.13 | 19.82 | 20.54 | 0.238 | 0.207 | 5.17 | -0.58 | 2.03 | 3.37 | -1.01 | [run](geneval-20261020_183733) |
| GenEval (paper) | **0.98** | 0.36 | - | 21.92 | 0.271 | 0.251 | 5.33 | 0.68 | 2.91 | - | - | - |
| GenEval (ours) | **0.95** | 0.48 | 21.01 | 22.23 | 0.281 | 0.263 | 5.31 | 0.82 | 3.11 | 4.20 | 9.46 | [run](geneval-20261020_183733) |
| OCR (paper) | 0.54 | **0.97** | - | 21.63 | 0.281 | 0.246 | 5.19 | 0.37 | 2.81 | - | - | - |
| OCR (ours) | 0.64 | **0.97** | 21.72 | 22.34 | 0.286 | 0.280 | 5.38 | 0.86 | 3.07 | 4.07 | 9.49 | [run](ocr-20261020_024519) |
| PickScore (paper) | 0.53 | 0.64 | - | **24.03** | 0.270 | 0.315 | 6.17 | 1.29 | 3.40 | - | - | - |
| PickScore (ours) | 0.57 | 0.73 | **23.78** | 24.04 | 0.274 | 0.317 | 6.20 | 1.35 | 3.43 | 4.26 | 12.65 | [run](pickscore-20261020_040000) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench.
- **Bold:** in-domain results.

## Command

### Training

Train SD3.5-M on one reward with its specific training data:
```bash
# EXP is geneval, ocr, or pickscore.
EXP=geneval bash scripts/reproduce/diffusion_nft_train.sh
```

### Evaluation

**Evaluate GenEval or PickScore runs:** with CFG off for every metric.
```bash
# GenEval, PickScore, and OCR, each on its own test prompts.
for TASK in geneval pickscore ocr; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/diffusion_nft_eval.sh
done

# The other metrics on DrawBench, with two reward models per evaluation to prevent OOM.
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/diffusion_nft_eval.sh
done
```

**Evaluate OCR runs:** with CFG for every metric except OCR, as in the paper.
```bash
# GenEval and PickScore, with CFG on.
for TASK in geneval pickscore; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/diffusion_nft_eval.sh evaluator.guidance_scale=4.5
done

# OCR, with CFG off.
EXP_NAME=<log_dir_name> DATA=ocr REWARDS=ocr \
  bash scripts/reproduce/diffusion_nft_eval.sh

# The other metrics on DrawBench, with CFG on.
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/diffusion_nft_eval.sh evaluator.guidance_scale=4.5
done
```
