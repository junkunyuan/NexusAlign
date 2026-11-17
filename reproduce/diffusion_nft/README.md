# DiffusionNFT

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *DiffusionNFT: Online Diffusion Reinforcement with Forward Process* ([ICLR 2026](https://arxiv.org/abs/2509.16117))

**Summary:** Run online RL on the forward process. One flow-matching loss learns the positive and negative policies that the reward defines on the rollouts.

**Implementation:** We follow the official code [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT), with the GenEval reward from [yifan123/reward-server](https://github.com/yifan123/reward-server), except for:

- **Paper and code:** Where the paper and the code differ, we follow the code. It trains 9 of the 10 timesteps and uses a KL weight of 1e-4. For OCR, its old policy stays on-policy for 75 steps, then decays at a rate of 0.0075.
- **Multi-reward:** We train on GenEval, PickScore, ClipScore, and HPSv2.1 on the GenEval prompts for 500 steps, with the GenEval recipe, as our DiffusionART and TwinTilt multi-reward runs do, so that the three compare. The official `sd3_multi_reward` trains on PickScore, HPSv2.1, and ClipScore on the PickScore prompts. The training reward is the mean of the group z-scores of the four calibrated scores, as in DiffusionART and TwinTilt. The official code instead normalizes their sum in its NFT recipes, which gave a higher and more stable run here (see the notes).
- **Qwen judge:** The judge task trains on the holistic score of a 27.8B Qwen judge on the PickScore prompts, with the PickScore recipe, for 100 steps. One judge per node serves every rank of that node, sharded over its GPUs. The judge is not in the official code.
- **T5 length in evaluation:** The evaluation encodes the prompts with 128 T5 tokens, as the training and `eval_fn` do. `evaluation.py` of the official code uses the diffusers default of 256.
- **GPUs:** We train on 64 GPUs (128 for the multi-reward) instead of 8, with the same 1152 images per update (576 for OCR).
- **MiniMax-H3-Image:** The HPSv3 task trains the H3-Image DiT on its native six-step PDD sampler at 768 pixels, with 128 groups of 24 images for 100 steps, and with no KL term. FSDP shards the 33B model, its EMA, and the text encoder over the GPUs of each node, and the rollout and the training forward passes take four members at a time. A packed sequence holds one prompt, so one rank rolls out one whole group, and `groups_per_epoch` must be a multiple of the GPU count.
- **Infrastructure:** `torch.compile`, fp16 frozen weights, FSDP, and the in-place old-policy swap speed up training and change only the rounding. The code for logging, checkpointing, and the reward servers is our own.

## Results

**Table 2 of the paper:** SD3.5-M trained on GenEval, OCR, and PickScore.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (paper) | 0.24 | 0.12 | - | 20.51 | 0.237 | 0.204 | 5.13 | -0.58 | 2.02 | - | - | - |
| SD3.5-M w/o CFG (ours) | 0.24 | 0.13 | 19.82 | 20.54 | 0.238 | 0.207 | 5.17 | -0.58 | 2.03 | 3.37 | -1.01 | [run](geneval-20261020_183733) |
| GenEval (paper) | **0.98** | 0.36 | - | 21.92 | 0.271 | 0.251 | 5.33 | 0.68 | 2.91 | - | - | - |
| GenEval (ours) | **0.95** | 0.40 | 20.71 | 21.93 | 0.276 | 0.242 | 5.15 | 0.65 | 3.00 | 3.95 | 6.57 | [run](geneval-20261020_183733) |
| OCR (paper) | 0.54 | **0.97** | - | 21.63 | 0.281 | 0.246 | 5.19 | 0.37 | 2.81 | - | - | - |
| OCR (ours) | 0.56 | **0.97** | 21.01 | 21.78 | 0.278 | 0.260 | 5.30 | 0.45 | 2.90 | 3.95 | 7.02 | [run](ocr-20261020_024519) |
| PickScore (paper) | 0.53 | 0.64 | - | **24.03** | 0.270 | 0.315 | 6.17 | 1.29 | 3.40 | - | - | - |
| PickScore (ours) | 0.56 | 0.73 | **23.78** | 24.04 | 0.272 | 0.317 | 6.20 | 1.34 | 3.40 | 4.26 | 12.55 | [run](pickscore-20261020_040000) |

**Multi-reward of DiffusionART and TwinTilt:** SD3.5-M trained on GenEval, PickScore, ClipScore, and HPSv2.1, on the GenEval prompts.

| Model | Multi† | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 2.046 | 0.24 | 0.13 | 19.82 | 20.54 | 0.238 | 0.207 | 5.17 | -0.58 | 2.03 | 3.37 | -1.01 | [run](multi_reward-20261116_184916) |
| Multi-reward (ours) | **2.580** | 0.47 | 0.18 | 20.34 | 21.71 | 0.267 | 0.259 | 5.71 | 0.86 | 2.64 | 3.43 | 6.93 | [run](multi_reward-20261116_184916) |

**HPSv3 on MiniMax-H3-Image:** its native six-step sampler at 768 pixels without CFG, with the student weights.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniMax-H3-Image (ours) | 0.61 | 0.74 | 20.86 | 21.96 | 0.277 | 0.267 | 5.04 | 0.75 | 3.15 | 4.22 | 9.05 | [run](h3_hpsv3-20261116_082204) |
| HPSv3 (ours) | 0.71 | 0.71 | 21.40 | 22.44 | 0.257 | 0.286 | 5.86 | 1.08 | 3.22 | 4.13 | **13.03** | [run](h3_hpsv3-20261116_082204) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench. Multi† is GenEval + PickScore / 26 + ClipScore logit / 30 + HPSv2.1 on the GenEval test prompts, as DiffusionART reports it.
- **Checkpoints:** the multi-reward row uses the final epoch 500, and its base row epoch 0. This run falls after epoch 400, so epoch 150 is higher on every metric except Aesthetic (Multi† 3.066, GenEval† 0.79, HPSv3 13.44). The HPSv3 row uses its final step 100, where no metric has reached a plateau yet.
- **Bold:** in-domain results.

## Command

### Training

Train SD3.5-M on one reward with its specific training data:
```bash
# EXP is geneval, ocr, pickscore, multi_reward (on the GenEval prompts), qwen_judge, or
# h3_hpsv3 (MiniMax-H3-Image on 128 GPUs).
EXP=geneval bash scripts/reproduce/diffusion_nft_train.sh
```

### Evaluation

**Evaluate GenEval, PickScore, or multi-reward runs:** with CFG off for every metric.
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

# The multi-reward and its four components, on the GenEval test prompts.
EXP_NAME=<log_dir_name> DATA=geneval REWARDS=multi_reward \
  bash scripts/reproduce/diffusion_nft_eval.sh

# The judge, on the PickScore test prompts.
EXP_NAME=<log_dir_name> DATA=pickscore REWARDS=qwen_judge \
  bash scripts/reproduce/diffusion_nft_eval.sh
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
