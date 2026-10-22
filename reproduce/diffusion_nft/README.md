# DiffusionNFT

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** [*DiffusionNFT: Online Diffusion Reinforcement with Forward Process* (ICLR 2026)](https://arxiv.org/abs/2509.16117)

**Official code:** [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT).

**Summary:** Optimize a diffusion model on the forward process with a flow matching objective, contrasting positive and negative generations to define an implicit policy improvement direction, without likelihoods or CFG.

## Results

Table 2 of paper.

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

- † means scoring on their own test prompts, and the other on DrawBench.
- In-domain results are highlighted in bold.

## Command

### Training

Train SD3.5-M model.
```bash
# EXP is geneval, ocr, or pickscore.
EXP=geneval bash scripts/reproduce/diffusion_nft_train.sh
```

### Evaluation

For a GenEval or PickScore training run:
```bash
# GenEval, PickScore, and OCR, each on its own test prompts.
for TASK in geneval pickscore ocr; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/diffusion_nft_eval.sh
done

# The other metrics on DrawBench, two reward models per eval to avoid OOM.
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/diffusion_nft_eval.sh
done
```

For an OCR training run (with CFG for every metric but OCR, as in the paper):
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
