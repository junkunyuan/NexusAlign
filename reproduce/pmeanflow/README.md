# pMeanFlow

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *One-step Latent-free Image Generation with Pixel Mean Flows* ([arXiv 2026](https://arxiv.org/abs/2601.22158))

**Summary:** Train iMeanFlow on pixels, not on VAE latents. The network predicts a denoised image x, and x gives the mean velocity u = (z_t - x) / t for the v-loss. Perceptual losses on x and the Muon optimizer improve the one-step images.

**Implementation:** We follow the official JAX code [Lyy-iiis/pMF](https://github.com/Lyy-iiis/pMF), and its [`hsdp` branch](https://github.com/Lyy-iiis/pMF/tree/hsdp) for the H models, except for:

- **Paper and code:** Where the paper and the configs of the code differ, we follow the configs. For example, the H models train 320 epochs, not 360, and only L/16 uses the CFG exponent β = 2.
- **Perceptual threshold of the H models:** Their configs set `convnext_max_t`, which the code does not read. We use t < 0.6 for both perceptual losses, as Appendix A.1 of the paper says.
- **One EMA:** Each run keeps one EDM EMA, with the half-life that the official `eval_config.yml` picks for the model, not the three half-lives of the official runs.
- **Infrastructure:** FSDP speeds up training. A uint8 cache of the center crops gives the same pixels as the official input pipeline. The logging and checkpointing code is our own.

## Results

**Table 6 of the paper:** ImageNet 256x256, 1-NFE.

| Model | FID | IS | Run |
|---|---|---|---|
| B/16 (paper) | 3.12 | 254.6 | - |
| B/16 (ours) | 3.43 | 254.0 | [run](b16-20261106_193046) |

## Command

### Training

Train pMeanFlowDiT on ImageNet with batch size 1024 (the paper setting):
```bash
# EXP is b16, l16, or h16 (256x256), or b32, l32, or h32 (512x512).
EXP=b16 bash scripts/reproduce/pmeanflow_train.sh
```

### Evaluation

Evaluate the FID and IS on 50k samples, with the official CFG setting of the model:
```bash
# B/16; pmeanflow_eval.sh lists the settings of the other models.
EXP_NAME=<log_dir_name> CFG_SCALE=8.5 CFG_MIN_T=0.1 CFG_MAX_T=0.7 \
  bash scripts/reproduce/pmeanflow_eval.sh

# The /32 models generate 512x512 images.
EXP_NAME=<log_dir_name> RESOLUTION=512 CFG_SCALE=6.5 CFG_MIN_T=0.1 CFG_MAX_T=0.7 \
  bash scripts/reproduce/pmeanflow_eval.sh
```
