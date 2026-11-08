# PixelDiT2

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *PixelDiT2: Representation-Grounded Pixel Diffusion Transformers* ([NeurIPS 2026](https://arxiv.org/abs/2609.24919))

**Summary:** Ground a pixel diffusion transformer on a frozen vision encoder. At each denoising step, DINOv3 encodes the noised image, and a timestep-conditioned DiT block maps its features to one grounding token per patch. A spatial adaLN injects these tokens into every block, so the transformer can focus on the pixels. The backbone is a JiT-style DiT that predicts the clean image, with the v-loss and a REPA loss. Training also drops the grounding tokens, so that CFG has a matching unconditional branch.

**Implementation:** We follow the official code [NVlabs/PixelDiT](https://github.com/NVlabs/PixelDiT), except for:

- **Infrastructure:** FSDP replaces DDP, and training does not use torch.compile. A uint8 cache of the center crops gives the same pixels as the official data pipeline, and the random flips occur at each load. The conditional and unconditional halves of CFG run in one batch. The logging, checkpointing, and FID code is our own, with the official reference statistics.

## Results

## Command

### Training

Train PixelDiT2 on ImageNet with batch size 1024 (the paper setting):
```bash
# EXP is b16, l16, or h16 (256x256), or h16_512 (512x512).
EXP=h16 bash scripts/reproduce/pixeldit2_train.sh
```

### Evaluation

Evaluate the FID and IS on 50k samples with 50 Heun steps, and the official CFG of H/16:
```bash
# 256x256: CFG 2.4 on [0.125, 0.9], the default.
EXP_NAME=<log_dir_name> bash scripts/reproduce/pixeldit2_eval.sh

# 512x512: CFG 2.1 on [0.1, 0.925].
EXP_NAME=<log_dir_name> RESOLUTION=512 CFG_SCALE=2.1 CFG_MIN_T=0.1 CFG_MAX_T=0.925 \
  bash scripts/reproduce/pixeldit2_eval.sh
```
