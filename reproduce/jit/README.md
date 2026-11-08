# JiT

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *Back to Basics: Let Denoising Generative Models Denoise* ([arXiv 2025](https://arxiv.org/abs/2511.13720))

**Summary:** Train a plain DiT on large patches of pixels, with no tokenizer, no pre-training, and no extra loss. The network predicts the clean image x, and x gives the velocity v = (x - z_t) / (1 - t) for the flow-matching loss. Clean images lie on a low-dimensional manifold, so x-prediction works also when a patch has more dimensions than the hidden size. With these patches, ε-prediction and v-prediction fail.

**Implementation:** We follow the official PyTorch code [LTH14/JiT](https://github.com/LTH14/JiT), except for:

- **One EMA:** Each run keeps one EMA with the decay 0.9999, which the official evaluation samples with. The paper picks the best of 0.9996, 0.9998, and 0.9999 for each model.
- **Infrastructure:** FSDP replaces DDP, and training does not use torch.compile. A uint8 cache of the center crops gives the same pixels as the official data pipeline, and the random flips occur at each load. The logging, checkpointing, and FID code is our own, with the official reference statistics.

## Results

## Command

### Training

Train JiTDiT on ImageNet with batch size 1024 (the paper setting):
```bash
# EXP is b16, l16, or h16 (256x256), or b32, l32, or h32 (512x512).
EXP=b16 bash scripts/reproduce/jit_train.sh
```

### Evaluation

Evaluate the FID and IS on 50k samples, with the official CFG scale of the model:
```bash
# B/16; jit_eval.sh lists the scales of the other models.
EXP_NAME=<log_dir_name> CFG_SCALE=3.0 bash scripts/reproduce/jit_eval.sh

# The /32 models generate 512x512 images.
EXP_NAME=<log_dir_name> RESOLUTION=512 CFG_SCALE=3.0 bash scripts/reproduce/jit_eval.sh
```
