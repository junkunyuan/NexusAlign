# PixelDiT

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *PixelDiT: Pixel Diffusion Transformers for Image Generation* ([CVPR 2026](https://arxiv.org/abs/2511.20645))

**Summary:** Train a diffusion transformer on pixels with two levels. A patch-level DiT on 16x16 patches models the semantics. Then pixel-level PiT blocks refine the details of each pixel, with an adaLN that gives each pixel its own parameters. To keep the attention affordable, each PiT block compacts the pixels of a patch into one token before the attention, and expands the output again. The model predicts the velocity, and a REPA loss aligns its patch tokens with DINOv2 features.

**Implementation:** We follow the official code [NVlabs/PixelDiT](https://github.com/NVlabs/PixelDiT), except for:

- **REPA encoder:** The DINOv2-B/14 comes from timm, with the position resampling of the official DINOv2 code, and not from torch.hub. Its features are the same up to one bf16 step.
- **512x512 fine-tuning:** The run resumes the model, the EMA, and the Adam state of the 320-epoch run. The official recipe does not say which states it keeps.
- **FID:** torch-fidelity's Inception scores the samples against ADM's reference batch, and not the TensorFlow evaluation suite of ADM.
- **Infrastructure:** FSDP replaces DDP, and training does not use torch.compile. A uint8 cache of the center crops gives the same pixels as the official data pipeline. The logging, checkpointing, and FID code is our own.

## Results

## Command

### Training

Train PixelDiT-B or PixelDiT-L on ImageNet 256x256 with batch size 256 (the paper setting):
```bash
EXP=b16 bash scripts/reproduce/pixeldit_train.sh  # or EXP=l16
```

Train PixelDiT-XL in three stages. Each stage resumes the last checkpoint of the stage before it:
```bash
# Epochs 0 to 160 with the learning rate 1e-4.
EXP=xl16_ep160 bash scripts/reproduce/pixeldit_train.sh

# Epochs 160 to 320 with the learning rate 1e-5 and the gradient clipping 0.5.
EXP=xl16_ep320 bash scripts/reproduce/pixeldit_train.sh trainer.resume_path=<checkpoint>

# Epochs 320 to 850 on ImageNet 512x512.
EXP=xl16_512 bash scripts/reproduce/pixeldit_train.sh trainer.resume_path=<checkpoint>
```

### Evaluation

Evaluate the FID and IS on 50k samples with 100 FlowDPMSolver steps:
```bash
# 320 epochs: CFG 2.75 on [0.1, 0.9], the default.
EXP_NAME=<log_dir_name> bash scripts/reproduce/pixeldit_eval.sh

# 80 epochs: CFG 3.25 on [0.1, 1.0].
EXP_NAME=<log_dir_name> CFG_SCALE=3.25 CFG_MAX_T=1.0 bash scripts/reproduce/pixeldit_eval.sh

# 512x512: CFG 3.5 on [0.1, 1.0], with the sampling time shift 2.0.
EXP_NAME=<log_dir_name> RESOLUTION=512 CFG_SCALE=3.5 CFG_MAX_T=1.0 SAMPLE_TIME_SHIFT=2.0 \
  bash scripts/reproduce/pixeldit_eval.sh
```
