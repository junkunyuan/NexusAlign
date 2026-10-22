# MeanFlow

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** [*Mean Flows for One-step Generative Modeling* (NeurIPS 2025)](https://arxiv.org/abs/2505.13447)

**Official code:** [Gsunshine/meanflow](https://github.com/Gsunshine/meanflow) (JAX).

**Unofficial code:** [zhuyu-cs/MeanFlow](https://github.com/zhuyu-cs/MeanFlow) (PyTorch).

**Summary:** Model the average velocity over an interval — a one-step generator trained from scratch, with guidance in the field.

## Results

**Table 2 of paper: Class-conditional generation on ImageNet 256x256, 1-NFE.**

| Model | Batch | LR | EMA | #GPU | FID | IS | Run |
|---|---|---|---|---|---|---|---|
| B/2 (paper) | 256 | 1e-4 | 0.9999 | - | 6.17 | - | - |
| B/2 (ours) | 256 | 1e-4 | 0.9999 | 16 | 6.26 | 205.1 | [run](b2-20261010_015217) |
| B/2 (ours) | 4096 | 4e-4 | 0.9984 | 16 | 6.21 | 203.9 | [run](b2-20261013_063753) |
| B/2 (ours) | 4096 | 4e-4 | 0.9984 | 64 | 6.32 | 202.8 | [run](b2-20261010_055205) |
| M/2 (paper) | 256 | 1e-4 | 0.9999 | - | 5.01 | - | - |
| M/2 (ours) | 256 | 1e-4 | 0.9999 | 16 | 5.02 | 246.3 | [run](m2-20261010_015424) |
| M/2 (ours) | 4096 | 4e-4 | 0.9984 | 64 | 5.02 | 237.7 | [run](m2-20261010_145954) |
| L/2 (paper) | 256 | 1e-4 | 0.9999 | - | 3.84 | - | - |
| L/2 (ours) | 256 | 1e-4 | 0.9999 | 16 | 3.89 | 235.0 | [run](l2-20261010_020033) |
| L/2 (ours) | 4096 | 4e-4 | 0.9984 | 64 | 3.36 | 263.2 | [run](l2-20261011_052151) |
| XL/2 (paper) | 256 | 1e-4 | 0.9999 | - | 3.43 | - | - |
| XL/2 (ours) | 4096 | 4e-4 | 0.9984 | 64 | 3.15 | 247.1 | [run](xl2-20261012_025403) |

## Command

### Training

Train on ImageNet 256x256 with batch size 256, as in the paper:
```bash
# EXP is b2, m2, l2, or xl2.
EXP=b2 bash scripts/reproduce/meanflow_train.sh
```

Train with batch size 4096, which is more efficient:
```bash
EXP=b2 bash scripts/reproduce/meanflow_train.sh \
  trainer.train_batch_size=4096 trainer.learning_rate=4e-4 trainer.ema_decay=0.9984
```

### Evaluation

FID and IS on 50k samples:
```bash
EXP_NAME=<log_dir_name> bash scripts/reproduce/meanflow_eval.sh
```
