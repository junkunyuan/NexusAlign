# MeanFlow

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *Mean Flows for One-step Generative Modeling* ([NeurIPS 2025](https://arxiv.org/abs/2505.13447))

**Summary:** Model the mean velocity over a time interval. This trains a one-step generator from scratch. The model learns CFG during training, so sampling needs no CFG.

**Implementation:** We follow the official JAX code [Gsunshine/meanflow](https://github.com/Gsunshine/meanflow) and the unofficial PyTorch port [zhuyu-cs/MeanFlow](https://github.com/zhuyu-cs/MeanFlow), except for:

- **Batch 4096:** Some runs use batch 4096 instead of 256 to train faster, with learning rate 4e-4 and EMA decay 0.9984.
- **Infrastructure:** FSDP and `torch.compile` speed up training. The logging and checkpointing code is our own.

## Results

**Table 2 of the paper:** ImageNet 256x256, 1-NFE.

| Model | Batch | LR | EMA | #GPU | FID | IS | Run |
|---|---|---|---|---|---|---|---|
| B/2 (paper) | 256 | 1e-4 | 0.9999 | - | 6.17 | - | - |
| B/2 (ours) | 256 | 1e-4 | 0.9999 | 16 | 6.26 | 205.1 | [run](b2-20261010_015217) |
| B/2 (ours) | 4096 | 4e-4 | 0.9984 | 16 | 6.20 | 203.9 | [run](b2-20261013_063753) |
| B/2 (ours) | 4096 | 4e-4 | 0.9984 | 64 | 6.40 | 202.8 | [run](b2-20261010_055205) |
| M/2 (paper) | 256 | 1e-4 | 0.9999 | - | 5.01 | - | - |
| M/2 (ours) | 256 | 1e-4 | 0.9999 | 16 | 5.02 | 246.3 | [run](m2-20261010_015424) |
| M/2 (ours) | 4096 | 4e-4 | 0.9984 | 64 | 5.08 | 237.7 | [run](m2-20261010_145954) |
| L/2 (paper) | 256 | 1e-4 | 0.9999 | - | 3.84 | - | - |
| L/2 (ours) | 256 | 1e-4 | 0.9999 | 16 | 3.84 | 235.0 | [run](l2-20261010_020033) |
| L/2 (ours) | 4096 | 4e-4 | 0.9984 | 64 | 3.38 | 263.2 | [run](l2-20261011_052151) |
| XL/2 (paper) | 256 | 1e-4 | 0.9999 | - | 3.43 | - | - |
| XL/2 (ours) | 4096 | 4e-4 | 0.9984 | 64 | 3.11 | 247.1 | [run](xl2-20261012_025403) |

**Note:** These old runs chose r = t and the label dropout separately for each sample. The code now applies both to the first samples of each batch, as the official code does. This change has only a small effect on the results.

## Command

### Training

Train MeanFlowDiT on ImageNet with batch size 256 (the paper setting):
```bash
# EXP is b2, m2, l2, or xl2.
EXP=b2 bash scripts/reproduce/meanflow_train.sh
```

Train MeanFlowDiT on ImageNet with batch size 4096 (the faster setting):
```bash
EXP=b2 bash scripts/reproduce/meanflow_train.sh \
  trainer.train_batch_size=4096 \
  trainer.learning_rate=4e-4 \
  trainer.ema_decay=0.9984
```

### Evaluation

Evaluate the FID and IS on 50k samples:
```bash
EXP_NAME=<log_dir_name> bash scripts/reproduce/meanflow_eval.sh
```
