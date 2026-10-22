# iMeanFlow

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** [*Improved Mean Flows: On the Challenges of Fastforward Generative Models* (arXiv 2025)](https://arxiv.org/abs/2512.02012)

**Official code:** [Lyy-iiis/imeanflow](https://github.com/Lyy-iiis/imeanflow).

**Summary:** Recast MeanFlow's self-written target as plain velocity regression, and turn its frozen guidance scale into one more test-time condition.

## Results

**Table 2 of paper: Class-conditional generation on ImageNet 256x256, 1-NFE.**

| Model | Batch | LR | EMA | #GPU | FID | IS | Run |
|---|---|---|---|---|---|---|---|
| B/2 (paper) | 1024 | 1e-4 | 0.9999 | - | 3.39 | - | - |
| B/2 (ours) | 1024 | 1e-4 | 0.9999 | 16 | 3.99 | 265.8 | [run](b2-20261013_232426) |
| B/2 (ours) | 4096 | 2e-4 | 0.9996 | 16 | 4.05 | 261.4 | [run](b2-20261014_003708) |
| B/2 (ours) | 4096 | 2e-4 | 0.9996 | 64 | 3.98 | 251.7 | [run](b2-20261013_064639) |

## Command

### Training

Train on ImageNet 256x256 with batch size 1024, as in the paper:
```bash
# EXP is b2, m2, l2, or xl2.
EXP=b2 bash scripts/reproduce/imeanflow_train.sh
```

Train with batch size 4096, which is more efficient:
```bash
EXP=b2 bash scripts/reproduce/imeanflow_train.sh \
  trainer.train_batch_size=4096 trainer.learning_rate=2e-4 trainer.ema_decay=0.9996
```

### Evaluation

FID and IS on 50k samples:
```bash
EXP_NAME=<log_dir_name> bash scripts/reproduce/imeanflow_eval.sh
```
