# iMeanFlow

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *Improved Mean Flows: On the Challenges of Fastforward Generative Models* ([CVPR 2026](https://arxiv.org/abs/2512.02012))

**Summary:** Change the self-dependent target of MeanFlow into a plain velocity regression. Also give the guidance scale to the model as one more condition, so you can change it at test time.

**Implementation:** We follow the official JAX code [Lyy-iiis/imeanflow](https://github.com/Lyy-iiis/imeanflow) and its [`torch` branch](https://github.com/Lyy-iiis/imeanflow/tree/torch) for attention and evaluation, except for:

- **Batch 4096:** Some runs use batch 4096 instead of 1024 to train faster, with learning rate 2e-4 and EMA decay 0.9996.
- **Infrastructure:** FSDP and `torch.compile` speed up training. The logging and checkpointing code is our own.

## Results

**Table 2 of the paper:** ImageNet 256x256, 1-NFE.

| Model | Batch | LR | EMA | FID | IS | Run |
|---|---|---|---|---|---|---|
| B/2 (paper) | 1024 | 1e-4 | 0.9999 | 3.39 | 255.3 | - |
| B/2 (ours) | 4096 | 2e-4 | 0.9996 | 3.54 | 241.1 | [run](b2-20261031_001218) |

**Note:** We did not run the paper setting (batch 1024), because it trains too slowly.

## Command

### Training

Train iMeanFlowDiT on ImageNet with batch size 1024 (the paper setting):
```bash
# EXP is b2, m2, l2, or xl2.
EXP=b2 bash scripts/reproduce/imeanflow_train.sh
```

Train iMeanFlowDiT on ImageNet with batch size 4096 (the faster setting):
```bash
EXP=b2 bash scripts/reproduce/imeanflow_train.sh \
  trainer.train_batch_size=4096 \
  trainer.learning_rate=2e-4 \
  trainer.ema_decay=0.9996
```

### Evaluation

Evaluate the FID and IS on 50k samples:
```bash
EXP_NAME=<log_dir_name> bash scripts/reproduce/imeanflow_eval.sh
```
