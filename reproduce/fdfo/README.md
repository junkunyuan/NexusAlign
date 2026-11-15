# FDFO

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** *Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models* ([arXiv 2026](https://arxiv.org/abs/2603.12893))

**Summary:** Roll out a pair of trajectories per prompt from one noise, with a small churn at each step that makes their details differ. The reward difference ΔR and the endpoint difference Δx of the pair point toward the better image. Then pull the velocities of all steps of both trajectories toward the rollout velocities minus Δx / ‖Δx‖², with an SPO surrogate loss that weights each pull with ΔR.

**Implementation:** We follow the official code [NVlabs/finite-difference-flow-optimization](https://github.com/NVlabs/finite-difference-flow-optimization), under its non-commercial license, except for:

- **Tasks:** We train on our GenEval, OCR, and PickScore tasks with the default hyperparameters of the official code, as for the other algorithms. The paper trains on PickScore, a VLM alignment reward, and their combination, all on the Pick-a-Pic prompts.
- **PickScore:** The reward scores the uint8 images with the PIL processor of Flow-GRPO, not the float pixels with the CLIP processor.
- **Slot shuffle:** Each rank shuffles its own (trajectory, timestep) slots. The official code shuffles the slots of all ranks together. Each optimizer step still trains on a random quarter of the slots of every rank.
- **Prompt sampling:** Each round draws distinct prompts and gives each rank its part, as our other GRPO-style runs do. In the official code, each rank goes through its own shuffled prompts.
- **Evaluation:** We sample the student with the 40-step Euler ODE and no CFG, as the paper does, but on the test prompts of each task and on DrawBench. The paper uses the Pick-a-Pic training prompts, HPDv2, and OneIG-Bench.
- **Infrastructure:** `torch.compile`, FSDP, fp16 frozen weights and text encoders, an fp32 VAE, and fp16 initial noise change only the speed and the rounding. The code for logging, checkpointing, and the reward servers is our own.

## Results

## Command

### Training

Train SD3.5-M on one reward with its specific training data. The default 432 pairs per epoch work on 1, 2, 3, 4, 6, 8, 12, 16, 24, or 48 GPUs:
```bash
# EXP is geneval, ocr, or pickscore.
EXP=pickscore bash scripts/reproduce/fdfo_train.sh

# On 128 GPUs, use 256 pairs of 2 per GPU, as the official code does.
EXP=pickscore bash scripts/reproduce/fdfo_train.sh \
  trainer.groups_per_epoch=256 trainer.sample_batch_size=4
```

### Evaluation

```bash
# GenEval, PickScore, and OCR, each on its own test prompts.
for TASK in geneval pickscore ocr; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/fdfo_eval.sh
done

# The other metrics on DrawBench, with two reward models per evaluation to prevent OOM.
for REWARDS in "unifiedreward aesthetic" "deqa imagereward" "hpsv3 pickscore" \
    "clipscore hpsv2"; do
  EXP_NAME=<log_dir_name> DATA=drawbench REWARDS="$REWARDS" \
    bash scripts/reproduce/fdfo_eval.sh
done
```
