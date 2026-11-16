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
- **Pairs and length:** The GenEval, OCR, and PickScore runs train 1200 epochs on 128 GPUs with 256 pairs per epoch, as the official code does on 128 GPUs. The paper trains up to 1000 epochs with 432 pairs, which do not split over 128 GPUs.
- **Evaluation:** We sample the student with the 40-step Euler ODE and no CFG, as the paper does, but on the test prompts of each task and on DrawBench. The paper uses the Pick-a-Pic training prompts, HPDv2, and OneIG-Bench.
- **Infrastructure:** `torch.compile`, FSDP, fp16 frozen weights and text encoders, an fp32 VAE, and fp16 initial noise change only the speed and the rounding. The code for logging, checkpointing, and the reward servers is our own.

## Results

**GenEval, OCR, and PickScore:** SD3.5-M trained on GenEval, on visual text rendering, and on PickScore, evaluated with the student weights and CFG off.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 0.22 | 0.13 | 19.83 | 20.51 | 0.237 | 0.206 | 5.14 | -0.58 | 2.02 | 3.36 | -1.02 | [run](pickscore-20261115_220041) |
| GenEval (ours) | **0.64** | 0.21 | 19.35 | 20.75 | 0.258 | 0.206 | 4.27 | -0.34 | 2.18 | 1.76 | -1.62 | [run](geneval-20261115_223814) |
| OCR (ours) | 0.02 | **0.91** | 17.36 | 18.67 | 0.175 | 0.115 | 3.71 | -1.94 | 1.22 | 1.43 | -9.56 | [run](ocr-20261115_221948) |
| PickScore (ours) | 0.51 | 0.63 | **22.52** | 22.94 | 0.271 | 0.289 | 6.17 | 0.93 | 3.21 | 3.67 | 9.50 | [run](pickscore-20261115_220041) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench.
- **Checkpoints:** the GenEval, OCR, and PickScore rows use the final epoch 1200, and the base row uses epoch 0. Epoch 200 of the GenEval run is higher on all 11 metrics (GenEval† 0.83). Epoch 1000 of the OCR run has a higher OCR† (0.98). Epochs 300 and 400 of the PickScore run are higher on most metrics (PickScore† 23.45 and 23.43), as the paper finds the best quality at about 200 epochs.
- **Bold:** in-domain results.
- **Paper:** it reports the rewards only in figures, on the Pick-a-Pic training prompts.

## Command

### Training

Train SD3.5-M on one reward with its specific training data. The default 432 pairs per epoch work on 1, 2, 3, 4, 6, 8, 12, 16, 24, or 48 GPUs:
```bash
# EXP is geneval, ocr, or pickscore.
EXP=pickscore bash scripts/reproduce/fdfo_train.sh

# On 128 GPUs, use 256 pairs of 2 per GPU, as the official code does. The GenEval,
# OCR, and PickScore rows train 1200 epochs.
EXP=pickscore bash scripts/reproduce/fdfo_train.sh \
  trainer.groups_per_epoch=256 trainer.sample_batch_size=4 trainer.epochs=1200
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
