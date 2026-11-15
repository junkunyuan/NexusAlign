# TwinTilt

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** TwinTilt has no paper yet. It extends *Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models* ([arXiv 2026](https://arxiv.org/abs/2603.12893)).

**Summary:** For each group of rollouts of one prompt, fit one correction in the span of the centered endpoints, so that the softmax of the endpoints along it matches the tilted rewards. Then regress the rollout velocities of all images and timesteps of the group onto the velocities minus the scaled correction.

**Implementation:** We follow the TwinTilt code [shenytzzz/twintilt](https://github.com/shenytzzz/twintilt) (private), a NexusAlign fork that extends the official FDFO code [NVlabs/finite-difference-flow-optimization](https://github.com/NVlabs/finite-difference-flow-optimization), except for:

- **Training length:** OCR trains 100 epochs instead of 500, with a checkpoint every 5, because its training reward reached 0.98 in about 30 epochs in a run with z-scored advantages.
- **Precision:** The frozen transformer weights and the text encoders are fp16, not bf16, and the VAE is fp32, not bf16. The compute dtype stays fp16 with a gradient scaler. TF32 matmuls stay on, while TwinTilt turns them off.
- **Rewards:** The PickScore reward resizes the float pixels with our antialiased bicubic resize, not with the CLIP processor of transformers. In the multi-reward, HPSv2.1 scores the uint8 pixels in fp32, not the float pixels under bf16 autocast.
- **EMA:** The decay counts every optimizer step, also a step that the gradient scaler skips.
- **Evaluation:** We evaluate as Flow-GRPO does, so that the results compare with the other algorithms: the EMA weights, CFG 4.5, and 128 T5 tokens, with our Euler loop under fp16 autocast and one noise per image. TwinTilt evaluates the student without CFG, with 256 T5 tokens, in the diffusers pipeline in bf16, and with the same noise for all prompts.
- **Options:** We leave out the SA solver and CFG in rollouts and training, which no TwinTilt task uses. The evaluation can use CFG.
- **Infrastructure:** FSDP, `torch.compile`, and fp16 frozen weights speed up training and change only the rounding. The prompt sampler, the group noise that the group ID seeds, the reward servers, the checkpoints every 50 epochs, and the resume from an epoch end are our own.

## Results

**OCR:** SD3.5-M trained on visual text rendering, evaluated with CFG 4.5.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M (ours) | 0.64 | 0.57 | 21.72 | 22.31 | 0.283 | 0.279 | 5.38 | 0.84 | 3.05 | 4.08 | 9.51 | [run](ocr-20261114_230204) |
| OCR (ours) | 0.61 | **0.97** | 21.01 | 21.87 | 0.280 | 0.271 | 5.32 | 0.66 | 3.00 | 4.09 | 7.72 | [run](ocr-20261114_230204) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench.
- **Checkpoint:** the OCR row uses step 70 of 100, and the base row uses step 0.
- **Bold:** in-domain results.

## Command

### Training

Train SD3.5-M on one task. The default batch of 9 images per GPU works on 8, 16, 32, 64, or 128 GPUs:
```bash
# EXP is pickscore, geneval, ocr, or multi_reward (on the GenEval prompts).
EXP=pickscore bash scripts/reproduce/twintilt_train.sh

# On 8 GPUs, 24 images per GPU are about 20% faster.
EXP=pickscore bash scripts/reproduce/twintilt_train.sh trainer.sample_batch_size=24
```

### Evaluation

Evaluate the EMA weights of each checkpoint, with CFG 4.5:
```bash
# PickScore with the PIL and the float-pixel preprocessing.
EXP_NAME=<log_dir_name> DATA=pickscore REWARDS="pickscore twintilt_pickscore" \
  bash scripts/reproduce/twintilt_eval.sh

# GenEval and OCR, each on its own test prompts.
for TASK in geneval ocr; do
  EXP_NAME=<log_dir_name> DATA=$TASK REWARDS=$TASK \
    bash scripts/reproduce/twintilt_eval.sh
done

# The multi-reward and its four components, on the GenEval test prompts.
EXP_NAME=<log_dir_name> DATA=geneval REWARDS=twintilt_multi \
  bash scripts/reproduce/twintilt_eval.sh
```
