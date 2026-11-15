# TwinTilt

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** TwinTilt has no paper yet. It extends *Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models* ([arXiv 2026](https://arxiv.org/abs/2603.12893)).

**Summary:** For each group of rollouts of one prompt, fit one correction in the span of the centered endpoints, so that the softmax of the endpoints along it matches the tilted rewards. Then regress the rollout velocities of all images and timesteps of the group onto the velocities minus the scaled correction.

**Implementation:** We follow the TwinTilt code [shenytzzz/twintilt](https://github.com/shenytzzz/twintilt) (private), a NexusAlign fork that extends the official FDFO code [NVlabs/finite-difference-flow-optimization](https://github.com/NVlabs/finite-difference-flow-optimization), except for:

- **Training length:** OCR trains 170 epochs, GenEval 800, and PickScore 600, instead of 500. The task scores of our longer OCR and GenEval runs are near their plateau at these steps. PickScore still improves after step 500, but step 600 of our 1500-step run already beats the baselines on 8 of the 11 metrics.
- **Base KL:** For OCR, GenEval, and PickScore, the KL to the base model uses CFG 4.5, as in AWM, and the weight 0.01, not 0.004. Without this change, the general metrics of OCR fall far below the base model, for example HPSv3 from 9.5 to 7.7.
- **Precision:** The frozen transformer weights and the text encoders are fp16, not bf16, and the VAE is fp32, not bf16. The compute dtype stays fp16 with a gradient scaler. TF32 matmuls stay on, while TwinTilt turns them off.
- **Rewards:** The PickScore reward resizes the float pixels with our antialiased bicubic resize, not with the CLIP processor of transformers. In the multi-reward, HPSv2.1 scores the uint8 pixels in fp32, not the float pixels under bf16 autocast.
- **EMA:** The decay counts every optimizer step, also a step that the gradient scaler skips.
- **Evaluation:** We evaluate as Flow-GRPO does, so that the results compare with the other algorithms: the EMA weights, CFG 4.5, and 128 T5 tokens, with our Euler loop under fp16 autocast and one noise per image. As DiffusionNFT does for its OCR runs, the GenEval run evaluates its own task without CFG. The PickScore run evaluates all tasks without CFG. TwinTilt evaluates the student without CFG, with 256 T5 tokens, in the diffusers pipeline in bf16, and with the same noise for all prompts.
- **Options:** We leave out the SA solver and CFG in the rollouts and the policy, which no TwinTilt task uses. The evaluation and the base KL can use CFG.
- **Infrastructure:** FSDP, `torch.compile`, and fp16 frozen weights speed up training and change only the rounding. The prompt sampler, the group noise that the group ID seeds, the reward servers, the checkpoint intervals, and the resume from an epoch end are our own.

## Results

**OCR and GenEval:** SD3.5-M trained on visual text rendering and GenEval, evaluated with CFG 4.5, except GenEval† of the GenEval row without CFG.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M (ours) | 0.64 | 0.59 | 21.72 | 22.34 | 0.284 | 0.280 | 5.38 | 0.85 | 3.06 | 4.07 | 9.46 | [run](ocr-20261115_184913) |
| OCR (ours) | 0.72 | **0.95** | 21.58 | 22.41 | 0.290 | 0.283 | 5.32 | 0.99 | 3.14 | 4.12 | 8.83 | [run](ocr-20261115_184913) |
| GenEval (ours) | **0.96** | 0.67 | 21.55 | 22.53 | 0.294 | 0.294 | 5.36 | 1.22 | 3.29 | 4.15 | 10.62 | [run](geneval-20261115_153411) |

**PickScore:** SD3.5-M trained on PickScore, evaluated with CFG off.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 0.23 | 0.13 | 19.82 | 20.54 | 0.239 | 0.206 | 5.17 | -0.58 | 2.02 | 3.36 | -1.07 | [run](pickscore-20261115_153604) |
| PickScore (ours) | 0.52 | 0.67 | **24.35** | 24.43 | 0.268 | 0.323 | 6.41 | 1.35 | 3.43 | 4.26 | 12.79 | [run](pickscore-20261115_153604) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench.
- **Checkpoints:** the OCR row uses step 170 of a 200-step run, the GenEval row uses step 800 of a 1500-step run, and the PickScore row uses step 600 of a 1500-step run. The configs now train these lengths. The base rows use step 0.
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

Evaluate the EMA weights of each checkpoint, with CFG 4.5. To evaluate without CFG, as for GenEval† of the GenEval runs and for the PickScore runs, add `evaluator.guidance_scale=1.0`:
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
