# TwinTilt

- [Introduction](#introduction)
- [Results](#results)
- [Command](#command)

## Introduction

**Paper:** TwinTilt has no paper yet. It extends *Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models* ([arXiv 2026](https://arxiv.org/abs/2603.12893)).

**Summary:** For each group of rollouts of one prompt, fit one correction in the span of the centered endpoints, so that the softmax of the endpoints along it matches the tilted rewards. Then regress the rollout velocities of all images and timesteps of the group onto the velocities minus the scaled correction.

**Implementation:** We follow the TwinTilt code [shenytzzz/twintilt](https://github.com/shenytzzz/twintilt) (private), a NexusAlign fork that extends the official FDFO code [NVlabs/finite-difference-flow-optimization](https://github.com/NVlabs/finite-difference-flow-optimization), except for:

- **Training length:** OCR trains 170 epochs, GenEval 900, and PickScore 600, instead of 500. The multi-reward keeps the 500 of the code, and its scores still rise there. The task scores of our longer OCR and GenEval runs are near their plateau at these steps. PickScore still improves after step 500, but step 600 of our 1500-step run already beats the baselines on 8 of the 11 metrics.
- **MiniMax-H3-Image:** The HPSv3 task trains the H3-Image DiT on its native six-step PDD sampler at 768 pixels, with 48 groups of 24 images for 100 steps. FSDP shards the 33B model, its EMA, and the text encoder over the GPUs of each node, and the rollout and the training forward passes take four members at a time. HPSv3 scores four images per request, because it needs tens of GB for a group of 768-pixel images.
- **Base KL:** For OCR, GenEval, and PickScore, the KL to the base model uses CFG 4.5, as in AWM, and the weight 0.01, not 0.004. Without this change, the general metrics of OCR fall far below the base model, for example HPSv3 from 9.5 to 7.7. The multi-reward task keeps the CFG-free KL of the code, because it already scores the general metrics.
- **Qwen judge:** The judge task trains SD3.5-M on the PickScore prompts with the CFG-free base KL of the multi-reward task, because the judge also scores the general quality. The TwinTilt code trains this judge task for 50 epochs. We train SD3.5-M, so that the judge compares with our other SD3.5-M tasks, and for 100 epochs, because the scores of our 50-epoch run still rise at its end.
- **Precision:** The frozen transformer weights and the text encoders are fp16, not bf16, and the VAE is fp32, not bf16. The compute dtype stays fp16 with a gradient scaler. TF32 matmuls stay on, while TwinTilt turns them off.
- **Rewards:** The PickScore reward resizes the float pixels with our antialiased bicubic resize, not with the CLIP processor of transformers. In the multi-reward, HPSv2.1 scores the uint8 pixels in fp32, not the float pixels under bf16 autocast.
- **EMA:** The decay counts every optimizer step, also a step that the gradient scaler skips.
- **Evaluation:** We evaluate as Flow-GRPO does, so that the results compare with the other algorithms: the EMA weights, CFG 4.5, and 128 T5 tokens, with our Euler loop under fp16 autocast and one noise per image. As DiffusionNFT does for its OCR runs, the GenEval run evaluates its own task without CFG. The PickScore run evaluates all tasks without CFG. TwinTilt evaluates the student without CFG, with 256 T5 tokens, in the diffusers pipeline in bf16, and with the same noise for all prompts.
- **Options:** We leave out the SA solver and CFG in the rollouts and the policy, which no TwinTilt task uses. The evaluation and the base KL can use CFG.
- **Infrastructure:** FSDP, `torch.compile`, and fp16 frozen weights speed up training and change only the rounding. The prompt sampler, the group noise that the group ID seeds, the reward servers, which include one judge per node that its ranks share, the checkpoint intervals, and the resume from an epoch end are our own.

## Results

**OCR and GenEval:** SD3.5-M trained on visual text rendering and GenEval, evaluated with CFG 4.5, except GenEval† of the GenEval row without CFG.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M (ours) | 0.64 | 0.59 | 21.72 | 22.34 | 0.284 | 0.280 | 5.38 | 0.85 | 3.06 | 4.07 | 9.46 | [run](ocr-20261115_184913) |
| OCR (ours) | 0.72 | **0.95** | 21.58 | 22.41 | 0.290 | 0.283 | 5.32 | 0.99 | 3.14 | 4.12 | 8.83 | [run](ocr-20261115_184913) |
| GenEval (ours) | **0.97** | 0.67 | 21.55 | 22.51 | 0.293 | 0.294 | 5.38 | 1.21 | 3.28 | 4.13 | 10.61 | [run](geneval-20261115_153411) |

**PickScore and multi-reward:** SD3.5-M trained on PickScore and on the multi-reward of GenEval, PickScore, ClipScore, and HPSv2.1, evaluated with CFG off.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD3.5-M w/o CFG (ours) | 0.23 | 0.13 | 19.82 | 20.54 | 0.239 | 0.206 | 5.17 | -0.58 | 2.02 | 3.36 | -1.07 | [run](pickscore-20261115_153604) |
| PickScore (ours) | 0.52 | 0.67 | **24.35** | 24.43 | 0.268 | 0.323 | 6.41 | 1.35 | 3.43 | 4.26 | 12.79 | [run](pickscore-20261115_153604) |
| Multi-reward (ours) | **0.82** | 0.45 | **22.26** | 23.56 | **0.282** | **0.352** | 6.21 | 1.40 | 3.29 | 4.47 | 14.10 | [run](multi_reward-20261116_035928) |

**HPSv3 on MiniMax-H3-Image:** its native six-step sampler at 768 pixels without CFG, with the student weights.

| Model | GenEval† | OCR† | PickScore† | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd | DeQA | HPSv3 | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniMax-H3-Image (ours) | 0.61 | 0.74 | 20.86 | 21.96 | 0.277 | 0.267 | 5.04 | 0.75 | 3.15 | 4.22 | 9.05 | [run](h3_hpsv3-20261116_082643) |
| HPSv3 (ours) | 0.51 | 0.83 | 20.92 | 22.21 | 0.240 | 0.311 | 6.31 | 0.92 | 3.12 | 4.52 | **14.48** | [run](h3_hpsv3-20261116_082643) |

- **Evaluation dataset:** † marks a task-specific dataset. The other columns use DrawBench.
- **Checkpoints:** the OCR row uses step 170 of a 200-step run, the GenEval row uses step 900 of a 1500-step run, the PickScore row uses step 600 of a 1500-step run, the multi-reward row uses its final step 500, and the HPSv3 row uses its final step 100. The configs now train these lengths. The base rows use step 0.
- **Bold:** in-domain results. The multi-reward trains on four of the metrics, so its row bolds four columns.

## Command

### Training

Train SD3.5-M on one task. The default batch of 9 images per GPU works on 8, 16, 32, 64, or 128 GPUs:
```bash
# EXP is pickscore, geneval, ocr, qwen_judge, or multi_reward (on the GenEval
# prompts).
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

# The Qwen judge, on the PickScore test prompts.
EXP_NAME=<log_dir_name> DATA=pickscore REWARDS=qwen_judge \
  bash scripts/reproduce/twintilt_eval.sh
```
