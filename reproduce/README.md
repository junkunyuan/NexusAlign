# Reproductions

## 📋 Algorithms

**Click an algorithm** to see its introduction, results, and commands to run.

### Visual Generation

| Category | Algorithm | Paper | Released |
|---|---|---|---|
| Paradigms | [iMeanFlow](imeanflow) | *Improved Mean Flows: On the Challenges of Fastforward Generative Models* ([CVPR 2026](https://arxiv.org/abs/2512.02012)) | 2025⁠-⁠12⁠-⁠01 |
| Paradigms | [MeanFlow](meanflow) | *Mean Flows for One-step Generative Modeling* ([NeurIPS 2025](https://arxiv.org/abs/2505.13447)) | 2025⁠-⁠05⁠-⁠19 |
| Post⁠-⁠Training | [DiffusionART](diffusion_art) | *DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models* ([arXiv 2026](https://arxiv.org/abs/2610.12572)) | 2026⁠-⁠10⁠-⁠08 |
| Post⁠-⁠Training | [DiffusionNFT](diffusion_nft) | *DiffusionNFT: Online Diffusion Reinforcement with Forward Process* ([ICLR 2026](https://arxiv.org/abs/2509.16117)) | 2025⁠-⁠09⁠-⁠19 |
| Post⁠-⁠Training | [Flow-GRPO](flow_grpo) | *Flow-GRPO: Training Flow Matching Models via Online RL* ([NeurIPS 2025](https://arxiv.org/abs/2505.05470)) | 2025⁠-⁠05⁠-⁠08 |
| Post⁠-⁠Training | [ReFL](refl) | *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation* ([NeurIPS 2023](https://arxiv.org/abs/2304.05977)) | 2023⁠-⁠04⁠-⁠12 |

## 📝 TODO

- [ ] **Rerun MeanFlow** with the fixed sampling. It now sets r = t and drops the labels
  on the first samples of each batch, as the official code does.
- [ ] **Rerun iMeanFlow B/2 at batch 1024,** the paper setting, with the fixed sampling
  and the per-channel latent normalization.
