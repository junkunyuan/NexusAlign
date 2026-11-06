<p align="center">
  <img src="assets/logo.png" alt="NexusAlign" width="400">
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-ee4c2c.svg)](https://pytorch.org/)

- NexusAlign is a unified research framework to reproduce and extend foundation model algorithms.
- We design it for fidelity over convenience, unification over duplication, and simplicity over generality.

## 📑 Table of Contents

- [🔥 Updates](#-updates)
- [📋 Algorithms](#-algorithms)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📚 Citation](#-citation)

## 🔥 Updates

- **[2026⁠-⁠11⁠-⁠04]** Add [DiffusionART](reproduce/diffusion_art) on SD3.5-M with GenEval, OCR, and PickScore rewards.
- **[2026⁠-⁠11⁠-⁠03]** Add [ReFL](reproduce/refl) on SD3.5-M with the PickScore reward.
- **[2026⁠-⁠10⁠-⁠18]** Add [DiffusionNFT](reproduce/diffusion_nft) on SD3.5-M with GenEval, OCR, and PickScore rewards.
- **[2026⁠-⁠10⁠-⁠17]** Add [Flow-GRPO](reproduce/flow_grpo) on SD3.5-M with GenEval, OCR, and PickScore rewards.
- **[2026⁠-⁠10⁠-⁠14]** Add [MeanFlow](reproduce/meanflow) and [iMeanFlow](reproduce/imeanflow) on DiT with ImageNet.

## 📋 Algorithms

**Click an algorithm** to see its introduction, results, and commands to run.

### Visual Generation

| Category | Algorithm | Paper | Released |
|---|---|---|---|
| Paradigms | [iMeanFlow](reproduce/imeanflow) | *Improved Mean Flows: On the Challenges of Fastforward Generative Models* ([CVPR 2026](https://arxiv.org/abs/2512.02012)) | 2025⁠-⁠12⁠-⁠01 |
| Paradigms | [MeanFlow](reproduce/meanflow) | *Mean Flows for One-step Generative Modeling* ([NeurIPS 2025](https://arxiv.org/abs/2505.13447)) | 2025⁠-⁠05⁠-⁠19 |
| Post⁠-⁠Training | [DiffusionART](reproduce/diffusion_art) | *DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models* ([arXiv 2026](https://arxiv.org/abs/2610.12572)) | 2026⁠-⁠10⁠-⁠08 |
| Post⁠-⁠Training | [DiffusionNFT](reproduce/diffusion_nft) | *DiffusionNFT: Online Diffusion Reinforcement with Forward Process* ([ICLR 2026](https://arxiv.org/abs/2509.16117)) | 2025⁠-⁠09⁠-⁠19 |
| Post⁠-⁠Training | [Flow-GRPO](reproduce/flow_grpo) | *Flow-GRPO: Training Flow Matching Models via Online RL* ([NeurIPS 2025](https://arxiv.org/abs/2505.05470)) | 2025⁠-⁠05⁠-⁠08 |
| Post⁠-⁠Training | [ReFL](reproduce/refl) | *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation* ([NeurIPS 2023](https://arxiv.org/abs/2304.05977)) | 2023⁠-⁠04⁠-⁠12 |

## 📦 Installation

**Requirements:** Linux, Python ≥3.11, and NVIDIA GPUs.

```bash
git clone https://github.com/junkunyuan/NexusAlign.git && cd NexusAlign
pip install uv && uv sync
source .venv/bin/activate
```

## 🚀 Quick Start

**Example:** train MeanFlow-B/2 on ImageNet 256×256, evaluate it, and stop the jobs.

```bash
# Train. The script saves the run to logs/meanflow_b2_reproduce-<timestamp>.
EXP=b2 bash scripts/reproduce/meanflow_train.sh

# Evaluate the FID and IS of each checkpoint in the run on 50k samples.
EXP_NAME=meanflow_b2_reproduce-<timestamp> bash scripts/reproduce/meanflow_eval.sh

# Show the jobs of this repo, then stop them after you confirm.
bash scripts/tools/kill_jobs.sh
```

## 📚 Citation

```bibtex
@misc{nexus-align-2026,
  title = {NexusAlign: A Unified Research Framework for Reproducing and Extending Foundation Model Algorithms},
  author = {Junkun Yuan},
  year = {2026},
  url = {https://github.com/junkunyuan/NexusAlign}
}
```
