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

- **[2026⁠-⁠11⁠-⁠10]** Add [TwinTilt](TWINTILT.md) training and evaluation for PickScore, GenEval, OCR, and multi-reward. See the [Chinese guide](TWINTILT_CONFIG_ZH.md).
- **[2026⁠-⁠11⁠-⁠04]** Add [DiffusionART](reproduce/diffusion_art) on SD3.5-M with GenEval, OCR, and PickScore rewards.
- **[2026⁠-⁠11⁠-⁠03]** Add [ReFL](reproduce/refl) on SD3.5-M with the PickScore reward.
- **[2026⁠-⁠10⁠-⁠18]** Add [DiffusionNFT](reproduce/diffusion_nft) on SD3.5-M with GenEval, OCR, and PickScore rewards.
- **[2026⁠-⁠10⁠-⁠17]** Add [Flow-GRPO](reproduce/flow_grpo) on SD3.5-M with GenEval, OCR, and PickScore rewards.
- **[2026⁠-⁠10⁠-⁠14]** Add [MeanFlow](reproduce/meanflow) and [iMeanFlow](reproduce/imeanflow) on DiT with ImageNet.

## 📋 Algorithms

**Click an algorithm** to see its introduction, results, and commands to run. The last
column compares one main result of our reproduction with the paper.

<table>
  <tr>
    <th>Algorithm</th>
    <th>Paper</th>
    <th>Released</th>
    <th>Ours / Paper</th>
  </tr>
  <tr>
    <td colspan="4" align="center"><b>Visual Generation - Paradigms</b></td>
  </tr>
  <tr>
    <td nowrap><a href="reproduce/imeanflow">iMeanFlow</a></td>
    <td>
      <i>Improved Mean Flows: On the Challenges of Fastforward Generative Models</i>
      (<a href="https://arxiv.org/abs/2512.02012">CVPR 2026</a>)
    </td>
    <td nowrap>2025-12-01</td>
    <td nowrap>B/2 FID: 3.54 / 3.39</td>
  </tr>
  <tr>
    <td nowrap><a href="reproduce/meanflow">MeanFlow</a></td>
    <td>
      <i>Mean Flows for One-step Generative Modeling</i>
      (<a href="https://arxiv.org/abs/2505.13447">NeurIPS 2025</a>)
    </td>
    <td nowrap>2025-05-19</td>
    <td nowrap>
      B/2 FID: 6.26 / 6.17<br>
      M/2 FID: 5.02 / 5.01<br>
      L/2 FID: 3.84 / 3.84
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center"><b>Visual Generation - Post-Training</b></td>
  </tr>
  <tr>
    <td nowrap><a href="reproduce/diffusion_art">DiffusionART</a></td>
    <td>
      <i>DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion
      Models</i>
      (<a href="https://arxiv.org/abs/2610.12572">arXiv 2026</a>)
    </td>
    <td nowrap>2026-10-08</td>
    <td nowrap>
      GenEval: 0.93 / -<br>
      OCR: 0.98 / -<br>
      PickScore: 23.38 / -
    </td>
  </tr>
  <tr>
    <td nowrap><a href="reproduce/diffusion_nft">DiffusionNFT</a></td>
    <td>
      <i>DiffusionNFT: Online Diffusion Reinforcement with Forward Process</i>
      (<a href="https://arxiv.org/abs/2509.16117">ICLR 2026</a>)
    </td>
    <td nowrap>2025-09-19</td>
    <td nowrap>
      GenEval: 0.95 / 0.98<br>
      OCR: 0.97 / 0.97<br>
      PickScore: 24.04 / 24.03
    </td>
  </tr>
  <tr>
    <td nowrap><a href="reproduce/flow_grpo">Flow-GRPO</a></td>
    <td>
      <i>Flow-GRPO: Training Flow Matching Models via Online RL</i>
      (<a href="https://arxiv.org/abs/2505.05470">NeurIPS 2025</a>)
    </td>
    <td nowrap>2025-05-08</td>
    <td nowrap>
      GenEval: 0.94 / 0.95<br>
      OCR: 0.91 / 0.92<br>
      PickScore: 23.25 / 23.31
    </td>
  </tr>
  <tr>
    <td nowrap><a href="reproduce/refl">ReFL</a></td>
    <td>
      <i>ImageReward: Learning and Evaluating Human Preferences for Text-to-Image
      Generation</i>
      (<a href="https://arxiv.org/abs/2304.05977">NeurIPS 2023</a>)
    </td>
    <td nowrap>2023-04-12</td>
    <td nowrap>PickScore: 23.93 / 23.92</td>
  </tr>
</table>

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
