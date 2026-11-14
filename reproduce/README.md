# Reproductions

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
    <td nowrap><a href="pmeanflow">pMeanFlow</a></td>
    <td>
      <i>One-step Latent-free Image Generation with Pixel Mean Flows</i>
      (<a href="https://arxiv.org/abs/2601.22158">arXiv 2026</a>)
    </td>
    <td nowrap>2026-01-29</td>
    <td nowrap>B/16 FID: 3.43 / 3.12</td>
  </tr>
  <tr>
    <td nowrap><a href="imeanflow">iMeanFlow</a></td>
    <td>
      <i>Improved Mean Flows: On the Challenges of Fastforward Generative Models</i>
      (<a href="https://arxiv.org/abs/2512.02012">CVPR 2026</a>)
    </td>
    <td nowrap>2025-12-01</td>
    <td nowrap>B/2 FID: 3.54 / 3.39</td>
  </tr>
  <tr>
    <td nowrap><a href="meanflow">MeanFlow</a></td>
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
    <td nowrap><a href="diffusion_art">DiffusionART</a></td>
    <td>
      <i>DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion
      Models</i>
      (<a href="https://arxiv.org/abs/2610.12572">arXiv 2026</a>)
    </td>
    <td nowrap>2026-10-08</td>
    <td nowrap>
      GenEval: 0.88 / -<br>
      OCR: 0.98 / -<br>
      PickScore: 23.38 / -
    </td>
  </tr>
  <tr>
    <td nowrap><a href="diffusion_nft">DiffusionNFT</a></td>
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
    <td nowrap><a href="flow_grpo">Flow-GRPO</a></td>
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
    <td nowrap><a href="refl">ReFL</a></td>
    <td>
      <i>ImageReward: Learning and Evaluating Human Preferences for Text-to-Image
      Generation</i>
      (<a href="https://arxiv.org/abs/2304.05977">NeurIPS 2023</a>)
    </td>
    <td nowrap>2023-04-12</td>
    <td nowrap>PickScore: 23.93 / 23.92</td>
  </tr>
</table>

## 📝 TODO

- [ ] **Run JiT B/16** on ImageNet 256x256, and record the run. Then add JiT to the
  Algorithms tables.
- [ ] **Run PixelDiT-XL and PixelDiT2-H/16** on ImageNet 256x256, and record the runs.
  Then add PixelDiT and PixelDiT2 to the Algorithms tables.
- [ ] **Rerun MeanFlow** with the fixed sampling. It now sets r = t and drops the labels
  on the first samples of each batch, as the official code does.
- [ ] **Rerun iMeanFlow B/2 at batch 1024,** the paper setting, with the fixed sampling
  and the per-channel latent normalization.
