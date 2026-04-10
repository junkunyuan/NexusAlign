<p align="center">
    <br>
    <img src="assets/logo.png" width="400"/>
    <br>
<p>

<p align="center">
  <a href="https://github.com/junkunyuan/NexusAlign/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg" alt="PyTorch"></a>
</p>

---

NexusAlign is a unified and extensible framework for aligning foundation models.

- [📦 Installation](#-installation)
- [🚀 Training](#-training)
- [📊 Evaluation](#-evaluation)
- [🛠️ Utilities](#️-utilities)
- [📚 Citation](#-citation)

## 📦 Installation

**Requirements**: Python ≥3.10, PyTorch ≥2.5, and CUDA.

```bash
git clone https://github.com/junkunyuan/NexusAlign.git
cd NexusAlign
pip install -e .
```

## 🚀 Training

**Launch training:** use command `nexus-align train`. 

**Load huggingface datasets/models/pipelines from local:** set `common.data_and_model_dir`.

**Config options:**
- ⌨️ **CLI overrides**: append `key=value`.
- 📄 **Config file**: `--config` / `-c` with a YAML file path.

```bash
# Example (single-node): 4 GPUs
nexus-align train \
  --nproc_per_node=4 \
  model=flux \
  log.exp_info=train-exp \
  common.data_and_model_dir=my_hf_cache
```

```bash
# Example (multi-node): 2 nodes, 8 GPUs per node
nexus-align train \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=${NODE_RANK} \
  --master_addr=$(hostname -I | awk '{print $1}') \
  --master_port=29500 \
  -c my_train_config.yaml
```

## 📊 Evaluation

**Launch evaluation:** use command `nexus-align eval`.

**Load huggingface datasets/models/pipelines from local:** set `common.data_and_model_dir`.

**Use a trained checkpoint:** set `model.eval.ckpt_path` (e.g. `checkpoints/model.pt`).

**Config options:**
- ⌨️ **CLI overrides**: append `key=value`.
- 📄 **Config file**: `--config` / `-c` with a YAML file path.

```bash
# Example (single-node): 4 GPUs
nexus-align eval \
  --nproc_per_node=4 \
  model=flux \
  data=hpd_v2_benchmark \
  reward_model=hps_v2 \
  log.exp_info="eval-exp" \
  common.data_and_model_dir=my_hf_cache \
  model.eval.ckpt_path=checkpoints/exp_name/model.pt
```

```bash
# Example (multi-node): 2 nodes, 8 GPUs per node
nexus-align eval \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_IP} \
  --master_port=29500 \
  -c my_eval_config.yaml
```

## 🛠️ Utilities

### Download models/datasets

We provide a simple script to download pre-defined models/datasets used for training/evaluation.

Example usage:
```bash
nexus-align download --repo_id "black-forest-labs/FLUX.1-dev" --cache_dir /home/my_name/data_and_model
```

### Plot results

We provide a simple script to plot results from training logs.

Example usage:
```bash
nexus-align draw --result logs/my_exp/results.jsonl
```

After excuting, a plot will be saved to `logs/my_exp/results.png` looks like:
<p align="center">
    <br>
    <img src="assets/example_results.png" width="500"/>
    <br>
<p>

## 📚 Citation

```bibtex
@misc{nexus-align-2026,
  title = {NexusAlign: A Unified and Extensible Framework for Aligning Foundation Models},
  author = {Junkun Yuan, You Huang},
  year = {2026},
  url = {https://github.com/junkunyuan/NexusAlign}
}
```
