---
paths:
  - "src/nexus_align/configs/**"
  - "src/nexus_align/algorithms/**"
---

# Configs

- Keep an algorithm's hyperparameters strictly aligned between `configs/algorithm/<name>.yaml`
  and the `__init__` of `algorithms/<name>.py`: the same fields, in the same order, under the
  same section comments, with identical trailing comments. `meanflow` is the reference.
- Keep `configs/reproduce/<name>/*.yaml` minimal: only the args that differ between the
  experiments of one paper. An arg shared by all of them (e.g. `model.fsdp.shard: 1`) belongs
  in the algorithm's own defaults (`configs/algorithm/<name>.yaml` or its model config
  `configs/model/<model>.yaml`), not repeated per experiment. Never hoist it into a default
  shared with other algorithms, such as `configs/trainer/sft.yaml` (see the Workflow in
  `AGENTS.md`).
- Set the checkpoint interval (`save_ckpt_every_n_epochs` / `save_ckpt_every_n_steps`) of
  every training config so that a full run saves 8 to 16 checkpoints. Compute it from the
  run's total epochs or steps, and count the initial checkpoint at epoch 0 where the
  trainer saves one.
