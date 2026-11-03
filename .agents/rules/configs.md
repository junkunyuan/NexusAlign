---
paths:
  - "src/nexus_align/configs/**"
  - "src/nexus_align/algorithms/**"
---

# Configs

Follow these conventions to keep the algorithm and reproduce configs consistent and
minimal.

1. **Mirror each algorithm config in its `__init__`.** `configs/algorithm/<name>.yaml`
   and the `__init__` of `algorithms/<name>.py` have the same fields, in the same order,
   under the same section comments, with identical trailing comments. `meanflow` is the
   reference.
2. **Keep reproduce configs minimal.** `configs/reproduce/<name>/*.yaml` holds only the
   args that differ between the experiments of one paper. Put an arg that all of them
   share (for example, `model.fsdp.shard: 1`) in the algorithm's own defaults:
   `configs/algorithm/<name>.yaml` or its model config `configs/model/<model>.yaml`.
   Never move it into a default that other algorithms share, such as
   `configs/trainer/sft.yaml` (see the Workflow in `AGENTS.md`). Those defaults cannot
   set a `data` arg. Put a shared `data` arg, such as the latent normalization of
   iMeanFlow, in every reproduce config instead.
3. **Give every arg a trailing comment.** Every arg in `configs/` says what it is in a
   short trailing comment. Put a longer explanation on a comment line above the arg.
4. **Copy each reproduce arg's comment from the config that defines it.** Every arg in
   `configs/reproduce/<name>/*.yaml` has the same trailing comment as in the config
   that defines it (for example, `cfg_omega` in `configs/algorithm/meanflow.yaml`).
   State anything special about an experiment's value, such as why it differs from the
   default, after that comment and a `;`. If the line would pass 88 characters, put
   the note on a comment line above the arg instead:
   ```yaml
   max_grad_norm: .inf  # max gradient norm for clipping; the paper uses no clipping
   # The official GenEval reward scores raw pixels in-process.
   jpeg: false  # send JPEG as Flow-GRPO does; false sends lossless PNG, as DiffusionNFT
   ```
5. **Save 8 to 16 checkpoints per run.** Set `save_ckpt_every_n_epochs` or
   `save_ckpt_every_n_steps` in every training config from the run's total epochs or
   steps. Count the initial checkpoint at epoch 0 if the trainer saves one.
