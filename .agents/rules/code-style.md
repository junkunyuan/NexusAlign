---
paths:
  - "**/*"
---

# Code Style

Follow these conventions in every file of the repo. They add to
[common-code-style](common-code-style.md).

1. **Name the registry keys and runs consistently.** Registry keys and config `name`
   values are snake_case, for example `imagenet_1k` and `meanflowdit_b_2`. A run
   directory joins the dataset, the model, the algorithm, and the timestamp, for
   example `imagenet_1k-meanflowdit_b_2-meanflow-20260903_180000`.
2. **Order the components as datasets, models, algorithms, trainers, evaluators,
   rewards.** Use this order wherever several of them appear together: imports,
   registrations, config groups and fields, arguments, calls, and docs. For example,
   `src/nexus_align/__init__.py` imports `nexus_align.datasets`, then `.models`,
   `.algorithms`, `.trainers`, `.evaluators`, and `.rewards`. A component's own config
   comes first in its arguments, for example `cfg_trainer, train_dataset, model,
   algorithm, reward`.
3. **Order the training counters as epoch, step, total_step.** Use this order wherever
   they appear together: definitions, arguments, calls, dict keys, config keys, and
   docs. For example, `save_ckpt_every_n_epochs` comes before `save_ckpt_every_n_steps`.
4. **Keep the comments of `*_train.sh` and `*_eval.sh` short.** The header names the
   paper table and the official settings that the script follows. It also gives one
   usage example. The paper README and the script's error messages cover the rest.
   Comment other lines only where the code alone is not clear.
