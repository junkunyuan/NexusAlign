---
paths:
  - "reproduce/**"
---

# Reproduction Records

`reproduce/<algorithm>/` is the committed record of one reproduced paper.

## Layout

- A single `README.md`: the results table against the paper, and the commands.
- One `<exp>-<timestamp>/` subfolder per run, copied from `logs/<run>/` with the same
  layout: `config_train.yaml`, `results.jsonl`, `figures/`, and one
  `eval/<evaluator>-<timestamp>/` per evaluation holding `config_eval.yaml`,
  `eval_summary.json`, and `figures/`. Keep only `step.png` and `epoch.png` in each
  `figures/`; they combine the per-metric curves.
- Do not write per-run READMEs, and do not commit checkpoints or logs.

## Private info

Strip private info from the copied configs:

- Make every absolute path relative to the project root
  (`<project_root>/data_and_model/stabilityai/sd-vae-ft-mse` ->
  `data_and_model/stabilityai/sd-vae-ft-mse`, `<project_root>/logs/<run>` -> `logs/<run>`,
  `project_path` -> `.`).
- Shorten the OS string to the kernel version
  (`Linux-6.8.0-45-generic-x86_64-with-glibc2.39` -> `Linux-6.8.0`).
- Drop the `provenance.git` block (branch, commit, dirty).
- Grep the folder afterwards to confirm no username, hostname, absolute path, or commit
  remains. The paper README lists no commit either.

## Paper README

- It has exactly three sections, `Introduction`, `Results`, and `Command`, as
  `reproduce/meanflow/README.md` does; never add other sections such as `Notes` or
  `Deviations`, nor explanatory paragraphs.
- Add a row for each new run; `reproduce/README.md` is a brief introduction to the folder
  and only lists papers, newest first, never results.
- Display every number in a results table as the original paper does, with the same scale
  and decimal places (paper GenEval 0.95 -> ours 0.96, not 0.955); for a metric the paper
  does not report, follow a paper that does (DiffusionNFT for ClipScore and HPSv2.1).
- Give each run one row, linked to its subfolder as `[run](<exp>-<timestamp>)`, where each
  number is the best value of that metric over all trained checkpoints (epoch 0 is the
  base model), so the numbers of one row may come from different checkpoints.
- Add a training-length column (e.g. `#Iter` 1k) or a GPU-count column (`#GPU`, the
  run's `world_size`) only if the rows of one task differ in it.
