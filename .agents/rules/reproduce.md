---
paths:
  - "reproduce/**"
---

# Reproduction Records

`reproduce/<algorithm>/` is the committed record of one reproduced paper. Follow these
conventions when you add or update one.

## Layout

1. **One `README.md` per paper:** how we implemented it, a table that compares our
   results with the paper, and the commands.
2. **One `<exp>-<timestamp>/` subfolder per run,** a copy of `logs/<run>/` with the
   same layout:
   - **Training:** `config_train.yaml`, `results.jsonl`, and `figures/`;
   - **Evaluation:** one `eval/<evaluator>[-<data>]-<timestamp>/` per evaluation, with
     `config_eval.yaml`, `eval_summary.json`, and `figures/`.
3. **Keep only `epoch.png` and `step.png` in each `figures/`.** They combine the
   per-metric curves.
4. **Commit no per-run READMEs, checkpoints, or logs.**

## Private info

Strip private info from the copied configs:

1. **Make every absolute path relative to the project root:**
   - **Assets:** `<project_root>/data_and_model/stabilityai/sd-vae-ft-mse` ->
     `data_and_model/stabilityai/sd-vae-ft-mse`;
   - **Runs:** `<project_root>/logs/<run>` -> `logs/<run>`;
   - **The project:** `project_path` -> `.`.
2. **Shorten the OS string to the kernel version:**
   `Linux-6.8.0-45-generic-x86_64-with-glibc2.39` -> `Linux-6.8.0`.
3. **Drop the `provenance.git` block** (branch, commit, dirty).
4. **Grep the folder afterwards** to confirm that no username, hostname, absolute path,
   or commit remains. The paper README lists no commit either.

## Paper README

1. **Use exactly three sections:** `Introduction`, `Results`, and `Command`, as
   `reproduce/meanflow/README.md` does. Never add other sections, such as `Notes` or
   `Deviations`, or explanatory paragraphs.
2. **Give each run one row,** with a link to its subfolder as
   `[run](<exp>-<timestamp>)`. Each number is the best value of that metric over all
   trained checkpoints (epoch 0 is the base model). Thus the numbers in one row can
   come from different checkpoints. If a later code change affects the recorded runs,
   add a `**Note:**` line below the table.
3. **Show every number as the paper does,** with the same scale and decimal places:
   paper GenEval 0.95 -> ours 0.96, not 0.955. For a metric that the paper does not
   report, follow a paper that does (DiffusionNFT for ClipScore and HPSv2.1).
4. **Add a column for the training length or the GPU count only if necessary:**
   `#Iter` (for example, 1k) or `#GPU` (the run's `world_size`). Add it only if the
   rows of one task differ in it.
5. **Write `Introduction` as `Paper`, `Summary`, and `Implementation`.**
   `Implementation` is one sentence that names and links the repo that we follow,
   with no file names or file links. This is usually the official repo. If another
   repo is better, for example with a newer model or with results that we can compare
   with, follow it. Then the sentence also says why, and names and links the official
   repo. It ends in `, except for:`, and one line per difference from the repo that we
   follow comes after it. Each line says what differs and why.
   List only the differences that can change the results. Put the engineering-only
   differences, such as speed-ups and servers, in one `Infrastructure` line. Keep
   longer analyses out of the README.
6. **Trace the code in the files, not in the README.** Under `References:` in the
   module docstring, list each source file and the functions that it follows below the
   code link, as `algorithms/meanflow.py` does. In its header, each reproduce config
   and eval script names the paper table and the official configs that it follows.
   Shared code cites only the place where someone first wrote it. For example, it
   cites DiT for the DiT blocks, not every algorithm that copied them. General code,
   such as `sft_trainer.py`, cites nothing. Check every entry against the source code
   first.
7. **Keep `reproduce/README.md` brief:** it lists the papers, never results, in the
   same table as the Algorithms table of `README.md`. Its links are relative to
   `reproduce/`. Its `TODO` section lists the planned runs as task items, for example a
   rerun after a fix. When you record a run, remove its item.
