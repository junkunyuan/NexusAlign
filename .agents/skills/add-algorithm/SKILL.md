---
name: add-algorithm
description: Add a new algorithm to NexusAlign from a GitHub link. Use when asked to add, port, or reproduce a paper's method; covers reading the paper and official code, wiring the algorithm into the registry and Hydra configs, launch scripts, and a smoke run.
---

# Add an Algorithm

Follow these steps in order.

1. **Read the paper first.** Find the paper linked from the repo README, read it carefully,
   and write down the core method: the training objective, its equations, the sampling
   procedure, and the hyperparameters used in the main experiments.
2. **Read the code against the paper.** Go through the official training and sampling code
   and map each part to the equations. Note where the code deviates from the paper
   (defaults, tricks, numerical details); the code is the ground truth for reproduction.
3. **Fit it into the existing structure.** Follow how the existing algorithms are wired
   in and do not change the repo layout unless it is necessary:
   - `algorithms/<name>.py`: subclass `BaseAlgorithm`, implement `forward` and `sample`,
     and cite the paper and code in the module docstring. Register it in
     `algorithms/__init__.py`.
   - `models/`, `datasets/`, `evaluators/`: reuse existing components; add a new one only if
     the algorithm truly needs it, and register it the same way.
   - `configs/algorithm/<name>.yaml`: one commented field per hyperparameter, grouped by
     section, with paper equation references where helpful. Defaults follow the paper.
   - `configs/reproduce/<name>/<exp>.yaml` and `scripts/reproduce/<name>_train.sh` /
     `<name>_eval.sh`: one pinned config per paper experiment, mirroring the existing ones.
   - `reproduce/<name>/README.md` and a line in `reproduce/README.md`: fill in once a run
     is done.
4. **Keep the code faithful and minimal.** Match the paper and official code exactly, keep
   the diff small, comment only what is non-obvious, and follow the existing code style.
   Do not touch shared defaults that other reproduce configs depend on (see the Workflow
   in `AGENTS.md`).
5. **Verify.** Run a short smoke training and sampling run before reporting the algorithm
   as added. Summarize any deviation from the paper or official code in the reply, not in
   the paper README.

The conventions these steps rely on are in [configs](../../rules/configs.md),
[reproduce](../../rules/reproduce.md), and [code-style](../../rules/code-style.md).
