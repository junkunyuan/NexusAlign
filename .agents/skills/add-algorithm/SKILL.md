---
name: add-algorithm
description: Add a new algorithm to NexusAlign from a GitHub link. Use it when the user asks to add, port, or reproduce a paper's method. It covers how to read the paper and the official code, connect the algorithm to the registry and the Hydra configs, write the launch scripts, and do a smoke run.
---

# Add an Algorithm

Follow these steps in order.

1. **Read the paper first.** Find the paper that the repo README links to, and read it
   carefully. Write down the core method: the training objective, its equations, the
   sampling procedure, and the hyperparameters of the main experiments.
2. **Read the code against the paper.** Read the official code for training and
   sampling, and map each part to the equations. Note where the code is different from
   the paper, for example in defaults, tricks, and numerical details. For reproduction,
   the code is the ground truth. This is usually the official code. If another
   implementation is better, for example with a newer model or with results that we
   can compare with, follow it instead.
3. **Fit it into the existing structure.** Add it in the same way as the existing
   algorithms. Change the repo layout only if necessary:
   - **`algorithms/<name>.py`:** subclass `BaseAlgorithm`, and implement `forward` and
     `sample`. In the module docstring, cite the paper, the code, and the source files
     and functions that it follows. Register it in `algorithms/__init__.py`.
   - **`datasets/`, `models/`, `evaluators/`:** reuse the existing components. Add a
     new one only if the algorithm truly needs it, and register it the same way.
   - **`configs/algorithm/<name>.yaml`:** one commented field per hyperparameter,
     grouped by section, with references to the paper equations where they help. The
     defaults follow the paper.
   - **`configs/reproduce/<name>/<exp>.yaml`, `scripts/reproduce/<name>_train.sh`, and
     `<name>_eval.sh`:** one pinned config per paper experiment, like the existing
     ones.
   - **`reproduce/<name>/README.md` and a line in `reproduce/README.md`:** fill them in
     after a run is done. You can write the `Implementation` part of `Introduction`
     before the run.
4. **Keep the code faithful and minimal.** Match the paper and the code that you
   follow exactly, and keep the diff small. Comment only what is not obvious, and
   follow the existing code style. Do not change the shared defaults that other
   reproduce configs depend on (see the Workflow in `AGENTS.md`).
5. **Verify.** Run a short smoke training and sampling before you report that you
   added the algorithm. Trace each part and config arg back to the paper or the
   official code. Record the sources and each deviation in the `Implementation` part
   of the paper README's `Introduction`.

**Conventions:** these steps rely on [common-workflow](../../rules/common-workflow.md),
[configs](../../rules/configs.md), [reproduce](../../rules/reproduce.md), and
[code-style](../../rules/code-style.md).
