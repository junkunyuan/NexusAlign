# AGENTS.md

Guidance for AI coding agents that work in this repository.

## Project overview

NexusAlign is a unified research framework to reproduce and extend foundation model
algorithms.

- **`src/nexus_align/`:** the library: `datasets/`, `models/`, `algorithms/`,
  `trainers/`, `evaluators/`, `rewards/`, `eval/`, `engine/`, `configs/`, `cli/`,
  `utils/`.
- **`scripts/reproduce/`:** launch scripts for reproduction runs, for example
  `meanflow_train.sh` and `meanflow_eval.sh`.
- **`scripts/tools/`:** development tools, for example `kill_jobs.sh` and
  `reward_server.sh`.
- **`reproduce/<algorithm>/`:** the committed record of one reproduced paper: a
  `README.md` and one `<exp>-<timestamp>/` subfolder per run.
- **`logs/` and `data_and_model/`:** run outputs, and datasets and checkpoints. Git
  ignores both.

## Common commands

```bash
# Train (EXP: b2, m2, l2, xl2, xl2_plus)
EXP=b2 bash scripts/reproduce/meanflow_train.sh

# Evaluate a run under logs/
EXP_NAME=<log_dir_name> bash scripts/reproduce/meanflow_eval.sh

# Kill the jobs of this repo
bash scripts/tools/kill_jobs.sh
```

**First run:** the scripts create `.venv` with `uv`, download the missing assets into
`data_and_model/` with `src/nexus_align/utils/download.py`, and set `PYTHONPATH=src`.

## Workflow

Follow [common-workflow](.agents/rules/common-workflow.md): understand the request
and ask early, explore first, keep the code simple, follow the existing style, keep
the comments current, and ask before you change a `common-*.md` rule. In this repo,
also:

- **Reproducibility:** When you add code or features, make sure that you can still
  reproduce every method that `reproduce/` records. It is easy to break a shared
  default. For example, a change to a default in `configs/trainer/sft.yaml` can silently
  change what `configs/reproduce/meanflow/*.yaml` resolves to. If you change a shared
  default, pin the reproduce configs to the old value.
- **Hardware:** Never assume a GPU type, GPU count, or memory size. Detect them at
  runtime, for example with `torch.cuda.get_device_name()`, or read them from the
  launch environment or config. Keep each per-GPU number, such as a peak TFLOPS, in a
  lookup table with the detected device as the key.
- **Download:** Keep `src/nexus_align/utils/download.py` pure: it only downloads assets
  from Hugging Face or URLs into `data_and_model/`. Do not add cluster or storage logic
  to it.

## Rules and skills

- **[`.agents/rules/`](.agents/rules/):** conventions. Each rule applies to the files
  that its `paths:` frontmatter names. Some tools load the rules automatically. In all
  tools, read the rules that cover a path before you edit it. The `common-*.md` rules
  do not depend on this repo, so other projects can use them unchanged. The other
  rules add to them.
- **[`.agents/skills/`](.agents/skills/):** step-by-step guides for frequent tasks,
  such as how to add an algorithm.

**Index:** [`.agents/README.md`](.agents/README.md) lists every rule and skill.
