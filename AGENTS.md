# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

NexusAlign is a unified research framework for tracking, reproducing, and extending foundation
model algorithms.

- `src/nexus_align/` — the library: `algorithms/`, `models/`, `datasets/`, `trainers/`,
  `rewards/` (reward models and their servers), `evaluators/`, `eval/` (FID/IS metrics and
  eval summaries), `engine/` (distributed, checkpoint, logger), `configs/` (Hydra YAML),
  `cli/` (training and evaluation entry point), `utils/` (e.g. `download.py`).
- `scripts/reproduce/` — launch scripts for reproduction runs, e.g. `meanflow_train.sh`,
  `meanflow_eval.sh`.
- `scripts/tools/` — dev tooling, e.g. `kill_jobs.sh` and `reward_server.sh`.
- `reproduce/<algorithm>/` — committed record of one reproduced paper: a `README.md` and
  one `<exp>-<timestamp>/` subfolder per run.
- `logs/` — run outputs (ignored by git). `data_and_model/` — datasets and checkpoints
  (ignored).

## Common commands

```bash
# Train (EXP: b2, m2, l2, xl2, xl2_plus)
EXP=b2 bash scripts/reproduce/meanflow_train.sh

# Evaluate a run under logs/
EXP_NAME=<log_dir_name> bash scripts/reproduce/meanflow_eval.sh

# Kill running jobs
bash scripts/tools/kill_jobs.sh
```

Scripts create `.venv` with `uv` on first run, download missing assets into
`data_and_model/` with `src/nexus_align/utils/download.py`, and set `PYTHONPATH=src`.

## Workflow

- Explore the relevant files and understand the existing code before making changes.
- Keep changes as simple as possible. Avoid unnecessary changes and unnecessary comments.
- Strictly follow the project's existing code style when writing new code and comments.
- When adding code or features, check that every method already recorded under
  `reproduce/` can still be reproduced. Shared defaults are easy to break: for example,
  changing a default in `configs/trainer/sft.yaml` can silently change what
  `configs/reproduce/meanflow/*.yaml` resolves to. If a change touches a shared default,
  keep the reproduce configs pinned to the old value.
- Keep `src/nexus_align/utils/download.py` pure: it only downloads assets from Hugging Face
  or URLs into `data_and_model/`. Do not add cluster or storage logic to it.

## Rules and skills

- [`.agents/rules/`](.agents/rules/) — conventions, each scoped to the files named in
  its `paths:` frontmatter. Some tools load them automatically; in any tool, read the
  rules that cover a path before editing it.
- [`.agents/skills/`](.agents/skills/) — step-by-step guides for recurring tasks, such
  as adding an algorithm.

See [`.agents/README.md`](.agents/README.md) for the index.
