# CoFiTi migration validation

Validated locally on 2026-11-09 with the parent project virtual environment and
2 NVIDIA A100 80 GB GPUs. These are implementation checks and short training runs,
not a paper reproduction or a completed 500-epoch experiment.

Detailed JSON, logs, and commands are retained under
[`logs/cofiti_validation_20261109/`](logs/cofiti_validation_20261109/).
Generated logs/checkpoints are local artifacts and remain git-ignored.

## Numerical comparison

Used the same SD3.5 Medium transformer, a trained student LoRA checkpoint, encoded
prompt, noise seed, and precision on both sides. Each sampling case used one
prompt, group size 4, 256 px, and 3 Euler steps. Compared against the current parent
source's Euler and group exact-projection implementations.

| Initialization | Churn | Trajectory, velocity, endpoint, image | CUDA RNG | Projection target |
|---|---:|---|---|---|
| trajectory | 0 | Exact | Exact | Exact |
| trajectory | 0.04 | Exact | Exact | Exact |
| distribution | 0 | Exact | Exact | Exact |
| distribution | 0.04 | Exact | Exact | Exact |

Initial latent values also matched exactly. Without initial churn, the source's
standalone initial tensor is BF16, while the first slice of the saved stacked
trajectory is FP32; comparison accounts for this lossless dtype promotion.

On the same real-model forward graph, the source loss with an all-one activity
mask and CoFiTi's unmasked loss both equaled **0.4990110695362091**. The loss,
derivative with respect to predictions, model parameter gradients, and resulting
first AdamW update matched exactly (max absolute difference and relative L2 both
zero). Repeating the native backward also matched exactly in this check.

This isolates the formula and backward computation. It does not establish bitwise
identity between arbitrary independent training runs, GPU counts, batch layouts,
or dataloader schedules. Source activity filtering was deliberately bypassed in
the loss oracle: removing that filtering is an intentional behavior change.

## Automated checks

**66 checks passed.** The main temporary suite passed 65 checks; one local
processor check initially skipped because moving the test changed its resource
path. After correcting the temporary test's path, that check passed separately.

Coverage included:

- FP32/BF16 endpoint projection against the source, duplicate/rank-deficient
  endpoints, constant rewards and constant endpoints.
- Full MSE and gradients with zero, tiny and ordinary displacements.
- Euler step arithmetic, RNG state and trajectory at zero/nonzero churn.
- All three task configurations, removed-field rejection including Hydra `+`,
  and exact constructor/YAML field, section, order and comment alignment.
- Actual-rank continuous shuffling, duplicates, wraparound and private RNG resume.
- Epoch/step checkpoint triggers and deduplication, mid-epoch continuation,
  two-rank CPU DDP restore, GradScaler-skip/EMA behavior and JSONL timing.
- Missing/incompatible checkpoint signatures, invalid counters and missing
  student LoRA parameters are rejected.
- Completed evaluation metrics cannot bypass changed sampling signatures;
  missing or incompatible training configurations are rejected.
- PickScore floating-image preprocessing, Unicode tokenization, reward service
  adapters, plotting, and unchanged upstream reproduce configurations.

The temporary suite and comparison scripts were removed after validation as
requested. They are not dependencies of the remaining training/evaluation code.

## Real training and evaluation

All short training runs used 8 groups, group size 4, 256 px, 3 Euler steps and one
optimizer update per epoch: **32 images and 96 member/time slots per epoch**.
Defaults retained eta=1, u_scale=0.1, KL=0.004, churn=0.04, and rollout EMA.

| Run directory under `logs/` | Task / mode / GPUs | Outcome |
|---|---|---|
| `cofiti_validation_trajectory_single_20261109` | PickScore / trajectory / 1 | One update, then real checkpoint resume to epoch/step 2; EMA count 2 |
| `cofiti_validation_distribution_dual_20261109` | PickScore / distribution / 2 | Successful DDP update; EMA count 1 |
| `cofiti_validation_geneval_20261109` | GenEval / trajectory / 1 | Local service scored 32 images; successful update |
| `cofiti_validation_ocr_20261109` | PaddleOCR / distribution / 1 | Local service scored 32 images; successful update |

The OCR short-run reward was zero. This verifies service execution and the
unmasked zero-displacement training path, not OCR image quality. The policy MSE
remained nonzero (approximately 3.83e-6), as intended.

Every saved checkpoint contained exactly student LoRA, rollout EMA, optimizer,
scaler, runtime and train-state files. No evaluation EMA or partial `.tmp`
checkpoint remained. Runtime includes the private dataloader stream. JSONL logs
contain optimizer success, EMA updates, epoch/step times, elapsed time and ETA.

Native student evaluation loaded the first PickScore checkpoint, generated four
real test-prompt images, and scored them successfully (raw PickScore 17.6262 on
this tiny subset). The images, full sampling signature and metrics are under the
run's `eval/` directory.

## Commands used

Run from the NexusAlign directory; common training overrides:

```text
trainer.groups_per_epoch=8 trainer.epochs=1 algorithm.group_size=4
algorithm.resolution=256 algorithm.num_steps=3 trainer.log_hardware=false
```

```bash
CUDA_VISIBLE_DEVICES=0 EXP=pickscore NPROC_PER_NODE=1 MASTER_PORT=29621 \
  bash scripts/reproduce/cofiti_train.sh \
  exp_info=cofiti_validation_trajectory_single_20261109 timestamp=false \
  trainer.groups_per_epoch=8 trainer.epochs=1 algorithm.group_size=4 \
  algorithm.resolution=256 algorithm.num_steps=3 trainer.log_hardware=false

CUDA_VISIBLE_DEVICES=0,1 EXP=pickscore NPROC_PER_NODE=2 MASTER_PORT=29622 \
  bash scripts/reproduce/cofiti_train.sh \
  exp_info=cofiti_validation_distribution_dual_20261109 timestamp=false \
  trainer.groups_per_epoch=8 trainer.epochs=1 algorithm.group_size=4 \
  algorithm.resolution=256 algorithm.num_steps=3 trainer.log_hardware=false \
  algorithm.optimization_level=distribution

# Same small overrides as the single-rank run; raise the epoch limit to 2.
CUDA_VISIBLE_DEVICES=0 EXP=pickscore NPROC_PER_NODE=1 MASTER_PORT=29623 \
  bash scripts/reproduce/cofiti_train.sh \
  trainer.resume_path="$PWD/logs/cofiti_validation_trajectory_single_20261109/checkpoints/epoch_1-step_0-total_step_1" \
  trainer.groups_per_epoch=8 trainer.epochs=2 algorithm.group_size=4 \
  algorithm.resolution=256 algorithm.num_steps=3 trainer.log_hardware=false
```

GenEval used `EXP=geneval`, one GPU and the common overrides. OCR used `EXP=ocr`,
one GPU, the common overrides and `algorithm.optimization_level=distribution`.
Both launchers started and cleaned up their own local reward service.

```bash
CUDA_VISIBLE_DEVICES=1 EXP=pickscore NPROC_PER_NODE=1 MASTER_PORT=29626 \
  EXP_NAME=cofiti_validation_trajectory_single_20261109 \
  bash scripts/reproduce/cofiti_eval.sh \
  data.path="$PWD/logs/cofiti_validation_20261109/data" \
  "evaluator.ckpts=[$PWD/logs/cofiti_validation_trajectory_single_20261109/checkpoints/epoch_1-step_0-total_step_1]"
```

No dependency installation, FA3 installation, or full-length training was run.

## Independent evaluation alignment (2026-11-09)

The evaluation entrypoint now follows the source independent evaluation instead
of the training rollout. Defaults are native Diffusers Flow Euler, 40 steps,
512 px, CFG 1, T5 length 256, and no churn or training autocast. Student weights
remain the default, as in the source. Every image uses a CPU generator seeded by
`(global_seed + repeat_index) % 2**31`.

Using the same loaded SD3.5 student checkpoint and prompts, compared the new
`CoFiTi.generate()` path against the current source `eval.models.run_pipe()`.
The source oracle used a separate pipeline/scheduler object sharing the weights.

| Resolution | Steps | CFG | Image seeds | Model-input max absolute difference | uint8 image max difference |
|---|---|---|---|---|---|
| 256 | 3 | 1 | 6, 7 | 0 | 0 |
| 256 | 3 | 0.5 | 6, 6 | 0 | 0 |
| 256 | 3 | 2 | 6, 7 | 0 | 0 |
| 512 | 40 | 1 | 6 | 0 | 0 |

This checks the real generation path under controlled weights and batch shapes.
It does not establish bitwise identity across arbitrary GPU counts or batch
layouts. The native evaluator still pads rank shards before discarding padding
from metrics; this can change the final batch shape compared with the old runner.
PNG output and the NexusAlign experiment layout are retained.

Ten CPU checks covered prompt/repeat seed assignment under 1/2/3 ranks and several
batch layouts, source-compatible subset selection, PNG round trips, source pixel
rounding, unchanged non-CoFiTi rounding/RNG behavior, generator seed wrapping,
autocast disabling, and rejection of old or changed sampling signatures. Config
checks covered independent evaluation defaults and existing reproduce configs.

A real two-GPU native evaluation loaded the saved student, generated two prompts
with two repeats (four images), and scored PickScore successfully. It used 256 px,
three steps, T5 length 256, and seed 6. Raw PickScore was approximately 17.7858;
this tiny smoke result is not a quality benchmark. Config, images, signature,
metrics and summary figures are in the training run's
`eval/reward-cofiti-20261109_071735/` directory.

```bash
CUDA_VISIBLE_DEVICES=0,1 EXP=pickscore \
  EXP_NAME=cofiti_validation_trajectory_single_20261109 \
  NPROC_PER_NODE=2 MASTER_PORT=29641 \
  bash scripts/reproduce/cofiti_eval.sh \
  evaluator.new_run=true evaluator.num_prompts=2 evaluator.num_repeats=2 \
  evaluator.num_steps=3 evaluator.resolution=256 evaluator.global_seed=6 \
  "evaluator.ckpts=[$PWD/logs/cofiti_validation_trajectory_single_20261109/checkpoints/epoch_1-step_0-total_step_1]"
```

Detailed comparison/config results and logs are retained under
`logs/cofiti_eval_alignment_20261109/`. Temporary validation scripts were removed
after verification. No training run or dependency installation was needed.

## Multi-reward preset migration (2026-11-09)

Added the source `multi_reward_geneval_clip30` default score ensemble. Training
uses four independently calibrated/group-standardized/clipped components, then
applies their weights; reporting uses their fixed unweighted calibrated sum.
Other source ensemble modes are not enabled.

Validation artifacts are in `logs/cofiti_multi_reward_validation_20261109/`:

- Eight CPU checks passed, including exact agreement with the current source
  `_compute_training_and_rollout_rewards()` at group sizes 4 and 32, default and
  zero/negative weights, clipping without a final aggregate clip, metadata
  isolation, weighted-score overflow rejection, and source HPS preprocessing on
  square/rectangular/tiny images. These checks also composed all 17 existing
  upstream reproduce configurations.
- Five invalid configurations were rejected before loading any scorer: all-zero
  weights, nonfinite weights, a missing component weight, and the unsupported
  `rank` and `diffusionnft` ensemble modes.
- Real GPU scoring of four float images matched the source HPSv2 and CLIPScore
  `compute()` methods exactly: raw-score maximum absolute differences were zero,
  as was the CLIP `1/30` calibrated-score difference. This comparison shares the
  loaded weights to isolate preprocessing and scoring. The shared CLIP divisor
  of 100 remained numerically unchanged.
- GenEval's legitimate main score `-10` was retained in evaluation averaging;
  only declared per-tag accuracy sentinels were excluded. The same sentinel
  declarations are used in CoFiTi training logs.

A real two-A100 training run used 8 groups, group size 4, 256 px, 3 Euler steps,
and one epoch: 32 scored images and 96 member/time training slots. Its single
optimizer update succeeded, rollout EMA advanced to 1, and all six checkpoint
files were saved. The checkpoint signature includes the four component weights,
normalization parameters and component configurations. This is a short execution
check, not a quality benchmark or paper reproduction.

```bash
CUDA_VISIBLE_DEVICES=0,1 EXP=multi_reward_geneval_clip30 \
  NPROC_PER_NODE=2 MASTER_PORT=29651 \
  bash scripts/reproduce/cofiti_train.sh \
  exp_info=cofiti_validation_multi_reward_geneval_clip30_20261109 timestamp=false \
  trainer.groups_per_epoch=8 algorithm.group_size=4 \
  algorithm.resolution=256 algorithm.num_steps=3 \
  trainer.epochs=1 trainer.log_hardware=false
```

The run is retained at `logs/cofiti_validation_multi_reward_geneval_clip30_20261109/`.
Training report means were PickScore 18.7182, HPSv2 0.1242, CLIPScore 21.6422,
GenEval 0.0208, and calibrated report sum 1.5864. The raw component means must not
be summed without the preset's calibration factors.

Native single-GPU evaluation then loaded that checkpoint and scored two GenEval
test prompts at 256 px and three native Flow Euler steps. It saved both PNGs,
component/calibrated metrics, the fixed report sum, GenEval accuracies and summary
figures under `eval/reward-cofiti-20261109_074448/`. The fixed report mean was
1.41947; this subset is only an execution check.

```bash
CUDA_VISIBLE_DEVICES=0 EXP=multi_reward_geneval_clip30 \
  EXP_NAME=cofiti_validation_multi_reward_geneval_clip30_20261109 \
  NPROC_PER_NODE=1 MASTER_PORT=29652 \
  bash scripts/reproduce/cofiti_eval.sh evaluator.new_run=true \
  evaluator.num_prompts=2 evaluator.num_steps=3 evaluator.resolution=256
```

Both launchers started and cleaned up their own GenEval service. No HPS service,
dependency installation or full training experiment was needed. Temporary CPU/GPU
validation scripts and their new-module caches were removed; logs, numerical
results, images and checkpoints remain. The cleanup list and full results are in
`logs/cofiti_multi_reward_validation_20261109/summary.json`.

## Unified evaluation entrypoint: four-task verification (2026-11-09)

After adding pre-service validation to `scripts/reproduce/cofiti_eval.sh`, all four
tasks were evaluated again through that entrypoint. Each used its existing
`epoch_1-step_0-total_step_1` student checkpoint, one A100, two test prompts, one
image per prompt, 256 px and three native Flow Euler steps. CFG remained 1, T5
length 256 and image seed 0. Every evaluation used a fresh directory.

| EXP | Existing training run under logs/ | New evaluation directory | Result |
|---|---|---|---|
| pickscore | cofiti_validation_trajectory_single_20261109 | reward-cofiti-20261109_080851 | PickScore 16.5459 |
| geneval | cofiti_validation_geneval_20261109 | reward-cofiti-20261109_081053 | Partial score / accuracy / strict accuracy all 0 |
| ocr | cofiti_validation_ocr_20261109 | reward-cofiti-20261109_080919 | OCR 0 |
| multi_reward_geneval_clip30 | cofiti_validation_multi_reward_geneval_clip30_20261109 | reward-cofiti-20261109_081052 | Report sum 1.41947; all four components scored |

Each run loaded student weights and saved two valid 256x256 PNGs, `samples.json`,
`metrics.json`, `config_eval.yaml`, `eval_summary.json`, and step/epoch figures.
Zero GenEval/OCR scores are valid outcomes on this tiny subset; these checks do
not establish image quality or full-test-set performance.

Nine launcher checks covered missing Python, missing experiment/config before
service startup, effective `log_dir`/`exp_info` overrides (including spaces),
argument order, all four data/reward/service routes, cleanup of owned services,
preservation of pre-existing services, native `--cfg job --resolve` inspection,
and unchanged evaluation defaults. Launcher control-flow tests used isolated
service/worker fixtures with the real Hydra preflight; the GPU runs used the real
scorers. All three test service ports (18185/18186/18187) were closed after the
real evaluations and both GPUs returned to 16 MiB usage.

Output checks also verify that matching caches are reusable, changed sampling
signatures are rejected even when metrics exist, and evaluation does not emit
training group-zscore metrics. The multi-reward report is checked against
`PickScore/26 + HPSv2 + CLIPScore/30 + GenEval` with absolute tolerance 1e-6.

Reproduction commands from the NexusAlign directory:

```bash
EVAL_ARGS=(evaluator.new_run=true evaluator.num_prompts=2 evaluator.num_repeats=1
           evaluator.num_steps=3 evaluator.resolution=256)
CHECKPOINT_NAME=epoch_1-step_0-total_step_1

EVAL_RUN=cofiti_validation_trajectory_single_20261109
CUDA_VISIBLE_DEVICES=0 EXP=pickscore EXP_NAME="$EVAL_RUN" \
  NPROC_PER_NODE=1 MASTER_PORT=29661 bash scripts/reproduce/cofiti_eval.sh \
  "${EVAL_ARGS[@]}" "evaluator.ckpts=[$PWD/logs/$EVAL_RUN/checkpoints/$CHECKPOINT_NAME]"

EVAL_RUN=cofiti_validation_ocr_20261109
CUDA_VISIBLE_DEVICES=1 EXP=ocr EXP_NAME="$EVAL_RUN" \
  NPROC_PER_NODE=1 MASTER_PORT=29662 REWARD_SERVER_PORT=18187 \
  bash scripts/reproduce/cofiti_eval.sh "${EVAL_ARGS[@]}" \
  "evaluator.ckpts=[$PWD/logs/$EVAL_RUN/checkpoints/$CHECKPOINT_NAME]"

EVAL_RUN=cofiti_validation_geneval_20261109
CUDA_VISIBLE_DEVICES=0 EXP=geneval EXP_NAME="$EVAL_RUN" \
  NPROC_PER_NODE=1 MASTER_PORT=29663 REWARD_SERVER_PORT=18185 \
  bash scripts/reproduce/cofiti_eval.sh "${EVAL_ARGS[@]}" \
  "evaluator.ckpts=[$PWD/logs/$EVAL_RUN/checkpoints/$CHECKPOINT_NAME]"

EVAL_RUN=cofiti_validation_multi_reward_geneval_clip30_20261109
CUDA_VISIBLE_DEVICES=1 EXP=multi_reward_geneval_clip30 EXP_NAME="$EVAL_RUN" \
  NPROC_PER_NODE=1 MASTER_PORT=29664 REWARD_SERVER_PORT=18186 \
  bash scripts/reproduce/cofiti_eval.sh "${EVAL_ARGS[@]}" \
  "evaluator.ckpts=[$PWD/logs/$EVAL_RUN/checkpoints/$CHECKPOINT_NAME]"
```

Console logs, launcher/output checks and cleanup records are retained under
`logs/cofiti_eval_four_tasks_20261109/`. Temporary verification code was removed;
images, metrics and existing checkpoints were preserved. No packages were
installed, no training was started and no full test split was evaluated.

## Cleanup and optional SA-Solver (2026-11-09)

The training sampler now accepts `algorithm.solver=euler|sa`; Euler remains the
default. Independent evaluation remains native Flow Euler. The SA implementation
contains only the fixed AWM data-prediction 2/2 PEC path needed by CoFiTi, with
constant churn, actual predictor inputs/times, and source RNG ordering.

Removed unused trainer attributes, dataset tag/alias/count fields, stored
projection mean/Gram fields, duplicate outer projection validation, training
sample overrides/fallback prompt encoding, and the optional no-trajectory path.
Full metadata, IDs, fingerprints, all-sample MSE, projection arithmetic, CFG,
rollout EMA, checkpoint state and shared upstream behavior remain intact.
The reward launcher now reads only the documented `COFITI_RESOURCE_ROOT` variable.

### Numerical and configuration checks

- CPU and A100 each passed 18 SA kernel comparisons against local
  `fdfo/sa_solver.py`: steps 2/3/10, churn 0/0.0206/0.04, FP32/BF16 initial
  tensors. Endpoint, every predictor input, velocity, model timestep and final
  RNG state matched exactly (maximum absolute error 0).
- CPU and A100 each passed 9 Euler kernel regressions and 8 complete lightweight
  sampler comparisons against the pre-cleanup CoFiTi code. Both initialization
  modes, churn 0/0.04 and CFG 1/2 were covered. Euler outputs, targets, losses and
  gradients matched exactly; SA integration returned the expected call times.
- Ten source projection comparisons covered ordinary, duplicate, rank-deficient,
  identical endpoints and zero advantages, with ridge scale 0/0.001. Maximum
  absolute correction error was 0. Invalid parameters and nonfinite inputs fail.
- PickScore, GenEval and OCR real datasets retain their fingerprints, metadata
  isolation and exact dataloader restoration. Removed fields have no consumers.
- All eight task/solver combinations and 17 upstream reproduce configs composed
  successfully. Algorithm YAML field order matches the constructor. Invalid
  solver, SA with one step, removed fields and unrepresentable SA churn reject.
- Existing Euler checkpoint algorithm signatures remain unchanged. A solver
  change is detected as `algorithm.solver` by resume validation.
- Python syntax, launcher shell syntax and Git whitespace checks passed.

These are controlled arithmetic comparisons, not a claim that independent GPU
training runs are bitwise identical. The old public SA wrapper only supported
shared initialization; distribution+SA is an extension here. CoFiTi's existing
zero-churn initialization remains unchanged, while SA's internal unused initial
and final RNG draws are preserved.

### Real small runs

All training checks used local SD3.5 Medium, PickScore, group size 4, 8 groups,
256 px, SA 3 steps and one optimizer update per epoch: 32 images and 96 slots.

| Run directory under `logs/` | Check | Result |
|---|---|---|
| `cofiti_sa_trajectory_cleanup_20261109` | Single A100, trajectory, epoch 1 | Update/EMA 1; loss 1.2846106887 |
| Same run | Resume epoch 1 checkpoint through epoch 2 | Update/EMA 2; runtime, optimizer and scaler restored |
| `cofiti_sa_distribution_cleanup_20261109` | Two A100s, distribution, epoch 1 | Update/EMA 1; loss 0.2629563933 |
| Distribution run, `eval/reward-cofiti-20261109_082827/` | Student evaluation, 2 prompts, 256 px, 3 native Euler steps | 2 valid PNGs; raw PickScore 16.6460795403; metrics/signature/summary saved |

Each checkpoint has exactly the six expected files, including rollout EMA;
there is no evaluation EMA or unfinished temporary checkpoint. Evaluation
inherits the SA-trained weights but records `diffusers_flow_euler`, churn 0 and
T5 length 256 in its sampling signature. These scores only establish execution.

Commands run from the NexusAlign directory:

```bash
SMALL=(trainer.groups_per_epoch=8 algorithm.group_size=4 \
  algorithm.resolution=256 algorithm.num_steps=3 algorithm.solver=sa \
  trainer.log_hardware=false)

CUDA_VISIBLE_DEVICES=0 EXP=pickscore NPROC_PER_NODE=1 MASTER_PORT=29651 \
  bash scripts/reproduce/cofiti_train.sh "${SMALL[@]}" trainer.epochs=1 \
  exp_info=cofiti_sa_trajectory_cleanup_20261109 timestamp=false

CUDA_VISIBLE_DEVICES=0,1 EXP=pickscore NPROC_PER_NODE=2 MASTER_PORT=29652 \
  bash scripts/reproduce/cofiti_train.sh "${SMALL[@]}" trainer.epochs=1 \
  algorithm.optimization_level=distribution \
  exp_info=cofiti_sa_distribution_cleanup_20261109 timestamp=false

CUDA_VISIBLE_DEVICES=0 EXP=pickscore NPROC_PER_NODE=1 MASTER_PORT=29653 \
  bash scripts/reproduce/cofiti_train.sh "${SMALL[@]}" trainer.epochs=2 \
  "trainer.resume_path=$PWD/logs/cofiti_sa_trajectory_cleanup_20261109/checkpoints/epoch_1-step_0-total_step_1"

CUDA_VISIBLE_DEVICES=1 EXP=pickscore NPROC_PER_NODE=1 MASTER_PORT=29654 \
  EXP_NAME=cofiti_sa_distribution_cleanup_20261109 \
  bash scripts/reproduce/cofiti_eval.sh \
  "evaluator.ckpts=[$PWD/logs/cofiti_sa_distribution_cleanup_20261109/checkpoints/epoch_1-step_0-total_step_1]" \
  evaluator.new_run=true evaluator.num_prompts=2 evaluator.num_repeats=1 \
  evaluator.num_steps=3 evaluator.resolution=256
```

Numerical/configuration reports and console logs are retained in
`logs/cofiti_cleanup_sa_20261109/`; training checkpoints and evaluation images
remain in the run directories above. Temporary checks, source snapshots and
their bytecode were removed, along with the obsolete Euler-module bytecode.
No verification scripts were added to the repository. No dependencies were
installed and no full training experiment was started.
