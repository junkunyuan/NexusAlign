# CoFiTi

CoFiTi is a local extension of FDFO, implemented through NexusAlign's native
algorithm, model, trainer, dataset, reward, and evaluator interfaces. It uses one
nonlinear preference projection per prompt group, endpoint-copy velocity targets,
and an all-sample MSE. It is not a reproduction claim for the original FDFO paper.

The source is the parent project's `finite-difference-flow-optimization/fdfo/`
implementation, especially `exact_projection.py`, `sampling.py`, `sa_solver.py`,
and `prompts.py`.
See [FDFO](https://github.com/NVlabs/finite-difference-flow-optimization), its
[paper](https://arxiv.org/abs/2603.12893), and [licenses](licenses/FDFO-LICENSE.txt).
The original source files in the parent project are unchanged.

## Method

For K endpoints, let D be their centered, flattened matrix, n the latent dimension,
and a the group reward advantages. The solver fits one correction u:

```text
q = softmax(eta * a)
p(u) = softmax(D @ u / n)
lambda = ridge_scale * mean(diag(D @ D.T / n)) + ridge_eps
objective = K * n * KL(q || p(u)) + lambda * ||u||² / 2
```

The solver uses FP64 without autograd or autocast, numerical-rank handling, and a
fixed number of Newton updates (default 5). Each update uses the source Armijo
line search with at most 50 trials. There is no approximate guidance fallback or
convergence-based early stopping. Nonfinite values and failed solves raise errors.

```text
displacement = (u_fp64 * u_scale).float()
target = rollout_velocity - displacement
loss = mean_over_all_samples(MSE(student_velocity, target))
       + kl_weight * mean_squared_base_velocity_difference
```

The correction is broadcast to every group member and saved timestep. `eta` is
used inside the projection, with no additional multiplication in the loss.
Every sample contributes, including zero or arbitrarily small corrections. A
zero correction still trains against the rollout velocity. The base-velocity
regularizer is separate from the projection's KL objective.

Loss, target construction, and projection type are fixed. There are no exposed
loss/gradient/estimator selectors, activity thresholds or masks, subgroups,
time-weight modes, first/second-order guidance, or state/query projection.
Unknown algorithm/trainer configuration fields are rejected, including Hydra `+`
injection of removed options.

## Defaults

| Setting | Default |
|---|---|
| Solver / model calls | Euler / 10 |
| Optimization level | `trajectory` |
| Resolution / CFG / churn | 512 / 1.0 / 0.04 |
| Group size / groups per epoch | 24 / 48 |
| `eta` / `u_scale` | 1.0 / 0.1 |
| Projection iterations | 5 |
| `ridge_scale` / `ridge_eps` | 0.001 / 1e-6 |
| Training `kl_weight` | 0.004 |
| Epochs / optimizer updates per epoch | 500 / 1 |
| AdamW LR / betas | 3e-4 / (0.9, 0.999) |
| AdamW weight decay / epsilon | 1e-4 / 1e-8 |
| Gradient clipping / seed | 1.0 / 6 |
| LoRA rank / alpha | 32 / 64 |
| Forward batch ceiling | 28 |
| Dtypes | BF16 base, FP32 LoRA, FP16 autocast + GradScaler |
| Parallelism | DDP |

`eta=null` explicitly selects `1/group_size`. The normal default is 1.0.
The default epoch generates 1,152 images and trains on 11,520 member/time slots,
accumulated into one optimizer update. Prompt embeddings are deduplicated and
text encoders are offloaded after encoding.

Training and rollout always use FP16 autocast with GradScaler on CUDA;
`algorithm.autocast_dtype` is not a configurable field and explicit overrides
are rejected. Frozen weights remain BF16 and trainable LoRA remains FP32.
The fixed FP16 precision stays in checkpoint semantic metadata, so existing
FP16 CoFiTi checkpoints retain the same resume signature. Evaluation ignores
this historical training field when loading saved configuration and uses its
independent BF16 pipeline described below.

`optimization_level` only changes initial noise:

- `trajectory`: one shared base noise per group, followed by independent initial
  churn perturbations for its members.
- `distribution`: independent initial noise per member, without initial churn.

With the default Euler solver, both modes retain intermediate churn when
`churn_weight>0`; the terminal step adds no noise. With `distribution` and
`churn_weight=0`, sampling is ordinary Euler from independent noise.
With `trajectory` and zero churn, group members
share the same trajectory; their zero projection is still included in MSE.

`algorithm.solver` accepts `euler` (default) or `sa` (at least two steps).
SA uses the source AWM data-prediction 2/2 PEC solver, with its endpoint-adjusted
time grid and fresh-noise magnitude matched to the original scheduler's churn.
Both solvers use the same `optimization_level` initialization described above.
SA records actual predictor inputs and call times for training. Its unused
initial/final noise draws are retained, including at zero churn, for source RNG
parity. A churn level that cannot be represented on the SA grid raises an error.
These predictor/corrector orders belong to the sampling solver, not guidance.
Independent evaluation remains native Flow Euler regardless of the training solver.
Resume requires the same solver as the checkpoint.

```bash
EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/cofiti_train.sh \
  algorithm.solver=sa
```

## Data and rewards

The default resource root is the parent project. `COFITI_RESOURCE_ROOT` can
relocate data and models; the launch scripts always use the parent project venv.
The SD3.5 model is `models/stabilityai/stable-diffusion-3.5-medium` under that root.

| Task | Files under `data/` |
|---|---|
| PickScore | `pickscore_train.txt`, `pickscore_test.txt` |
| GenEval | `geneval/train_metadata.jsonl`, `geneval/test_metadata.jsonl` |
| OCR | `ocr/train.txt`, `ocr/test.txt` |

Records retain duplicates, ordering, full metadata, identities, and content
fingerprints. Metadata is copied independently for each member. Each actual rank
uses its own continuous NumPy stream seeded with `seed + rank`; exhausting the
dataset reshuffles it, while epoch boundaries do not reset it. Ranks may select
the same prompt. There is no virtual eight-rank reference schedule or associated
divisibility requirement. Groups must still divide across actual ranks.

PickScore uses the source floating-image preprocessing and `/26` scaling. Training
then standardizes within the full group with population standard deviation plus
1e-4 and clips to [-5, 5]. GenEval and OCR use the native HTTP reward clients;
local OCR is PaddleOCR. The launchers reuse ready services or start and clean up
only their own service processes. They do not install dependencies or download
models.

`EXP=multi_reward_geneval_clip30` adds the source four-reward preset on the GenEval
dataset. It calibrates raw PickScore by `1/26`, HPSv2.1 by `1`, raw CLIP-L logits
by `1/30`, and GenEval partial score by `1`. Each calibrated component is separately
group-standardized (population std + 1e-4) and clipped to [-5, 5], then summed with
`reward.weights.{pickscore,hpsv2,clipscore,geneval}` (all 1 by default). There is no
final aggregate normalization or clipping. The single-PickScore normalization
switch is disabled by this task config. Other source ensemble modes are not added.

The headline report is always the unweighted calibrated sum, including components
whose training weight is zero, matching the source preset's reporting override.
Component metrics are raw scores, `*_calibrated` are the scaled values, and
`*_group_zscore` plus `training_reward` describe the training advantages. In this
preset, `reward_pickscore` is raw, unlike the scaled single-PickScore task metric.
Only GenEval's per-tag accuracy sentinels are excluded from metric aggregation;
legal negative main scores, including -10, remain included.

HPSv2.1 uses the local `xswu/HPSv2/HPS_v2.1_compressed.pt` checkpoint with FP32
float-pixel preprocessing and BF16 autocast. CLIPScore uses the local
`openai/clip-vit-large-patch14` model and FP32 tensor preprocessing. No HPS server
is needed. See [the Chinese configuration guide](COFITI_CONFIG_ZH.md#7-multi_reward_geneval_clip30四项奖励组合)
for formulas, metric meanings, and weight overrides.

## Training and evaluation

Run from the NexusAlign repository directory. Set `EXP` to `pickscore`, `geneval`,
`ocr`, or `multi_reward_geneval_clip30`. The default names are `cofiti_<task>` with
a timestamp.

```bash
EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/cofiti_train.sh

EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/cofiti_train.sh \
  algorithm.optimization_level=distribution

EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/cofiti_train.sh \
  algorithm.optimization_level=distribution algorithm.churn_weight=0

EXP=pickscore EXP_NAME=<training-directory-name> \
  bash scripts/reproduce/cofiti_eval.sh

EXP=multi_reward_geneval_clip30 NPROC_PER_NODE=2 \
  bash scripts/reproduce/cofiti_train.sh

EXP=multi_reward_geneval_clip30 EXP_NAME=<training-directory-name> \
  bash scripts/reproduce/cofiti_eval.sh evaluator.new_run=true
```

Evaluation uses the student, matching the source FDFO checkpoint's default adapter.
It inherits the saved model configuration, then uses the source independent
`eval/calculate_metrics.py` / `eval/models.py` generation path: a native Diffusers
SD3 pipeline with Flow Euler, BF16 frozen weights, FP32 LoRA, and no autocast or
churn. The pipeline shares the loaded weights and has its own scheduler.

| Evaluation setting | Default |
|---|---|
| `evaluator.num_steps` | 40 |
| `evaluator.resolution` | 512 |
| `evaluator.guidance_scale` | 1.0 |
| `evaluator.max_sequence_length` | 256 T5 tokens |
| `evaluator.global_seed` | 0, the source `image_seed` |
| `evaluator.num_repeats` | 1 image per prompt |
| `evaluator.num_prompts` / `evaluator.prompt_seed` | null (full split) / 0 |
| `evaluator.per_proc_batch_size` | 4 |

These values are independent of the training sampler. Each image has its own CPU
generator seeded with `(global_seed + repeat_index) % 2**31`; repeat indices start
at zero. All prompts therefore share the same initial noise for a given repeat,
as in the source. Rank and batch layout do not change that noise assignment,
although GPU batch shapes can still affect numerical results. If `num_prompts`
is set, prompts and metadata are shuffled together with NumPy `prompt_seed` before
selecting the subset. Otherwise their original order is preserved.

The source's `(images * 255 + 0.5).clamp(0, 255).to(uint8)` conversion is preserved
in the original image dtype. Those pixels are scored and saved as lossless PNGs.
For multiple repeats, image slot `i` belongs to prompt `i // num_repeats` and
repeat `i % num_repeats`.

```bash
EXP=pickscore EXP_NAME=<training-directory-name> \
  bash scripts/reproduce/cofiti_eval.sh evaluator.new_run=true

# Optional subset and repeated samples, using the source's seed convention.
EXP=pickscore EXP_NAME=<training-directory-name> \
  bash scripts/reproduce/cofiti_eval.sh evaluator.new_run=true \
  evaluator.num_prompts=32 evaluator.num_repeats=2 evaluator.global_seed=6
```

`evaluator.ckpts=[/path/to/ckpt]` limits the checkpoint sweep. Sampling identity
version 2 includes the model, dataset fingerprint, prompt selection, repeats,
native sampler, resolution, steps, CFG, T5 length, seed policy, precision, pixel
conversion, and sampling layout. Training churn and optimization level no longer
control evaluation. Even completed metrics must match this identity. Results from
the previous training-rollout evaluation path require `evaluator.new_run=true`;
they are never reused as native Flow Euler results. Old Group-FDFO checkpoints
are still not interpreted as CoFiTi checkpoints.

## Checkpoints and resume

```yaml
trainer:
  save_ckpt_every_n_epochs: 1
  save_ckpt_every_n_steps: 1
  save_ckpt_keep_last_n_epochs: null
  save_ckpt_keep_last_n_steps: null
  resume_path: null
```

Intervals are independent of training length; null disables the corresponding
trigger. Coincident epoch/step triggers write once. Global steps count optimizer
step attempts, not microbatches; EMA counts only successful updates. Both retention
limits default to null, retaining all checkpoints. Rank 0 writes a temporary
directory and publishes it after all components are saved.

Each default checkpoint contains:

| File | Contents |
|---|---|
| `model/state.pt` | Student LoRA weights |
| `train_ema/state.pt` | Rollout EMA LoRA and successful-update count |
| `optimizer/state.pt` | AdamW state and parameter groups |
| `scaler/state.pt` | GradScaler state |
| `runtime/state.pt` | Per-rank RNG, private dataloader stream, dataset identity, epoch buffer |
| `train_state.json` | Method/format, progress, world size, semantic resume signature |

There is no evaluation EMA or duplicate frozen base/VAE/text-encoder checkpoint.
Disabling rollout EMA omits `train_ema/state.pt`. Default rollout EMA decay is
`min(0.01 * successful_updates, 0.9)`; skipped updates do not advance it.

```bash
EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/cofiti_train.sh \
  trainer.resume_path=/absolute/path/to/checkpoint
```

Resume requires the same world size, dataset, model architecture, and training
semantics. Supply the original overrides for a nondefault run. Epoch limits,
logging and saving intervals may change; optimizer hyperparameters are restored
from the checkpoint. The original experiment directory is reused. Mid-epoch
resume continues the saved buffer without resampling and restores the private
prompt stream as well as process RNG. There is no stage-transition mode, EMA
warmup restart, or permissive legacy checkpoint fallback.

## Outputs and code review

Outputs live under `logs/<experiment>-<timestamp>/`: `config_train.yaml`,
`results.jsonl`, `checkpoints/`, `figures/`, and `eval/`. JSONL records include
loss/reward metrics, epoch/step average times, process elapsed time, and ETA.
Elapsed time is per process lifetime, not accumulated across restarts.

- `scripts/reproduce/cofiti_train.sh`: composes task defaults, selects the parent
  interpreter and local sources, owns reward-service lifecycle, and launches DDP.
- `scripts/reproduce/cofiti_eval.sh`: locates the training experiment, selects
  rewards, and launches native student evaluation.
- `scripts/tools/cofiti_reward_server.sh`: serves/checks/waits for local GenEval or
  PaddleOCR through the existing native service factories.
- `src/nexus_align/algorithms/cofiti.py`: `CoFiTi.build_targets`, `loss_from_predictions`, `sample`,
  and `generate` implement the fixed objective, Euler/SA training rollouts, and
  dispatch to independent native Flow Euler evaluation.
- `src/nexus_align/algorithms/components/cofiti_projection.py`: prepares group geometry and solves
  the regularized preference projection with fixed Newton updates.
- `src/nexus_align/algorithms/components/cofiti_sampling.py`: `sample_euler` and
  `sample_sa` record model inputs/velocities and execute their source-compatible
  updates; SA retains the fixed AWM 2/2 PEC schedule and noise draw order.
- `src/nexus_align/datasets/cofiti.py`: `CoFiTiPromptDataset`, `PromptBatchPlan`, and
  `PromptDataLoader` read records and preserve rank-local stream state.
- `src/nexus_align/trainers/cofiti_trainer.py`: `CoFiTiTrainer`, `RolloutEMA`, and `RolloutRuntime`
  handle sampling, accumulation, successful-update EMA, checkpoints and resume.

- `src/nexus_align/models/cofiti_sd3.py`: `CoFiTiSD3Model` loads the local frozen
  BF16 pipeline, prepares FP32 LoRA/DDP, and exposes prompt encoding and decoding;
  `generate_evaluation` shares those weights with a native SD3 pipeline and uses
  per-image CPU generators without training autocast.
- `src/nexus_align/engine/setup.py`: `prepare_env` resolves the native experiment
  and resume directories; `inherit_train_config` rejects incompatible CoFiTi
  experiment formats before applying evaluation overrides.
- `src/nexus_align/engine/checkpoint.py`: `CheckpointManager.save_ckpt/load_ckpt`
  publishes complete checkpoints and verifies CoFiTi LoRA keys/shapes on load.
- `src/nexus_align/evaluators/reward_evaluator.py`: `RewardEvaluator` checks the
  full CoFiTi sampling identity before reusing either images or completed metrics;
  it selects prompt subsets, expands repeats, assigns image seeds, and scores the
  same uint8 pixels that it saves as PNG.
- `src/nexus_align/rewards/cofiti_multi.py`: `CoFiTiMultiReward.__call__` computes
  raw/calibrated components and the fixed report sum; `training_rewards` produces
  the source weighted component advantages for group projection.
- `src/nexus_align/rewards/cofiti_hpsv2.py`: `CoFiTiHPSv2` loads the local scorer;
  `preprocess` preserves float pixels and `__call__` scores under BF16 autocast.
- `src/nexus_align/rewards/clipscore.py`: `ClipScore` retains the shared divisor
  of 100; the CoFiTi composite requests raw logits with `score_divisor=1`.

Validation results are recorded in [COFITI_VALIDATION.md](COFITI_VALIDATION.md).
Migration-only tests and comparison scripts are removed after verification as
requested; they are not runtime dependencies. No full 500-epoch experiment or
paper reproduction result is claimed.
