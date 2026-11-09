"""TwinTilt epochs: complete groups, frozen targets, all-sample slot accumulation."""

from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from nexus_align.datasets.twintilt import (
    PromptBatchPlan, PromptDataLoader,
)
from nexus_align.engine.checkpoint import CheckpointManager
from nexus_align.engine.meter import WindowMeter
from nexus_align.engine.tracker import TrainingTracker
from nexus_align.trainers.base_trainer import BaseTrainer


def cpu_tree(value):
    """Detach checkpoint/rollout tensors without retaining autograd graphs."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def capture_rng():
    """Capture process-local Python, NumPy, Torch, and current CUDA RNG states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())


def semantic_config(config):
    """Copy configuration without resource locations or service transport settings."""
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if config is None:
        return {}
    if not isinstance(config, dict):
        config = vars(config)
    ignored = {"url", "timeout", "max_retries", "local_files_only"}
    result = {}
    for key, value in config.items():
        if (
            key in ignored or key == "path" or key.endswith("_path")
            or key.endswith("_dir")
        ):
            continue
        if isinstance(value, dict):
            value = semantic_config(value)
        result[key] = value
    return json.loads(json.dumps(result))


def resume_signature(trainer):
    """
    Identify effective training semantics, allowing relocated resources.

    Optimizer hyperparameters come from the checkpoint. Epoch limits, logging,
    save intervals, and train microbatch partitioning may change on restoration.
    """
    algorithm = semantic_config(trainer.algorithm.cfg_algorithm)
    # Fixed execution precision remains in the checkpoint's semantic signature.
    algorithm["autocast_dtype"] = "fp16"
    reward = semantic_config(getattr(trainer.reward, "cfg_reward", None))
    reward.update({
        "name": trainer.reward.name,
        "class": type(trainer.reward).__module__ + "." + type(trainer.reward).__name__,
    })
    for key in ("normalize", "image_input", "jpeg"):
        if hasattr(trainer.reward, key):
            reward[key] = getattr(trainer.reward, key)
    fields = (
        "seed", "deterministic", "groups_per_epoch", "updates_per_epoch",
        "sample_batch_size", "max_batch_size", "max_grad_norm",
        "scale_pickscore_reward_for_training", "reward_std_epsilon",
        "use_ema", "rollout_ema_decay",
        "rollout_ema_start_decay", "rollout_ema_warmup_rate",
    )
    return {
        "version": 1,
        "algorithm": algorithm,
        "model": semantic_config(getattr(trainer.model, "cfg_model", None)),
        "reward": reward,
        "trainer": {key: getattr(trainer, key, None) for key in fields},
    }


def signature_differences(saved, current, prefix=""):
    """Return the changed semantic fields for a useful resume error."""
    if isinstance(saved, dict) and isinstance(current, dict):
        changed = []
        for key in sorted(saved.keys() | current.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in saved or key not in current:
                changed.append(path)
            else:
                changed.extend(signature_differences(saved[key], current[key], path))
        return changed
    return [] if saved == current else [prefix]


class RolloutEMA:
    """FP32 LoRA shadow with the source's successful-update warmup."""

    def __init__(self, module, decay=0.9, warmup_rate=0.01, start_decay=None):
        self.module = module
        self.decay = float(decay)
        self.warmup_rate = float(warmup_rate)
        self.start_decay = start_decay
        if not 0 <= self.decay <= 1 or self.warmup_rate < 0:
            raise ValueError("Invalid rollout EMA decay/warmup")
        if start_decay is not None and not 0 <= start_decay <= self.decay:
            raise ValueError("EMA start decay must lie between zero and its cap")
        self.updates = 0
        self.shadow = {
            key: value.detach().float().clone()
            for key, value in module.named_parameters() if value.requires_grad
        }

    @torch.no_grad()
    def update(self):
        self.updates += 1
        initial = self.warmup_rate if self.start_decay is None else self.start_decay
        decay = min(initial + self.warmup_rate * (self.updates - 1), self.decay)
        for key, parameter in self.module.named_parameters():
            if key in self.shadow:
                self.shadow[key].mul_(decay).add_(
                    parameter.detach().float(), alpha=1 - decay,
                )
        return decay

    @contextmanager
    def apply(self):
        """Temporarily use the shadow, restoring student parameters on exceptions."""
        parameters = dict(self.module.named_parameters())
        backup = {key: parameters[key].detach().clone() for key in self.shadow}
        try:
            with torch.no_grad():
                for key, value in self.shadow.items():
                    parameters[key].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for key, value in backup.items():
                    parameters[key].copy_(value)

    def state_dict(self):
        return {"updates": self.updates, "shadow": cpu_tree(self.shadow)}

    def load_state_dict(self, state):
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("Rollout EMA parameter names do not match this model")
        self.updates = int(state["updates"])
        for key, value in state["shadow"].items():
            if value.shape != self.shadow[key].shape:
                raise ValueError(f"Rollout EMA shape mismatch: {key}")
            self.shadow[key].copy_(value.to(self.shadow[key]))



class RolloutRuntime:
    """Rank-local RNG and remaining epoch buffers, stored by rank 0 atomically."""

    def __init__(self, trainer):
        self.trainer = trainer

    def state_dict(self):
        trainer = self.trainer
        local = {
            "rng": capture_rng(),
            "buffer": cpu_tree(trainer.buffer),
            "dataloader": trainer.train_dataloader.state_dict(),
            "dataset": trainer.train_dataset.dataset_fingerprint,
        }
        states = [None] * trainer.world_size if trainer.rank == 0 else None
        dist.gather_object(local, states, dst=0)
        return {"world_size": trainer.world_size, "ranks": states}

    def load_state_dict(self, state):
        trainer = self.trainer
        if any(
            local["dataset"] != trainer.train_dataset.dataset_fingerprint
            for local in state["ranks"]
        ):
            raise ValueError("TwinTilt resume requires the same prompt dataset")
        if int(state["world_size"]) != trainer.world_size:
            raise ValueError("TwinTilt resume requires the same world size")
        local = state["ranks"][trainer.rank]
        trainer.train_dataloader.load_state_dict(local["dataloader"])
        trainer.buffer = local["buffer"]
        if trainer.step and trainer.buffer is None:
            raise ValueError("Mid-epoch checkpoint is missing its rollout buffer")
        trainer.pending_rng = local["rng"]


class TwinTiltTrainer(BaseTrainer):
    """
    Keep source epoch semantics while using NexusAlign tracking/checkpoints.

    Targets are constructed before global slot shuffling. Each update averages
    all member/time slots, including those with zero correction.
    """

    def __init__(
        self, cfg_trainer, train_dataset, model, algorithm, reward,
        log_dir="logs/temp",
    ):
        allowed = {
            'adam_beta1',
            'adam_beta2',
            'adam_epsilon',
            'adam_weight_decay',
            'debug',
            'deterministic',
            'epochs',
            'groups_per_epoch',
            'learning_rate',
            'log_every_n_steps',
            'log_hardware',
            'log_tensorboard',
            'log_window_size',
            'max_batch_size',
            'max_grad_norm',
            'name',
            'offload_text_encoders',
            'plot_every_n_seconds',
            'resume_path',
            'reward_std_epsilon',
            'rollout_ema_decay',
            'rollout_ema_start_decay',
            'rollout_ema_warmup_rate',
            'sample_batch_size',
            'save_ckpt_every_n_epochs',
            'save_ckpt_every_n_steps',
            'save_ckpt_keep_last_n_epochs',
            'save_ckpt_keep_last_n_steps',
            'scale_pickscore_reward_for_training',
            'seed',
            'train_batch_size',
            'updates_per_epoch',
            'use_ema',
        }
        unknown = set(cfg_trainer) - allowed
        if unknown:
            raise ValueError(f"Unsupported TwinTilt trainer fields: {sorted(unknown)}")
        for key, value in cfg_trainer.items():
            setattr(self, key, value)
        self.train_dataset = train_dataset
        self.model, self.algorithm, self.reward = model, algorithm, reward
        self.device = model.device
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.log_dir = str(log_dir)
        self.algorithm.prepare(model, self.device)
        self.batch_plan = PromptBatchPlan.build(
            total_groups_per_epoch=self.groups_per_epoch,
            world_size=self.world_size,
            batch_size=self.sample_batch_size,
            max_batch_size=self.max_batch_size,
        )
        self.train_dataloader = PromptDataLoader(
            train_dataset, self.rank, self.batch_plan, seed=self.seed,
        )
        self.steps_per_epoch = int(self.updates_per_epoch)
        members = self.groups_per_epoch // self.world_size * algorithm.group_size
        if self.steps_per_epoch < 1 or members % self.steps_per_epoch:
            raise ValueError("Local group members must divide updates_per_epoch")
        if self.train_batch_size is None:
            units = members // self.steps_per_epoch
            self.train_batch_size = next(
                value for value in range(self.max_batch_size, 0, -1)
                if units % value == 0
            )
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be positive")
        self.max_epochs = int(self.epochs)
        if self.max_epochs < 1:
            raise ValueError("epochs must be positive")
        for key in ("save_ckpt_every_n_steps", "save_ckpt_every_n_epochs"):
            interval = getattr(self, key)
            if interval is not None and (
                isinstance(interval, bool) or not isinstance(interval, int)
                or interval < 1
            ):
                raise ValueError(f"{key} must be a positive integer or null")
        self.max_total_steps = self.max_epochs * self.steps_per_epoch
        self.total_step = self.epoch = self.step = 0
        self.buffer = None
        self.pending_rng = None
        self._metrics = {}
        self.trainable_params = [
            value for value in model.model.parameters() if value.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay, eps=self.adam_epsilon,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda",
        )
        self.train_ema = (
            RolloutEMA(
                model.transformer, self.rollout_ema_decay,
                self.rollout_ema_warmup_rate, self.rollout_ema_start_decay,
            ) if self.use_ema else None
        )
        self.runtime = RolloutRuntime(self)
        self.restore_optimizer_hparams = True
        self.save_ckpt_keys = [
            "train_state", "model", "optimizer", "scaler", "runtime",
        ]
        if self.train_ema is not None:
            self.save_ckpt_keys.insert(2, "train_ema")
        self.load_ckpt_keys = self.save_ckpt_keys.copy()
        self.checkpoint_manager = CheckpointManager(
            save_ckpt_dir=str(Path(log_dir) / "checkpoints"),
            load_ckpt_dir=self.resume_path,
            save_ckpt_every_n_steps=self.save_ckpt_every_n_steps,
            save_ckpt_every_n_epochs=self.save_ckpt_every_n_epochs,
            save_ckpt_keep_last_n_steps=self.save_ckpt_keep_last_n_steps,
            save_ckpt_keep_last_n_epochs=self.save_ckpt_keep_last_n_epochs,
        )
        self.meters = WindowMeter(hardware=self.log_hardware)
        self.meters.register_timing_meters(step_window=self.log_window_size)
        self.meters.set_max_train_state(
            self.max_epochs, self.steps_per_epoch, self.max_total_steps,
        )
        self.tracker = TrainingTracker(
            log_dir, self.plot_every_n_seconds, self.log_tensorboard,
        )
        print(
            f"📌 TwinTilt: {self.groups_per_epoch} groups x "
            f"{algorithm.group_size} members x {algorithm.num_steps} times, "
            f"{self.steps_per_epoch} updates/epoch, "
            f"{self.train_batch_size} slots/microbatch"
        )

    def state_dict(self):
        return {
            "method": "twintilt", "format_version": 1,
            "target_format": "group_projection_displacement_v1",
            "epoch": self.epoch, "step": self.step, "total_step": self.total_step,
            "steps_per_epoch": self.steps_per_epoch,
            "world_size": self.world_size,
            "groups_per_epoch": self.groups_per_epoch,
            "group_size": self.algorithm.group_size,
            "num_steps": self.algorithm.num_steps,
            "resume_signature": resume_signature(self),
        }

    def load_state_dict(self, state):
        for key, expected in (
            ("method", "twintilt"), ("format_version", 1),
            ("target_format", "group_projection_displacement_v1"),
            ("world_size", self.world_size),
            ("steps_per_epoch", self.steps_per_epoch),
            ("groups_per_epoch", self.groups_per_epoch),
            ("group_size", self.algorithm.group_size),
            ("num_steps", self.algorithm.num_steps),
        ):
            if state.get(key) != expected:
                raise ValueError(f"Incompatible TwinTilt checkpoint: {key}")
        if "resume_signature" not in state:
            raise ValueError("TwinTilt checkpoint requires a resume_signature")
        changes = signature_differences(
            state["resume_signature"], resume_signature(self)
        )
        if changes:
            raise ValueError(
                "TwinTilt resume changes training semantics: " + ", ".join(changes)
            )
        for key in ("epoch", "step", "total_step"):
            value = state.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Invalid checkpoint counter: {key}")
        if state["step"] >= self.steps_per_epoch:
            raise ValueError("Checkpoint step must be within the current epoch")
        self.epoch, self.step = int(state["epoch"]), int(state["step"])
        self.total_step = int(state["total_step"])
        if self.total_step != self.epoch * self.steps_per_epoch + self.step:
            raise ValueError("Inconsistent checkpoint training counters")
        self.meters.set_max_train_state(
            self.max_epochs, self.steps_per_epoch, self.max_total_steps,
        )

    def restore(self):
        """Restore the complete experiment, including each rank's prompt stream."""
        if not self.resume_path:
            return
        self.checkpoint_manager.load_ckpt(self)
        if (
            self.buffer is not None and self.offload_text_encoders
            and hasattr(self.model, "text_encoders_to")
        ):
            # Buffered mid-epoch resumes skip sample_epoch's normal offload.
            self.model.text_encoders_to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)
            self.pending_rng = None

    def train_mode(self):
        self.model.model.train()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def forward(self, data):
        return self.algorithm.forward(data)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def no_sync(self):
        if isinstance(self.model.model, DistributedDataParallel):
            return self.model.model.no_sync()
        return nullcontext()

    def clip_grad(self):
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            self.trainable_params, self.max_grad_norm,
        )
        self._grad_norm = float(norm)

    def optimizer_step(self):
        previous = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_succeeded = self.scaler.get_scale() >= previous

    def lr_scheduler_step(self):
        pass

    def ema_step(self):
        if self.train_ema is not None and self.update_succeeded:
            self.train_ema.update()

    def gather(self, tensor):
        """Gather compact or trajectory tensors in source-rank order."""
        tensor = tensor.contiguous().to(self.device)
        if self.world_size == 1:
            return tensor.cpu()
        combined = torch.empty(
            (self.world_size * tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype, device=self.device,
        )
        dist.all_gather_into_tensor(combined, tensor)
        return combined.cpu()

    @torch.no_grad()
    def sample_epoch(self):
        """Encode first, roll out whole groups, then build and shuffle frozen labels."""
        self.model.model.eval()
        if hasattr(self.model, "text_encoders_to"):
            self.model.text_encoders_to(self.device)
        batches, embed_chunks, pool_chunks = [], [], []
        size = self.batch_plan.batch_size
        for _ in range(self.batch_plan.num_batches_per_epoch):
            batch = self.train_dataloader.get_prompt_batch(size)
            embeds, pooled = self.model.encode_prompt(
                batch.prompts, self.algorithm.max_sequence_length,
            )
            batches.append(batch)
            embed_chunks.append(embeds.cpu())
            pool_chunks.append(pooled.cpu())
        negative = {}
        if abs(self.algorithm.guidance_scale - 1) > 1e-6:
            embeds, pooled = self.model.encode_prompt(
                [""], self.algorithm.max_sequence_length,
            )
            negative = {
                "negative_prompt_embeds": embeds.cpu(),
                "negative_pooled_prompt_embeds": pooled.cpu(),
            }
        if self.offload_text_encoders and hasattr(self.model, "text_encoders_to"):
            self.model.text_encoders_to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        pieces = []
        metrics = {}
        saved_embeds, saved_pooled = [], []
        start = time.monotonic()
        context = (
            self.train_ema.apply() if self.train_ema is not None else nullcontext()
        )
        with context:
            for index, prompt_batch in enumerate(batches):
                data = {
                    "prompt_embeds": embed_chunks[index].to(self.device),
                    "pooled_prompt_embeds": pool_chunks[index].to(self.device),
                    **{key: value.to(self.device) for key, value in negative.items()},
                }
                sampled = self.algorithm.sample(data)
                group = self.algorithm.group_size
                prompts = [p for p in prompt_batch.prompts for _ in range(group)]
                metadata = [
                    copy.deepcopy(m) for m in (
                        prompt_batch.metadata or [{} for _ in prompt_batch.prompts]
                    ) for _ in range(group)
                ]
                scored = self.reward(sampled["images"], prompts, metadata)
                rewards = torch.as_tensor(
                    scored[self.reward.name], device=self.device, dtype=torch.float32,
                )
                if not bool(torch.isfinite(rewards).all()):
                    raise ValueError("Reward service returned non-finite scores")
                normalized = (
                    self.scale_pickscore_reward_for_training
                    and self.reward.name == "pickscore"
                )
                training_rewards = rewards
                postprocess = getattr(self.reward, "training_rewards", None)
                if postprocess is not None:
                    training_rewards = postprocess(scored, group, self.device)
                elif normalized:
                    grouped = rewards.reshape(-1, self.algorithm.group_size)
                    training_rewards = (
                        (grouped - grouped.mean(1, keepdim=True))
                        / (grouped.std(1, unbiased=False, keepdim=True)
                           + self.reward_std_epsilon)
                    ).flatten()
                sentinels = getattr(self.reward, "metric_sentinels", {})
                for key, values in scored.items():
                    values = torch.as_tensor(values, dtype=torch.float32)
                    valid = (
                        values[values != sentinels[key]] if key in sentinels else values
                    )
                    metrics.setdefault("reward_" + key, []).append(
                        (float(valid.sum()), valid.numel())
                    )
                targets = self.algorithm.build_targets(
                    sampled["final_latents"], training_rewards,
                )
                member_count = rewards.numel()
                bank_multiplier = (
                    1 if self.algorithm.deduplicate_prompt_embeddings else group
                )
                saved_embeds.append(sampled["prompt_embeds"].cpu())
                saved_pooled.append(sampled["pooled_prompt_embeds"].cpu())
                pieces.append(cpu_tree({
                    "xs": sampled["xs"], "velocities": sampled["velocities"],
                    "timesteps": sampled["timesteps"].expand(member_count, -1),
                    "displacement": targets["displacement"],
                    "prompt_indices": sampled["prompt_indices"] + bank_multiplier * (
                        index * size
                        + self.rank * self.groups_per_epoch // self.world_size
                    ),
                }))
                for key, value in targets["metrics"].items():
                    metrics.setdefault(key, []).append((float(value), 1))
                del sampled, targets, data
        local = {
            key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]
        }
        members, steps = local["xs"].shape[:2]
        local_slots = members * steps
        permutation = torch.randperm(
            local_slots * self.world_size,
            generator=torch.Generator().manual_seed(self.seed + self.epoch * 12345),
        )
        indices = permutation[
            self.rank * local_slots:(self.rank + 1) * local_slots
        ]
        buffer = {"member_indices": indices // steps, **negative}
        for key in ("xs", "velocities", "timesteps"):
            global_value = self.gather(local[key].flatten(0, 1))
            buffer[key] = global_value.index_select(0, indices)
        for key in ("displacement", "prompt_indices"):
            buffer[key] = self.gather(local[key])
        buffer["prompt_embeds"] = self.gather(torch.cat(saved_embeds))
        buffer["pooled_prompt_embeds"] = self.gather(torch.cat(saved_pooled))
        all_metrics = [None] * self.world_size
        dist.all_gather_object(all_metrics, metrics)
        metrics = {}
        for rank_metrics in all_metrics:
            for key, values in rank_metrics.items():
                metrics.setdefault(key, []).extend(values)
        buffer["rollout_metrics"] = {
            key: sum(v for v, n in values) / max(sum(n for v, n in values), 1)
            for key, values in metrics.items()
        }
        buffer["rollout_metrics"]["rollout_seconds"] = time.monotonic() - start
        buffer["rollout_metrics"]["samples_per_epoch"] = (
            self.groups_per_epoch * self.algorithm.group_size
        )
        buffer["rollout_metrics"]["slots_per_epoch"] = local_slots * self.world_size
        return buffer

    def make_batch(self, start, end):
        buffer = self.buffer
        member = buffer["member_indices"][start:end]
        prompt = buffer["prompt_indices"].index_select(0, member)
        batch = {
            key: buffer[key][start:end].to(self.device)
            for key in ("xs", "velocities", "timesteps")
        }
        for key in ("displacement",):
            batch[key] = buffer[key].index_select(0, member).to(self.device)
        for key in ("prompt_embeds", "pooled_prompt_embeds"):
            batch[key] = buffer[key].index_select(0, prompt).to(self.device)
        for key in ("negative_prompt_embeds", "negative_pooled_prompt_embeds"):
            if key in buffer:
                batch[key] = buffer[key].to(self.device)
        return batch

    def train_update(self):
        """Accumulate one equal global slot partition, then update student and EMA."""
        self.train_mode()
        slots = self.buffer["xs"].shape[0]
        if slots % self.steps_per_epoch:
            raise ValueError("Training slots must divide optimizer updates")
        count = slots // self.steps_per_epoch
        start, stop = self.step * count, (self.step + 1) * count
        self.zero_grad()
        metrics = {}
        for offset in range(start, stop, self.train_batch_size):
            end = min(offset + self.train_batch_size, stop)
            weight = (end - offset) / count
            context = nullcontext() if end == stop else self.no_sync()
            with context:
                result = self.forward(self.make_batch(offset, end))
                self.backward(result["loss"] * weight)
            for key, value in result.get("metrics", {}).items():
                if key == "loss":
                    continue
                metrics[key] = metrics.get(key, 0.0) + float(value) * weight
            metrics["loss"] = metrics.get("loss", 0.0) + float(
                result["loss"].detach()
            ) * weight
        self.clip_grad()
        self.optimizer_step()
        self.ema_step()
        self.zero_grad()
        metrics["optimizer_step_succeeded"] = float(self.update_succeeded)
        metrics["grad_norm"] = self._grad_norm
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics["ema_update_step"] = (
            self.train_ema.updates if self.train_ema is not None else 0
        )
        self._metrics = {**self.buffer["rollout_metrics"], **metrics}

    def update_log(self, epoch_finished=False, update_metrics=True):
        if not epoch_finished and self.total_step % self.log_every_n_steps:
            return
        if update_metrics:
            keys = sorted(self._metrics)
            values = torch.tensor(
                [self._metrics[key] for key in keys],
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(values)
            values /= self.world_size
            for key, value in zip(keys, values.tolist()):
                if key not in self.meters.meters:
                    self.meters.register_meter(key, self.log_window_size)
                self.meters.update(key, value)
        self.meters.update_train_state(self.state_dict())
        self.tracker.log_all_meters(self.meters, epoch_finished=epoch_finished)

    def run(self):
        """Restore, sample each epoch once, update, and checkpoint exact boundaries."""
        self.restore()
        self.tracker.open_writer()
        try:
            while self.epoch < self.max_epochs:
                self.meters.start("epoch")
                if self.buffer is None:
                    self.buffer = self.sample_epoch()
                while self.step < self.steps_per_epoch:
                    self.meters.start("step")
                    self.train_update()
                    self.total_step += 1
                    self.step += 1
                    finished = self.step == self.steps_per_epoch
                    self.meters.end("step")
                    if finished:
                        self.epoch += 1
                        self.step = 0
                        self.buffer = None
                        self.meters.end("epoch")
                    self.checkpoint_manager.save_ckpt(
                        self, epoch_finished=finished,
                    )
                    self.update_log(epoch_finished=finished)
                    if finished:
                        break
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            self.tracker.close_writer()
            self.meters.stop_hardware_monitoring()
        dist.barrier()
