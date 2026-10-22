"""GRPO trainer: online RL epochs of group sampling, rewarding, and policy updates."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, Dataset
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.models.base_model import BaseModel
from nexus_align.trainers.base_trainer import BaseTrainer
from nexus_align.algorithms.base_algorithm import BaseAlgorithm
from nexus_align.datasets.dist_dataloader import DistributedGroupSampler


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenate a tensor of the same shape from every rank, in rank order."""
    buffer = torch.empty(
        (dist.get_world_size() * tensor.shape[0], *tensor.shape[1:]),
        dtype=tensor.dtype, device=tensor.device,
    )
    dist.all_gather_into_tensor(buffer, tensor.contiguous())
    return buffer


def zero_std_ratio(prompts: list[str], rewards: np.ndarray) -> tuple[float, float]:
    """Fraction of prompt groups whose rewards are all equal, and the mean group std."""
    prompts = np.array(prompts)
    stds = np.array([np.std(rewards[prompts == p]) for p in np.unique(prompts)])
    return float(np.mean(stds == 0)), float(stds.mean())


class GRPOTrainer(BaseTrainer):
    """
    Generic online RL trainer; the algorithm owns the model-specific rollout and loss.

    - Each epoch samples groups_per_epoch prompts * group_size responses over
      num_batches_per_epoch batches of sample_batch_size per rank, then rewards them.
    - It then trains on every batch in algorithm.num_micro_steps forward passes; one
      optimizer step accumulates grad_accu_step batches, so an epoch has
      updates_per_epoch steps.
    - EMA is stepped after every batch, as in Flow-GRPO.

    The algorithm provides:
    - group_size: responses per prompt; autocast_dtype: the compute dtype of forward.
    - num_micro_steps: forward passes per batch, e.g. one per trained denoising step.
    - prepare(model, device, pipeline): bind the trained network and the model wrapper.
    - rollout(prompts, epoch): {"samples": per-sample training tensors, "responses"}, where
      the responses are scored by the reward.
    - compute_advantages(prompts, rewards): one advantage per response.
    - forward(batch): {"loss", "metrics"} of a batch of samples with "advantages" and
      the micro step index "step".
    """

    def __init__(
        self,
        cfg_trainer,
        train_dataset: Dataset,
        model: BaseModel,
        algorithm: BaseAlgorithm,
        reward,
        log_dir: str = "logs/temp",
    ) -> None:
        # Batches per epoch follow from the prompt groups per epoch and the world size.
        cfg_trainer = dict(cfg_trainer)
        samples_per_round = dist.get_world_size() * cfg_trainer["sample_batch_size"]
        samples_per_epoch = cfg_trainer["groups_per_epoch"] * algorithm.group_size
        num_batches = samples_per_epoch // samples_per_round
        updates = cfg_trainer["updates_per_epoch"]
        if samples_per_epoch % samples_per_round or num_batches % updates:
            raise ValueError(
                f"❌ groups_per_epoch * group_size ({samples_per_epoch}) must split into "
                f"batches of world_size * sample_batch_size ({samples_per_round}), and "
                f"their number into updates_per_epoch ({updates})"
            )
        if cfg_trainer.get("save_ckpt_every_n_steps") is not None:
            raise ValueError(
                "❌ GRPO saves only end-of-epoch checkpoints; use save_ckpt_every_n_epochs"
            )
        cfg_trainer["num_batches_per_epoch"] = num_batches
        cfg_trainer["grad_accu_step"] = num_batches // updates
        self.group_size = algorithm.group_size
        super().__init__(
            cfg_trainer=cfg_trainer,
            train_dataset=train_dataset,
            model=model,
            algorithm=algorithm,
            log_dir=log_dir
        )
        self.reward = reward
        self.algorithm.prepare(model.model, model.device, model)

        # Only the trainable weights, such as LoRA, are optimized.
        self.trainable_params = [p for p in model.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )
        autocast_dtype = DTYPE_MAP[self.algorithm.autocast_dtype]
        self.scaler = ShardedGradScaler(enabled=autocast_dtype == torch.float16)

        # A pool that computes rewards asynchronously.
        self.executor = ThreadPoolExecutor(max_workers=8)

        num_trainable = sum(p.numel() for p in self.trainable_params) / 1e6
        info = [f"\n📌 {self.name.upper()}:"]
        info += [f"    groups per epoch: {self.groups_per_epoch} x {self.group_size}"]
        info += [f"    batches per epoch: {num_batches} x {self.sample_batch_size}"]
        info += [f"    optimizer steps per epoch: {updates}"]
        info += [f"    grad_accu_step: {self.grad_accu_step}"]
        info += [f"    micro steps per batch: {self.algorithm.num_micro_steps}"]
        info += [f"    trainable params: {num_trainable:.2f} M"]
        print("\n".join(info))

    def get_train_dataloader(self, dataset: Dataset) -> DataLoader:
        """Build a dataloader that repeats each prompt group_size times over ranks."""
        sampler = DistributedGroupSampler(
            dataset,
            batch_size=self.sample_batch_size,
            group_size=self.group_size,
            num_batches=self.num_batches_per_epoch,
            seed=self.seed,
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

    def train_mode(self):
        self.model.model.train()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def forward(self, data):
        results = self.algorithm.forward(data)
        return {"loss": results["loss"], "metrics": results["metrics"]}

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    @contextmanager
    def no_sync(self):
        self.model.model.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            self.model.model.set_requires_gradient_sync(True)

    def clip_grad(self):
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        self._grad_norm = grad_norm.detach()

    def optimizer_step(self):
        # The scaler skips non-finite steps itself; otherwise skip them here.
        if not self.scaler.is_enabled() and not torch.isfinite(self._grad_norm):
            print(f"⚠️ WARNING: grad norm is {self._grad_norm.item()}; skip this step")
            return
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def lr_scheduler_step(self):
        pass

    def ema_step(self):
        """Step EMA every ema_update_interval steps, warming up its decay early on."""
        if self.model.ema is None or (self.total_step + 1) % self.ema_update_interval:
            return
        decay = min((1 + self.total_step) / (10 + self.total_step), self.ema_decay)
        self.model.ema_step(decay=decay)

    # ----------------------------------------
    # Sampling.
    # ----------------------------------------
    def sampler_epoch(self) -> int:
        """Return the epoch that seeds the prompts of this epoch."""
        return self.epoch

    @torch.no_grad()
    def sample_epoch(self) -> tuple[dict, dict]:
        """Sample one epoch of response groups, reward them, and compute advantages."""
        # 1. Roll out each batch and submit its rewards asynchronously.
        self.model.model.eval()
        self.train_dataloader.batch_sampler.set_epoch(self.sampler_epoch())
        batches, futures, all_prompts = [], [], []
        for prompts, metadatas in self.train_dataloader:
            out = self.algorithm.rollout(prompts, self.epoch)
            future = self.executor.submit(
                self.reward, out["responses"], prompts, metadatas, True
            )
            futures.append(future)
            all_prompts += prompts
            batches.append(out["samples"])
        samples = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}

        # 2. Gather the rewards and prompts of all ranks.
        rewards = defaultdict(list)
        for future in futures:
            for key, value in future.result().items():
                rewards[key] += list(value)
        rewards = {
            k: gather_tensor(torch.as_tensor(v, device=self.model.device).float())
            for k, v in rewards.items()
        }
        rewards = {k: v.cpu().numpy() for k, v in rewards.items()}
        gathered_prompts = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_prompts, all_prompts)
        gathered_prompts = sum(gathered_prompts, [])

        # 3. Advantages of this rank.
        reward = rewards[self.reward.name]
        advantages = self.algorithm.compute_advantages(gathered_prompts, reward)
        advantages = torch.as_tensor(advantages).reshape(dist.get_world_size(), -1)
        samples["advantages"] = advantages[dist.get_rank()].to(self.model.device)

        # Headline reward metrics over the samples they apply to (-10 marks others).
        ratio, std_mean = zero_std_ratio(gathered_prompts, reward)
        metrics = {
            f"reward_{k}": float(np.mean(v[v != -10])) for k, v in rewards.items()
            if k in self.reward.metrics
        }
        metrics.update({
            "zero_std_ratio": ratio,
            "reward_std_mean": std_mean,
            "advantages_abs_mean": samples["advantages"].abs().mean().item(),
        })
        return samples, metrics

    def filter_samples(self, samples: dict) -> dict:
        """Drop zero-advantage samples, keeping a multiple of num_batches_per_epoch."""
        mask = samples["advantages"] != 0
        num_batches = self.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0 or true_count == 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        return {k: v[mask] for k, v in samples.items()}

    # ----------------------------------------
    # Training loop.
    # ----------------------------------------
    def train_epoch(self, samples: dict, rollout_metrics: dict) -> None:
        """Train on the sampled epoch: shuffle, rebatch, and run each micro step."""
        samples = self.filter_samples(samples)
        total_batch_size = samples["advantages"].shape[0]
        if self.shuffle:
            perm = torch.randperm(total_batch_size, device=self.model.device)
            samples = {k: v[perm] for k, v in samples.items()}
        batch_size = total_batch_size // self.num_batches_per_epoch
        batches = [
            {k: v[i * batch_size: (i + 1) * batch_size] for k, v in samples.items()}
            for i in range(self.num_batches_per_epoch)
        ]
        rollout_metrics["actual_batch_size"] = batch_size

        self.train_mode()
        num_steps = self.algorithm.num_micro_steps
        accu_micro_steps = self.grad_accu_step * num_steps
        micro_step = 0
        info = defaultdict(list)
        for batch in batches:
            synced = False
            for j in range(num_steps):
                if micro_step % accu_micro_steps == 0:
                    self.zero_grad()
                micro_step += 1
                is_sync = micro_step % accu_micro_steps == 0
                with nullcontext() if is_sync else self.no_sync():
                    results = self.forward({**batch, "step": j})
                    self.backward(results["loss"] / accu_micro_steps)
                for key, value in results["metrics"].items():
                    info[key].append(value)

                if is_sync:
                    self.clip_grad()
                    self.optimizer_step()
                    self.lr_scheduler_step()
                    self.total_step += 1
                    self.step += 1
                    synced = True

            # EMA steps after every batch with the current optimizer step count.
            self.ema_step()
            if not synced:
                continue

            # Update runtime state, save checkpoints, and log after each optimizer step.
            epoch_finished = self.step >= self.steps_per_epoch
            if epoch_finished:
                self.epoch += 1
                self.step = 0
                self.end_epoch()
            self.checkpoint_manager.save_ckpt(self, epoch_finished=epoch_finished)
            self._metrics = {
                **{k: torch.stack(v).mean().item() for k, v in info.items()},
                **rollout_metrics,
            }
            info = defaultdict(list)
            self.meters.end("step")
            self.meters.update_train_state(self.state_dict())
            self.update_log()
            self.meters.start("step")

    def end_epoch(self) -> None:
        """Run after the last optimizer step of an epoch, before its checkpoint is saved."""
        pass

    def run(self) -> None:
        """Run training to max_epochs."""
        # Load checkpoint and runtime state.
        self.checkpoint_manager.load_ckpt(self)
        if self.step != 0:
            raise ValueError("❌ GRPO resumes only from end-of-epoch checkpoints")
        self.meters.update_train_state(self.state_dict())

        # Save the initial weights as the step-0 point of evaluation.
        if self.total_step == 0:
            self.checkpoint_manager.save_ckpt(self, epoch_finished=True)

        self.tracker.open_writer()
        for _ in range(self.epoch, self.max_epochs):
            self.meters.start("epoch")
            self.meters.start("step")
            samples, rollout_metrics = self.sample_epoch()
            self.train_epoch(samples, rollout_metrics)
            self.meters.end("epoch")
            self.meters.update_train_state(self.state_dict())
            self.update_log(epoch_finished=True)

        self.tracker.close_writer()
        self.meters.stop_hardware_monitoring()
        self.executor.shutdown()

        dist.barrier()
