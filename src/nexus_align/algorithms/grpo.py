"""GRPO algorithm."""

import torch
import torch.distributed as dist

from nexus_align.core import BaseAlgorithm
from nexus_align.core.config import DTYPE_MAP
from nexus_align.core.registry import registry
from nexus_align.utils.progress import TqdmBar
from nexus_align.engine.meter import WindowMeter
from nexus_align.core.base_model import BaseModel
from nexus_align.core.base_reward_model import BaseRewardModel
from nexus_align.engine.model_utils import clone_params, compute_update_ratio


class GRPOAlgorithm(BaseAlgorithm):
    """GRPO algorithm. Checkpoint, logger, etc. are handled by Trainer."""

    def __init__(
        self,
        reward_model: BaseRewardModel,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        amp_dtype: str,
        kwargs: dict,
    ) -> None:
        cfg_algo_run = kwargs["algorithm"].get("run", {})
        cfg_algo_train = kwargs["algorithm"].get("train", {})

        # Model & data
        self.model = model.model
        self.reward_model = reward_model

        # Optimize
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dtype = self.model.dtype
        self.amp_dtype = DTYPE_MAP[amp_dtype]
        self.num_epochs = cfg_algo_train["num_epochs"]
        self.max_train_total_steps = cfg_algo_train["max_train_total_steps"]
        self.grad_accu_step = cfg_algo_train["grad_accu_step"]
        self.max_grad_norm = cfg_algo_train["max_grad_norm"]

        # Pipeline
        model_name = kwargs['model']['name']
        self.pipeline = registry.get("pipeline", f"{model_name}_train")(
            model=model, 
            device=device, 
            model_dtype=self.model_dtype,
            amp_dtype=self.amp_dtype,
            kwargs=kwargs
        )

        # GRPO-specific
        self.group_size = cfg_algo_run["group_size"]
        self.adv_clip_max = cfg_algo_run["adv_clip_max"]
        self.clip_range = cfg_algo_run["clip_range"]
        self.kl_coeff = cfg_algo_run.get("kl_coeff", 0.0)
        
        # Meters (algorithm-specific metrics)
        meters = WindowMeter()
        meters.add_epoch_step(epoch_window=5, step_window=100)
        meters.add_new_meter("loss", window_size=100)
        meters.add_new_meter("clipped_loss", window_size=100)
        meters.add_new_meter("unclipped_loss", window_size=100)
        meters.add_new_meter("lr", window_size=100, report_mean=False)
        meters.add_new_meter("grad_norm", window_size=100)
        meters.add_new_meter("update_ratio", window_size=100)
        meters.add_new_meter("ratio_mean", window_size=100)
        meters.add_new_meter("ratio_std", window_size=100)
        meters.add_new_meter("clipped_ratio", window_size=100)
        meters.add_new_meter("reward_mean", window_size=100)
        meters.add_new_meter("reward_std", window_size=100)
        meters.add_new_meter("advantage_mean", window_size=100)
        meters.add_new_meter("advantage_std", window_size=100)
        meters.add_new_meter("kl", window_size=100)
        self.meters = meters
    
    def prepare_data(self, data: dict) -> dict:
        """Prepare data from dataloader batch."""
        data = self.pipeline.prepare_data(data)
        
        # Duplicate data to build GRPO groups for each input
        for key, value in list(data.items()):
            if key in data["keys_to_build_groups"]:
                if isinstance(value, torch.Tensor):
                    data[key] = torch.repeat_interleave(
                        value, repeats=self.group_size, dim=0
                    )
                elif isinstance(value, list):
                    data[key] = [v for v in value for _ in range(self.group_size)]
                else:
                    raise ValueError(f"❌ Unsupported data type: {type(value)}")

        return data

    @torch.no_grad()
    def sample_responses(self, data: dict) -> dict:
        """Rollout: generate responses from the prepared data."""
        return self.pipeline.sample_responses(data)

    def compute_rewards(self, data: dict) -> dict:
        """Score responses with reward model."""
        # Calculate rewards by calling reward model
        rewards = self.reward_model.evaluate(
            data=data["reward_inputs"], 
            return_tensor=True
        ).to(self.model_dtype)

        data["rewards"] = rewards.detach()
        
        # Gather rewards from all processes and print
        gathered = [torch.empty_like(rewards) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, rewards)
        gathered = torch.cat(gathered)

        mean = gathered.mean().item()
        std= gathered.std().item()
        
        self.meters.update("reward_mean", mean)
        self.meters.update("reward_std", std)
        
        dec = 4 if mean > 1 else 6
        print(f"{len(gathered)} rewards (mean +- std): {mean:.{dec}f} +- {std:.{dec}f}")

        return data

    def compute_advantages(self, data: dict) -> dict:
        """Turn rewards into advantages."""
        # Calculate advantages by normalizing the rewards among each group
        rewards = data["rewards"]
        train_batch_size = len(rewards) // self.group_size
        advantages = torch.zeros_like(rewards)

        for cond_idx in range(train_batch_size):
            start_idx = cond_idx * self.group_size
            end_idx = (cond_idx + 1) * self.group_size
            group_rewards = rewards[start_idx:end_idx]

            mean = group_rewards.mean()
            std = group_rewards.std() + 1e-8

            advantages[start_idx:end_idx] = (group_rewards - mean) / std

        data["advantages"] = advantages
        
        # Gather advantages from all processes and print
        gathered = [torch.empty_like(advantages) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, advantages)
        gathered = torch.cat(gathered)

        mean = gathered.mean().item()
        std = gathered.std().item()

        self.meters.update("advantage_mean", mean)
        self.meters.update("advantage_std", std)

        dec = 4 if mean > 1 else 6
        print(f"{len(gathered)} advantages (mean +- std): {mean:.{dec}f} +- {std:.{dec}f}")
        
        return data

    def train_one_step(self, data: dict) -> None:
        """Compute loss and update the model for one step."""
        metrics = {
            "loss": 0.0,
            "clipped_loss": 0.0,
            "unclipped_loss": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "clipped_ratio": 0.0,
            "kl": 0.0,
        }
        num_log = 0

        self.model.train()
        self.optimizer.zero_grad()

        self.lr_scheduler.step()
        self.meters.update("lr", self.lr_scheduler.get_last_lr()[0])

        last_group_idx = -1
        bar = None
        info = ""
        
        for item in self.pipeline.iterate_training_items(data):
            old_log_probs = item["old_log_probs"]
            ref_log_probs = item["ref_log_probs"]
            advantages = torch.clamp(
                input=item["advantages"], 
                min=-self.adv_clip_max, 
                max=self.adv_clip_max
            )
            backward_scale = item["backward_scale"]
            should_optimizer_step = item["should_optimizer_step"]
            group_idx = item["group_idx"]
            
            # Maintain progress bar for each group
            if group_idx != last_group_idx:
                if bar is not None:
                    bar.close()
                    print("  ".join(info))
                last_group_idx = group_idx
                pi = item.get("progress_info")
                if pi:
                    bar = TqdmBar(
                        total=pi["total"],
                        desc=pi["desc"],
                        unit=pi["unit"],
                    )
                else:
                    bar = None

            new_log_probs = item["forward_fn"]()
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Compute clipped ratio (percentage of samples that are clipped)
            ratio_lower = 1.0 - self.clip_range
            ratio_upper = 1.0 + self.clip_range
            is_clipped = (ratio < ratio_lower) | (ratio > ratio_upper)
            clipped_ratio = is_clipped.detach().float().mean()

            # PPO clipped surrogate loss
            unclipped_loss = advantages * ratio
            clipped_loss = advantages * torch.clamp(ratio, ratio_lower, ratio_upper)
            policy_loss = -torch.mean(torch.min(unclipped_loss, clipped_loss))

            # KL divergence against reference model (Schulman estimator)
            log_ratio_ref = ref_log_probs - new_log_probs
            kl = torch.mean(torch.exp(log_ratio_ref) - log_ratio_ref - 1)

            loss = policy_loss + self.kl_coeff * kl
            (loss * backward_scale).backward()

            loss = loss.detach()
            kl_val = kl.detach()
            clipped_loss = -clipped_loss.detach().mean()
            unclipped_loss = -unclipped_loss.detach().mean()
            ratio_mean = ratio.detach().mean()
            ratio_std = ratio.detach().std()
            
            # Log metrics
            info = []
            info += [f"loss: {loss.item():.6f}"]
            info += [f"clipped_loss: {clipped_loss.item():.6f}"]
            info += [f"unclipped_loss: {unclipped_loss.item():.6f}"]
            info += [f"ratio_mean: {ratio_mean.item():.6f}"]
            info += [f"ratio_std: {ratio_std.item():.6f}"]
            info += [f"clipped_ratio: {clipped_ratio:.6f}"]
            info += [f"kl: {kl_val.item():.6f}"]

            metrics["loss"] += loss
            metrics["clipped_loss"] += clipped_loss
            metrics["unclipped_loss"] += unclipped_loss
            metrics["ratio_mean"] += ratio_mean
            metrics["ratio_std"] += ratio_std
            metrics["clipped_ratio"] += clipped_ratio
            metrics["kl"] += kl_val
            num_log += 1

            if bar is not None:
                bar.update(1)
            
            # Update model by gradient backward and optimizer step
            if should_optimizer_step:
                prev_params = clone_params(self.model)
                grad_norm = self.model.clip_grad_norm_(self.max_grad_norm).item()
                self.meters.update("grad_norm", grad_norm)
                self.optimizer.step()
                
                update_ratio = compute_update_ratio(self.model, prev_params)
                self.meters.update("update_ratio", update_ratio)
                grad_info = f"update_ratio: {update_ratio:.6f} grad_norm: {grad_norm:.6f}"
                
                self.optimizer.zero_grad()
                
                print(f"✅ Updated model on the batch {group_idx + 1} / {self.pipeline.train_batch_size} ({grad_info})")

        if bar is not None:
            bar.close()

        # save log metrics
        for k, v in metrics.items():
            v /= num_log
            dist.all_reduce(v, op=dist.ReduceOp.AVG)
            self.meters.update(k, v.item() if isinstance(v, torch.Tensor) else v)

        dist.barrier()
        torch.cuda.empty_cache()
