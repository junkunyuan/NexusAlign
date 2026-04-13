"""DPO algorithm (Diffusion-DPO)."""

import torch
import torch.nn.functional as F
import torch.distributed as dist

from nexus_align.core import BaseAlgorithm
from nexus_align.core.config import DTYPE_MAP
from nexus_align.core.registry import registry
from nexus_align.engine.meter import WindowMeter
from nexus_align.core.base_model import BaseModel
from nexus_align.core.base_reward_model import BaseRewardModel
from nexus_align.pipelines.pipeline_flux_dpo import FluxDPOTrainPipeline


def diffusion_dpo_loss(
    model_pred_w: torch.Tensor,
    model_pred_l: torch.Tensor,
    ref_pred_w: torch.Tensor,
    ref_pred_l: torch.Tensor,
    target_w: torch.Tensor,
    target_l: torch.Tensor,
    beta_dpo: float = 5000.0,
) -> dict:
    """Compute diffusion-DPO loss and expose key scalar terms for logging."""
    dims = list(range(1, model_pred_w.ndim))
    losses_w = F.mse_loss(model_pred_w, target_w, reduction="none").mean(dim=dims)
    losses_l = F.mse_loss(model_pred_l, target_l, reduction="none").mean(dim=dims)
    ref_losses_w = F.mse_loss(ref_pred_w, target_w, reduction="none").mean(dim=dims)
    ref_losses_l = F.mse_loss(ref_pred_l, target_l, reduction="none").mean(dim=dims)
    model_diff = losses_w - losses_l
    ref_diff = ref_losses_w - ref_losses_l
    diff_minus_ref = model_diff - ref_diff
    inside_term = -1.0 * beta_dpo * diff_minus_ref
    loss = -1.0 * F.logsigmoid(inside_term).mean()
    metrics = {
        "loss_mean": loss.detach().mean().item(),
        "inside_mean": inside_term.detach().mean().item(),
        "diff_minus_ref_mean": diff_minus_ref.detach().mean().item(),
        "inside_min": inside_term.detach().min().item(),
        "inside_max": inside_term.detach().max().item(),
    }
    return {"loss": loss, "metrics": metrics}


class DPOAlgorithm(BaseAlgorithm):
    """DPO algorithm. Checkpoint, logger, etc. are handled by Trainer."""

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
        self.ref_model = getattr(model, "ref_model", None)
        if self.ref_model is None:
            raise ValueError("❌ ref_model is None. Please set model.ref.enable=true")

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
        model_name = kwargs["model"]["name"]
        self.pipeline: FluxDPOTrainPipeline = registry.get(
            "pipeline", f"{model_name}_dpo_train"
        )(
            model=model,
            device=device,
            model_dtype=self.model_dtype,
            amp_dtype=self.amp_dtype,
            kwargs=kwargs,
        )

        # DPO-specific
        self.group_size = cfg_algo_run["group_size"]
        self.beta_dpo = cfg_algo_run["beta_dpo"]

        # Meters (algorithm-specific metrics)
        meters = WindowMeter()
        meters.add_epoch_step(epoch_window=5, step_window=100)
        meters.add_new_meter("loss", window_size=100)
        meters.add_new_meter("lr", window_size=100, report_mean=False)
        meters.add_new_meter("grad_norm", window_size=100)
        meters.add_new_meter("reward_mean", window_size=100)
        meters.add_new_meter("reward_std", window_size=100)
        meters.add_new_meter("reward_margin_mean", window_size=100)
        meters.add_new_meter("dpo_inside_term", window_size=100)
        meters.add_new_meter("dpo_diff_minus_ref", window_size=100)
        meters.add_new_meter("dpo_inside_term_min", window_size=100)
        meters.add_new_meter("dpo_inside_term_max", window_size=100)
        self.meters = meters

    def prepare_data(self, data: dict) -> dict:
        """Prepare data from dataloader batch."""
        data = self.pipeline.prepare_data(data)

        # Duplicate data to build groups for each input (same prompt -> group_size samples)
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
        rewards = self.reward_model.evaluate(
            data=data["reward_inputs"],
            return_tensor=True,
        ).to(self.model_dtype)
        data["rewards"] = rewards.detach()
        gathered = [torch.empty_like(rewards) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, rewards)
        gathered = torch.cat(gathered)
        mean = gathered.mean().item()
        std = gathered.std().item()
        self.meters.update("reward_mean", mean)
        self.meters.update("reward_std", std)
        dec = 4 if mean > 1 else 6
        print(f"{len(gathered)} rewards (mean +- std): {mean:.{dec}f} +- {std:.{dec}f}")
        return data

    def compute_advantages(self, data: dict) -> dict:
        """Build winner/loser pairs from rewards; Trainer calls this, DPO only fills pairs."""
        data = self.pipeline.build_preference_pairs(data, self.meters)
        return data

    def train_one_step(self, data: dict) -> None:
        """Compute loss and update the model for one step."""
        self.model.train()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.meters.update("lr", self.lr_scheduler.get_last_lr()[0])

        metrics = {
            "loss": 0.0,
            "dpo_inside_term": 0.0,
            "dpo_diff_minus_ref": 0.0,
        }
        num_log = 0
        inside_min = None
        inside_max = None

        for item in self.pipeline.iterate_training_items(data):
            with torch.no_grad():
                ref_pred_w, ref_pred_l = self.pipeline.run_reference_on_pair(
                    ref_model=self.ref_model,
                    item=item,
                )
            model_pred_w, model_pred_l = self.pipeline.run_trainable_on_pair(
                model=self.model,
                item=item,
            )
            loss_out = diffusion_dpo_loss(
                model_pred_w=model_pred_w,
                model_pred_l=model_pred_l,
                ref_pred_w=ref_pred_w,
                ref_pred_l=ref_pred_l,
                target_w=item["target_w"],
                target_l=item["target_l"],
                beta_dpo=self.beta_dpo,
            )
            loss = loss_out["loss"]
            (loss * item["backward_scale"]).backward()

            loss_metrics = loss_out["metrics"]

            metrics["loss"] += loss_metrics["loss_mean"]
            metrics["dpo_inside_term"] += loss_metrics["inside_mean"]
            metrics["dpo_diff_minus_ref"] += loss_metrics["diff_minus_ref_mean"]
            cur_min = loss_metrics["inside_min"]
            cur_max = loss_metrics["inside_max"]
            inside_min = cur_min if inside_min is None else min(inside_min, cur_min)
            inside_max = cur_max if inside_max is None else max(inside_max, cur_max)
            num_log += 1

            if item["should_optimizer_step"]:
                grad_norm = self.model.clip_grad_norm_(self.max_grad_norm).item()
                self.meters.update("grad_norm", grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        if num_log == 0:
            raise ValueError("❌ No training items produced for DPO step.")

        for key in metrics:
            v = torch.tensor(
                metrics[key] / num_log, device=self.device, dtype=torch.float32
            )
            dist.all_reduce(v, op=dist.ReduceOp.AVG)
            self.meters.update(key, v.item())

        v_min = torch.tensor(inside_min, device=self.device, dtype=torch.float32)
        v_max = torch.tensor(inside_max, device=self.device, dtype=torch.float32)
        dist.all_reduce(v_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(v_max, op=dist.ReduceOp.MAX)
        self.meters.update("dpo_inside_term_min", v_min.item())
        self.meters.update("dpo_inside_term_max", v_max.item())

        dist.barrier()
        torch.cuda.empty_cache()
