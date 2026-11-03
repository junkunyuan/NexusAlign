"""
ReFL algorithm: backpropagate a differentiable reward of one-step clean predictions.

References:
    - ReFL:
        - Paper: ImageReward: Learning and Evaluating Human Preferences for
          Text-to-Image Generation (NeurIPS 2023)
        - Official code: https://github.com/zai-org/ImageReward
            - ImageReward/ReFL.py: Trainer.train, for the random late step, the
              one-step clean prediction, the VAE decoding, and the reward loss.
              F.relu(-rewards+2) and --grad_scale 1e-3 set margin and reward_weight
    - DiffusionOPSD:
        - Paper: On-Policy Self-Distillation in Diffusion Models (arXiv 2026)
        - Code: https://github.com/worldbench/DiffusionOPSD
            - config/refl.py: _refl_block, for late_fraction 0.25 (train_step_range)
              and calibration_num_prompts 512 (calibration_prompts)
            - scripts/train_refl_sd35m.py: _calibrate and main, for the reward
              standardization with the base model, and the hinge on it
"""

import numpy as np
import torch
import torch.distributed as dist

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.engine.distributed import gather_tensor
from nexus_align.algorithms.diffusion_nft import DiffusionNFT
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


class ReFL(BaseAlgorithm):
    """ReFL training algorithm, with the samplers of DiffusionNFT."""

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Sampling configs.
        self.resolution = cfg_algorithm.get("resolution", 512)  # image resolution
        self.num_steps = cfg_algorithm.get("num_steps", 10)  # rollout sampling steps
        self.max_sequence_length = cfg_algorithm.get("max_sequence_length", 128)  # T5 prompt tokens
        self.autocast_dtype = cfg_algorithm.get("autocast_dtype", "fp16")  # transformer and VAE compute dtype
        self.group_size = cfg_algorithm.get("group_size", 24)  # images per prompt; ReFL needs no groups and uses DiffusionNFT's value

        # Reward loss configs.
        # [7, 10) of 10 steps are the last 25% of the steps, as in DiffusionOPSD.
        self.train_step_range = cfg_algorithm.get("train_step_range", [7, 10])  # train each image at a random sampling step in [low, high); see paper Algorithm 1
        self.margin = cfg_algorithm.get("margin", 2.0)  # margin of the loss phi(r) = relu(margin - r) on the standardized reward
        self.calibration_prompts = cfg_algorithm.get("calibration_prompts", 512)  # first training prompts that set the reward mean and std
        self.reward_weight = cfg_algorithm.get("reward_weight", 1.0e-3)  # reward loss weight lambda; see paper Eq. 2

        self.pipeline = None
        self.scheduler = None
        self.reward = None  # differentiable reward, only for training
        self.num_micro_steps = 1  # forward passes per training batch
        self.reward_mean = None  # reward mean and std of the base model, from calibrate
        self.reward_std = None

    def prepare(self, model, device, pipeline, reward=None) -> None:
        """Bind the transformer and device, the pipeline, and the training reward."""
        super().prepare(model, device)
        self.pipeline = pipeline
        self.scheduler = pipeline.scheduler
        # Recompute the VAE decoder blocks in backward. No-grad decoding stays the same.
        pipeline.vae.enable_gradient_checkpointing()
        if reward is not None and not hasattr(reward, "score_grad"):
            raise ValueError(f"❌ ReFL needs a differentiable reward, not {reward.name}")
        self.reward = reward

    @torch.no_grad()
    def calibrate(self, prompts: list[str], batch_size: int) -> None:
        """
        Set the reward mean and std of the base model, as DiffusionOPSD does.

        The ranks share the prompts, and roll out each prompt once. The reward scores
        the one-step prediction at a random late state of each rollout.
        """
        world_size = dist.get_world_size()
        if len(prompts) % world_size:
            raise ValueError(
                f"❌ calibration_prompts ({len(prompts)}) must be a multiple of the "
                f"world size ({world_size})"
            )
        prompts = prompts[dist.get_rank()::world_size]
        rewards = [
            self.score(self.rollout(prompts[i:i + batch_size], 0)["samples"])
            for i in range(0, len(prompts), batch_size)
        ]
        rewards = gather_tensor(torch.cat(rewards))
        self.reward_mean, self.reward_std = rewards.mean().item(), rewards.std().item()
        print(
            f"📊 Reward of the base model on {len(rewards)} prompts: "
            f"mean {self.reward_mean:.4f}, std {self.reward_std:.4f}"
        )

    def score(self, data: dict) -> torch.Tensor:
        """Reward the one-step predictions at the late states, with gradients."""
        # 1. Predict the clean latents in one step, at the integer timesteps that the
        # sampler used.
        latents, sigmas = data["latents"], data["sigmas"]
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            velocity = self.model(
                hidden_states=latents,
                timestep=(sigmas * 1000).long(),
                encoder_hidden_states=data["prompt_embeds"],
                pooled_projections=data["pooled_prompt_embeds"],
                return_dict=False,
            )[0]
        x0 = latents.float() - sigmas.view(-1, 1, 1, 1) * velocity.float()

        # 2. Reward the decoded images. Gradients flow back through the VAE and reward.
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            images = self.pipeline.decode(x0)
        return self.reward.score_grad(images.float(), data["reward_prompt_embeds"])

    def forward(self, data):
        """Calculate the ReFL loss of one batch at its late states; see paper Eq. 2."""
        # Standardize the rewards with the base model, so that the margin fits any
        # reward scale, as DiffusionOPSD does.
        rewards = self.score(data)
        z = (rewards - self.reward_mean) / self.reward_std
        loss = self.reward_weight * torch.relu(self.margin - z).mean()
        metrics = {
            "pred_reward": rewards.mean().detach(),
            "hinge_active": (z.detach() < self.margin).float().mean(),
            "loss": loss.detach(),
        }
        return {"loss": loss, "metrics": metrics}

    # Sample as DiffusionNFT does, so both train on the same CFG-free dpm2 rollouts and
    # use the same Euler samples for evaluation.
    predict = DiffusionNFT.predict
    sample = DiffusionNFT.sample
    decode = DiffusionNFT.decode
    generate = DiffusionNFT.generate

    @torch.no_grad()
    def rollout(self, prompts: list[str], epoch: int) -> dict:
        """Sample images. Return them and a random late state of each to train on."""
        max_length = self.max_sequence_length
        embeds, pooled_embeds = self.pipeline.encode_prompt(prompts, max_length)
        out = self.sample({
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "solver": "dpm2",
        })

        # Each image trains at one random late step; see paper Algorithm 1.
        batch_size = len(prompts)
        steps = torch.randint(*self.train_step_range, (batch_size,), device=self.device)
        rows = torch.arange(batch_size, device=self.device)
        samples = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "reward_prompt_embeds": self.reward.embed_prompts(prompts),
            "sigmas": self.scheduler.sigmas.float()[steps],  # their noise levels
            "latents": out["latents"][rows, steps],  # states to denoise with gradients
        }
        return {"samples": samples, "responses": self.decode(out["latents"][:, -1])}

    def compute_advantages(self, prompts: list[str], rewards: np.ndarray) -> np.ndarray:
        """Return zeros that keep the layout, because ReFL backpropagates the reward."""
        return np.zeros_like(rewards)

    def state_dict(self) -> dict:
        """Return the reward calibration, so that a resumed run keeps it."""
        if self.reward_std is None:
            return {}
        return {"reward_mean": self.reward_mean, "reward_std": self.reward_std}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the reward calibration, if the checkpoint has one."""
        self.reward_mean = state_dict.get("reward_mean")
        self.reward_std = state_dict.get("reward_std")
