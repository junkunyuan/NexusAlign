"""Flow-matching Euler discrete scheduler for image generation."""

import math
import numpy as np
from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class RLFlowMatchEulerDiscreteScheduler:
    """Lightweight scheduler used for RL training with flow-matching models."""

    def __init__(self, sample_eta: float) -> None:
        self.eta = sample_eta

    def step_fn(
        self,
        model_output: torch.Tensor,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        index: int | torch.Tensor,
        prev_sample: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the previous sample, clean sample, and log prob.

        Args:
            model_output (`torch.Tensor`): Model output.
            latents (`torch.Tensor`): Latents at the current timestep
            sigmas (`torch.Tensor`): Timestep sigmas.
            index (`int | torch.Tensor`): Timestep.
            prev_sample (`torch.Tensor | None`): Latents at the previous timestep.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]: 
                Previous sample, clean sample, and log prob.

        References:
            DanceGRPO: https://github.com/XueZeyue/DanceGRPO
        """
        device = model_output.device
        if not isinstance(index, int):
            sigmas = sigmas.to(device)

        sigma = sigmas[index].view(-1, 1, 1).to(device)
        dsigma = (sigmas[index + 1] - sigmas[index]).view(-1, 1, 1).to(device)

        prev_mean = latents + dsigma * model_output
        pred_ori_sample = latents - sigma * model_output

        delta_t = (sigmas[index] - sigmas[index + 1]).view(-1, 1, 1).to(device)
        std_dev_t = self.eta * torch.sqrt(delta_t)

        score = -(latents - pred_ori_sample * (1 - sigma)) / sigma**2
        prev_mean = prev_mean - 0.5 * self.eta**2 * score * dsigma

        if prev_sample is None:
            prev_sample = prev_mean + torch.randn_like(prev_mean) * std_dev_t

        # Log prob: log N(x;μ,σ2) = −(x−μ)**2/2σ**2 ​−logσ −1/2​*log(2π)
        prev_sample = prev_sample.detach().to(torch.float32)
        prev_mean = prev_mean.to(torch.float32)
        first_term = -((prev_sample - prev_mean) ** 2) / (2 * (std_dev_t**2))
        second_term = -torch.log(std_dev_t)
        third_term = -torch.log(torch.sqrt(2 * torch.as_tensor(torch.pi)))
        log_prob = first_term + second_term + third_term
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, pred_ori_sample, log_prob


# --------------------------------------------------------------------------------
# Flow-Matching Euler Discrete Scheduler
# --------------------------------------------------------------------------------
# A simplified version of the diffusers implementation for easy customization.
# NOTE: Some checks have been removed for simplicity, which may increase risk.
# --------------------------------------------------------------------------------
# Usage:
#     from nexus_align.pipelines.pipeline_flux import FlowMatchEulerDiscreteScheduler
#     (official) from diffusers import FlowMatchEulerDiscreteScheduler
# --------------------------------------------------------------------------------
@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    A simplified flow-matching Euler discrete scheduler.

    References:
        Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,  # number of diffusion steps to train the model
        shift: float = 1.0,  # shift value for the timestep schedule
        use_dynamic_shifting: bool = False,  # if true, apply timestep shifting on-the-fly based on the image resolution
        base_shift: (
            float | None
        ) = 0.5,  # increase it to reduce variation and image is more consistent with desired output
        max_shift: (
            float | None
        ) = 1.15,  # increase it to encourage more variation and image may be more exaggerated or stylized
        base_image_seq_len: int | None = 256,  # base image sequence length
        max_image_seq_len: int | None = 4096,  # maximum image sequence length
        invert_sigmas: bool = False,  # if true, invert the sigmas
        shift_terminal: (
            float | None
        ) = None,  # the end value of the shifted timestep schedule
        use_karras_sigmas: (
            bool | None
        ) = False,  # if true, use Karras sigmas for step sizes
        use_exponential_sigmas: (
            bool | None
        ) = False,  # if true, use exponential sigmas for step sizes
        use_beta_sigmas: bool | None = False,  # if true, use beta sigmas for step sizes
        time_shift_type: str = "exponential",  # the type of dynamic resolution-dependent shifting, "exponential" or "linear"
        stochastic_sampling: bool = False,  # if true, use stochastic sampling
    ):
        timesteps = np.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None
        self._shift = shift

    @property
    def shift(self):
        return self._shift

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        self._shift = shift

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        """Get the index for the given timestep."""
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def scale_noise(
        self,
        sample: torch.FloatTensor,  # input sample
        timestep: float | torch.FloatTensor,  # current timestep
        noise: torch.FloatTensor | None = None,  # sampled noise
    ) -> torch.FloatTensor:
        """Forward process in flow-matching."""
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in timestep
            ]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        """Stretches and shifts the timestep schedule to ensure it terminates at the `shift_terminal` value."""
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def set_timesteps(
        self,
        num_inference_steps: (
            int | None
        ) = None,  # number of diffusion steps to generate samples
        device: str | torch.device = None,  # the used device
        sigmas: list[float] | None = None,  # custom sigmas
        mu: (
            float | None
        ) = None,  # the amount of shifting applied to sigmas when performing resolution-dependent shifting
        timesteps: list[float] | None = None,  # custom timesteps
    ) -> None:
        """Sets the discrete timesteps used for the diffusion chain (to be run before inference)."""
        if num_inference_steps is None:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)
        self.num_inference_steps = num_inference_steps

        is_timesteps_provided = timesteps is not None
        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max),
                    self._sigma_to_t(self.sigma_min),
                    num_inference_steps,
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        # Perform timestep shifting
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # Stretch the sigmas schedule to terminate at the configured `shift_terminal` value
        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        # Convert sigmas to one of karras, exponential, or beta sigma schedules
        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(
                in_sigmas=sigmas, num_inference_steps=num_inference_steps
            )
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(
                in_sigmas=sigmas, num_inference_steps=num_inference_steps
            )
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(
                in_sigmas=sigmas, num_inference_steps=num_inference_steps
            )

        # Convert sigmas and timesteps
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(
                dtype=torch.float32, device=device
            )

        # Append the terminal sigma value.
        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def step(
        self,
        model_output: torch.FloatTensor,  # model output
        timestep: float | torch.FloatTensor,  # current timestep
        sample: torch.FloatTensor,  # current sample
        per_token_timesteps: (
            torch.Tensor | None
        ) = None,  # timesteps for each token in the sample
        return_dict: bool = True,  # if true, return Output class; else, return tuple
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple:
        """Predict the sample from the previous timestep by reversing the SDE."""
        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            sigma_next = lower_sigmas[..., None]
            dt = current_sigma - sigma_next
        else:
            sigma_idx = self.step_index
            current_sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]
            dt = sigma_next - current_sigma

        if self.config.stochastic_sampling:
            x0 = sample - current_sigma * model_output
            noise = torch.randn_like(sample)
            prev_sample = (1.0 - sigma_next) * x0 + sigma_next * noise
        else:
            prev_sample = sample + dt * model_output

        self._step_index += 1
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
